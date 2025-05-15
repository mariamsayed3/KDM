import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from PIL import Image


class SpatialTranscriptomicsDataset(Dataset):
    def __init__(self, img_paths, seg_paths=None, 
                 cutting=None, transform=None,
                 channels=None, outtype='3d', envi_type='img',
                 multi_class=1, classes=[1, 2, 3, 4, 5], ignore_index=0):
        """
        Initialize Spatial Transcriptomics Dataset
        
        Args:
            img_paths: List of paths to gene expression matrix files (.npy)
            seg_paths: List of paths to segmentation mask files (.npy)
            cutting: Tuple for spatial cutting (height, width), default None
            transform: Data augmentation transforms, default None
            channels: Number of gene channels to use, default None (use all)
            outtype: Output type ('3d' or '2d'), default '3d'
            envi_type: Environment type, default 'img'
            multi_class: Multi-class segmentation flag, default 1
            classes: Active classes (without background/ignore class) [1, 2, 3, 4, 5]
            ignore_index: Index to ignore in loss computation (default: 0)
        """
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.classes = classes  # Only active classes [1, 2, 3, 4, 5]
        self.ignore_index = ignore_index
        self.transform = transform
        self.cutting = cutting if cutting is not None else (32, 32)  # Default to 32x32
        self.channels = channels
        self.outtype = outtype
        self.envi_type = envi_type
        self.multi_class = multi_class
        
        # Include ignore index for one-hot encoding but not for active training
        self.all_classes = [ignore_index] + classes  # [0, 1, 2, 3, 4, 5]
        
        # Verify that all files exist
        self._verify_files()
        
        print(f"Dataset initialized with {len(self.img_paths)} samples")
        print(f"Active classes: {self.classes}")
        print(f"All classes (including ignore): {self.all_classes}")
        print(f"Cutting size: {self.cutting}")
    
    def _verify_files(self):
        """Verify that all specified files exist"""
        missing_files = []
        
        for img_path in self.img_paths:
            if not os.path.exists(img_path):
                missing_files.append(img_path)
        
        if self.seg_paths:
            for seg_path in self.seg_paths:
                if not os.path.exists(seg_path):
                    missing_files.append(seg_path)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} files not found:")
            for file in missing_files[:5]:  # Show first 5 missing files
                print(f"  {file}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")
    
    def __getitem__(self, index):
        """
        Get a single sample from the dataset
        
        Returns:
            dict: {
                'input': torch.Tensor (C, H, W) - Gene expression data
                'ground_truth_seg': torch.Tensor (1, H, W) - Segmentation mask  
                'ground_truth_onehot': torch.Tensor (num_classes, H, W) - One-hot mask
                'name': str - Sample name
            }
        """
        img_path = self.img_paths[index]
        
        # Load gene expression data
        img = np.load(img_path)
               
        
        # Load segmentation mask if available
        if self.seg_paths and index < len(self.seg_paths):
            mask_path = self.seg_paths[index]
            mask = np.load(mask_path)
        else:
            # Create dummy mask if no segmentation is provided
            mask = np.zeros(img.shape[:2], dtype=np.int64)
        
        # Handle different input shapes
        if len(img.shape) == 3:
            # Shape: [H, W, C] -> need to transpose to [C, H, W]
            if img.shape[2] > img.shape[0]:  # More channels than height
                h, w, c = img.shape
                img = np.transpose(img, (2, 0, 1))  # [C, H, W]
            else:
                # Already in [C, H, W] format
                c, h, w = img.shape
        elif len(img.shape) == 2:
            # Single channel: [H, W] -> [1, H, W]
            img = img[np.newaxis, ...]
            c, h, w = img.shape
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        # Apply spatial cutting if needed
        if self.cutting:
            cut_h, cut_w = self.cutting
            # Take from top-left corner
            img = img[:, :cut_h, :cut_w]
            mask = mask[:cut_h, :cut_w]
            h, w = cut_h, cut_w
        
        # Select specific channels if specified
        if self.channels is not None:
            img = img[:self.channels, :, :]
            c = self.channels
        
        # Ensure float32 for gene expression data
        img = img.astype(np.float32)
        
        # Clean and process mask
        mask = self._process_mask(mask)
        
        # Create one-hot encoding for ALL classes (including ignore_index)
        mask_onehot = self._create_onehot(mask, h, w)
        
        # Apply transforms if specified
        if self.transform:
            # Convert to PIL or apply transforms as needed
            # This would need to be customized based on your transform requirements
            pass
        
        # Convert to torch tensors
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask).long()
        mask_onehot_tensor = torch.from_numpy(mask_onehot).float()
        
        # Get sample name from file path
        sample_name = os.path.basename(img_path).replace('.npy', '')
        
        return {
            'input': img_tensor,                           # Shape: (C, H, W)
            'ground_truth_seg': mask_tensor.unsqueeze(0),  # Shape: (1, H, W)
            'ground_truth_onehot': mask_onehot_tensor,     # Shape: (num_classes, H, W)
            'name': sample_name
        }
    
    def _process_mask(self, mask):
        """
        Process segmentation mask to handle ignore_index properly
        
        Args:
            mask: Raw segmentation mask
            
        Returns:
            Processed mask with proper class handling
        """
        # Ensure mask is integer type
        mask = mask.astype(np.int64)
        
        # Handle case where mask has background as 0
        # Map background (0) to ignore_index if they're different
        if self.ignore_index != 0:
            mask[mask == 0] = self.ignore_index
        
        # Ensure only valid classes are present
        # Set any invalid classes to ignore_index
        valid_mask = np.zeros_like(mask, dtype=bool)
        for class_id in self.all_classes:
            valid_mask |= (mask == class_id)
        
        mask[~valid_mask] = self.ignore_index
        
        return mask
    
    def _create_onehot(self, mask, h, w):
        """
        Create one-hot encoding for all classes including ignore_index
        
        Args:
            mask: Processed segmentation mask
            h, w: Height and width of the mask
            
        Returns:
            One-hot encoded mask
        """
        mask_onehot = np.zeros((len(self.all_classes), h, w), dtype=np.float32)
        
        for i, class_label in enumerate(self.all_classes):
            mask_onehot[i, mask == class_label] = 1.0
        
        return mask_onehot
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.img_paths)
    
    def save_pred(self, pred_tensor, sv_path, name):
        """
        Save prediction visualization
        
        Args:
            pred_tensor: Prediction tensor (can be logits or probabilities)
            sv_path: Directory to save the visualization
            name: Filename (without extension)
        """
        # Convert tensor to numpy if needed
        if torch.is_tensor(pred_tensor):
            pred_np = pred_tensor.detach().cpu().numpy()
        else:
            pred_np = pred_tensor.copy()
        
        # Handle different input shapes
        if len(pred_np.shape) == 4:  # Batch dimension
            pred_np = pred_np[0]
        
        if len(pred_np.shape) == 3:  # Multi-class predictions [C, H, W]
            pred_np = np.argmax(pred_np, axis=0)
        
        # Ensure the prediction is in valid range
        pred_np = np.clip(pred_np, 0, len(self.all_classes) - 1).astype(np.uint8)
        
        # Define colors for each class
        color_map_by_label = {
            0: [119, 158, 203],  # Background/ignore - blue-gray
            1: [124, 252, 0],    # Class 1 - bright green
            2: [155, 118, 83],   # Class 2 - brown
            3: [255, 0, 0],      # Class 3 - red
            4: [255, 255, 0],    # Class 4 - yellow
            5: [0, 255, 255]     # Class 5 - cyan
        }
        
        # Create color palette for PIL
        palette = []
        for i in range(256):
            if i in color_map_by_label:
                palette.extend(color_map_by_label[i])
            else:
                palette.extend([0, 0, 0])  # Black for undefined classes
        
        # Create directory if it doesn't exist
        os.makedirs(sv_path, exist_ok=True)
        
        # Convert to PIL image and apply color palette
        save_img = Image.fromarray(pred_np)
        save_img.putpalette(palette)
        
        # Save the image
        save_path = os.path.join(sv_path, f'{name}.png')
        save_img.save(save_path)
        
        print(f"Saved prediction to {save_path}")
    
    def get_sample_info(self, index):
        """
        Get information about a specific sample
        
        Args:
            index: Sample index
            
        Returns:
            dict: Information about the sample
        """
        sample = self[index]
        
        # Get unique classes in this sample
        mask = sample['ground_truth_seg'].squeeze().numpy()
        unique_classes = np.unique(mask)
        active_classes = [c for c in unique_classes if c != self.ignore_index]
        
        return {
            'name': sample['name'],
            'input_shape': sample['input'].shape,
            'mask_shape': sample['ground_truth_seg'].shape,
            'unique_classes': unique_classes.tolist(),
            'active_classes': active_classes,
            'num_active_classes': len(active_classes),
            'ignore_pixels': np.sum(mask == self.ignore_index),
            'total_pixels': mask.size
        }
    
    def get_class_distribution(self):
        """
        Get the distribution of classes across the entire dataset
        
        Returns:
            dict: Class distribution information
        """
        class_counts = {cls: 0 for cls in self.all_classes}
        sample_class_info = []
        
        for i in range(len(self)):
            sample_info = self.get_sample_info(i)
            sample_class_info.append(sample_info)
            
            # Count pixels for each class
            sample = self[i]
            mask = sample['ground_truth_seg'].squeeze().numpy()
            unique, counts = np.unique(mask, return_counts=True)
            
            for cls, count in zip(unique, counts):
                if cls in class_counts:
                    class_counts[cls] += count
        
        return {
            'class_counts': class_counts,
            'class_frequencies': {
                cls: count / sum(class_counts.values()) 
                for cls, count in class_counts.items()
            },
            'samples_per_class': {
                cls: len([s for s in sample_class_info if cls in s['active_classes']])
                for cls in self.classes
            },
            'sample_info': sample_class_info
        }
def get_spatial_transcriptomics_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create dataloaders for spatial transcriptomics data using dataset_divide.json
    """
    print(f"\n{'='*60}")
    print(f"LOADING SPATIAL TRANSCRIPTOMICS DATA")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    # Load dataset division JSON
    dataset_divide_path = os.path.join(data_dir, 'dataset_divide.json')
    
    if not os.path.exists(dataset_divide_path):
        print(f"❌ dataset_divide.json not found at {dataset_divide_path}")
        print("Please create it first using the create_dataset_division.py script")
        raise FileNotFoundError(f"dataset_divide.json not found. Run create_dataset_division.py first.")
    
    # Load the JSON file
    print(f"📁 Loading dataset split from {dataset_divide_path}")
    with open(dataset_divide_path, 'r') as f:
        dataset_dict = json.load(f)
    
    # Extract splits
    train_indices = dataset_dict.get('train', [])
    val_indices = dataset_dict.get('val', [])
    test_indices = dataset_dict.get('test', [])
    
    print(f"Loaded splits:")
    print(f"Train: {len(train_indices)} samples")
    print(f"Val: {len(val_indices)} samples")  
    print(f"Test: {len(test_indices)} samples")
    
    # Verify indices have corresponding files
    def verify_and_create_paths(indices, split_name):
        valid_img_paths = []
        valid_mask_paths = []
        
        for idx in indices:
            gene_file = os.path.join(data_dir, f"gene_expre_matrix_{idx}.npy")
            label_file = os.path.join(data_dir, f"label_matrix_{idx}.npy")
            
            if os.path.exists(gene_file) and os.path.exists(label_file):
                valid_img_paths.append(gene_file)
                valid_mask_paths.append(label_file)
            else:
                print(f"Warning: Missing files for {split_name} index {idx}")
        
        print(f"  {split_name}: {len(valid_img_paths)}/{len(indices)} files found")
        return valid_img_paths, valid_mask_paths
    
    # Create file paths for each split
    train_img_paths, train_mask_paths = verify_and_create_paths(train_indices, "Train")
    val_img_paths, val_mask_paths = verify_and_create_paths(val_indices, "Val")
    test_img_paths, test_mask_paths = verify_and_create_paths(test_indices, "Test")
    
    # Check if we have data
    if len(train_img_paths) == 0:
        raise ValueError("No training files found! Check your dataset_divide.json and file paths.")
    
    # Create datasets
    print(f"\n📦 Creating datasets...")
    train_dataset = SpatialTranscriptomicsDataset(
        img_paths=train_img_paths,
        seg_paths=train_mask_paths,
        classes=[1, 2, 3, 4, 5],
        ignore_index=0,
        cutting=(32, 32)
    )
    
    val_dataset = SpatialTranscriptomicsDataset(
        img_paths=val_img_paths,
        seg_paths=val_mask_paths,
        classes=[1, 2, 3, 4, 5],
        ignore_index=0,
        cutting=(32, 32)
    )
    
    test_dataset = SpatialTranscriptomicsDataset(
        img_paths=test_img_paths,
        seg_paths=test_mask_paths,
        classes=[1, 2, 3, 4, 5],
        ignore_index=0,
        cutting=(32, 32)
    )
    
    print(f"Dataset lengths:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Verify datasets are not empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after creation!")
    
    # Test loading one sample
    try:
        sample = train_dataset[0]
        print(f"✅ Sample test successful - Input shape: {sample['input'].shape}")
    except Exception as e:
        print(f"❌ Error loading sample: {e}")
        raise
    
    # Create dataloaders
    print(f"\n🚀 Creating dataloaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ Dataloaders created successfully!")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader
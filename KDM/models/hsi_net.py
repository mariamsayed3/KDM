# -*- coding: utf-8 -*-
"""
Created on 14/08/2020 5:36 pm

@author: Soan Duong, UOW
"""
# Standard library imports
from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
import pprint
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.heads import SegmentationHead
import numpy as np

DEBUG = False

# Third party imports
import torch.nn as nn
import torch
from .modules import *

# Local application imports

class NormLayer(nn.Module):
    """
    - [NormLayer] is a class that can pad the input N-D tensor so that the size
     in each dimension can be divisible by a divisible_number (default = 16).

    - This class can also crop the input N-D tensor given a padding size.

    Note that the format of the input data: (batch, n_channels, 1st_size, ..., nth_size)
    """

    def __init__(self, mode="pad", divisible_number=16, max_norm_val=None):
        """
        :param mode: "pad" or "crop"
        """
        super(NormLayer, self).__init__()
        self.mode = mode
        self.divisible_number = divisible_number
        self.max_norm_val = max_norm_val
        self.padding = []

    def forward(self, x, padding=None):
        # Normalize the x into [0, self.max_val]
        if self.max_norm_val is not None:
            print(self.max_norm_val)
            x = norm_intensity(x, max_val=self.max_norm_val)

        if self.mode == "pad":  # do padding
            # Compute the padding amount so that each dimension is divisible by divisible_number
            padding = compute_padding(x, divisible_number=self.divisible_number)
            self.padding = padding

            # Apply padding
            x = nn.functional.pad(x, padding)

            return x

        else:  # do cropping
            # Cropping by padding with minus amount of padding
            padding = tuple([-p for p in padding])
            # print('cropping', padding)

            # Apply cropping
            x = nn.functional.pad(x, padding)

            return x


# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------

# 2. FIX THE norm_intensity FUNCTION:
def norm_intensity(x, max_val=1):
    """
    This function normalizes the intensity of the input tensor x into [0, max_val]
    :param x: input tensor
    :param max_val: maximum value
    :return: a normalized tensor
    """
    if max_val is None:
        max_val = 1

    # FIXED: Changed 'input' to 'x'
    if len(list(torch.unique(x).size())) != 1:  # avoid the case that the tensor contains only one value
        x = x - torch.min(x)

    x = x / torch.max(x) * max_val
    return x


def compute_padding(x, divisible_number=16):
    """
    Computes the padding for each spatial dim (exclude depth) so that it is divisible by divisible_number
    :param x: N-D tensor with the data format
    :param divisible_number:
    :return: padding value at each dimension, e.g. 2D->(d2_p1, d2_p2, d1_p1, d1_p2)
    """
    padding = []
    input_shape_list = x.size()

    # Reversed because pytorch's pad() receive in a reversed order
    for org_size in reversed(input_shape_list[2:]):
        # compute the padding amount in two sides
        p = int((int((org_size - 1) / divisible_number) + 1) * divisible_number - org_size)
        # padding amount in one size
        p1 = int(p / 2)
        padding.append(p1)
        padding.append(p - p1)

    return tuple(padding)


def conv_block(c_in, c_out, stride=1, kernel_size=3, negative_slope=0.2,
               do_upsample=False, do_batchnorm=False):
    """
    Creates a convolutional building block: Conv + LeakyReLU + Upsample (optional) + BatchNorm (optional)

    :param c_in: input channel size
    :param c_out: output channel size
    :param kernel_size: filter size of the conv layer
    :param stride: stride of the convolutional layer
    :param negative_slope: the parameter that controls the angle of the negative slope of the LeakyReLU layer
    :param do_upsample: a boolean param indicating whether an upsample layer is added after the (Conv + LeakyReLU)
    :param do_batchnorm: a boolean param indicating whether an upsample layer is added at the end of the block
    :return: a convolutional building block
    """

    block = nn.ModuleList()
    # compute the padding amount
    padding = int(np.ceil(((kernel_size - 1) + 1 - stride) / 2.))

    block.append(nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding))
    block.append(nn.LeakyReLU(negative_slope))

    # append an Upsample layer if it is required
    if do_upsample:
        block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    # append an BatchNormalization layer if it is required
    if do_batchnorm:
        block.append(nn.BatchNorm2d(c_out))

    return nn.Sequential(*block)

class HSINet(nn.Module):
    def __init__(self, classes=[0, 1, 2, 3, 4,5], n_bands=136, nf_enc=[32, 64, 64, 64], nf_dec=[64, 64, 32, 32, 16, 8],
                 do_batchnorm=True, max_norm_val=None):
        """
        Initialize a hsi_net instance
        :param classes: list of classes in the segmentation problem
        :param n_bands: number of bands in the hsi images
        :param nf_enc: list of number of filters for the encoder
        :param nf_dec: list of number of filters for the decoder
        :param do_batchnorm: boolean to implement batchnorm after every convolution layer
        :param max_norm_val: maximum value for normalization
        """
        super(HSINet, self).__init__()

        self.classes = classes
        self.n_bands = n_bands
        self.nf_enc = nf_enc
        self.nf_dec = nf_dec
        self.do_batchnorm = do_batchnorm
        self.max_norm_val = max_norm_val

        self.norm_input = NormLayer(mode='pad', max_norm_val=max_norm_val,
                                    divisible_number=2 ** len(nf_enc))
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.post_layers = nn.ModuleList()
        self.recovery_size = NormLayer(mode='crop')
        self.softmax = nn.Softmax2d()

        # Define the encoder
        for (idx, n_filters) in enumerate(nf_enc):
            if idx == 0:  # first filter
                c_in = self.n_bands
            else:
                c_in = nf_enc[idx - 1]
            self.encoder.append(conv_block(c_in, n_filters, stride=2,
                                           do_batchnorm=do_batchnorm))

        # Define the decoder
        n_encoder_layers = len(nf_enc)
        n_decoder_layers = len(nf_dec)
        for idx in range(n_encoder_layers):
            if idx == 0:
                c_in = nf_enc[-1]
            else:
                c_in = nf_dec[idx - 1] + nf_enc[n_encoder_layers - idx - 1]
            self.decoder.append(conv_block(c_in, nf_dec[idx],
                                           do_upsample=True, do_batchnorm=do_batchnorm))

        # Define the post_layer
        for idx in range(n_encoder_layers, n_decoder_layers):
            c_in = nf_dec[idx - 1]
            if idx == n_decoder_layers - 1:
                c_in = c_in + self.n_bands
            self.post_layers.append(conv_block(c_in, nf_dec[idx],
                                               do_upsample=False, do_batchnorm=do_batchnorm))

        # Append a conv_block that output the segmentation image
        self.post_layers.append(conv_block(nf_dec[-1], len(self.classes),
                                           do_upsample=False, do_batchnorm=do_batchnorm))

    def forward(self, x):
        # Normalize the input
        x = self.norm_input(x)
        input_org = x  # will be used by last post_layer via skip-connect

        # Add layers in the encoder (E1-4)
        enc_outputs = []
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)  # store encoder output

        # Add layers in the decoder (D1-4)
        n_encoder_layers = len(self.nf_enc)
        for (idx, layer) in enumerate(self.decoder):
            # concatenate the output with the corresponding encoder layer if it is required
            if idx != 0:  # concatenate on the channel dimension
                x = torch.cat((x, enc_outputs[n_encoder_layers - idx - 1]), 1)

            # Add the decoder layer
            x = layer(x)


        # Add layers after the decoder (D5 and D6)
        for (idx, layer) in enumerate(self.post_layers):
            # Concatenate the input of second last layer with the original
            if idx == (len(self.post_layers) - 2):
                x = torch.cat((x, input_org), 1)

            # Add the layer
            x = layer(x)
            if idx == len(self.post_layers) - 2:
                f = x.clone()

        # Recovery the size of the input
        x = self.recovery_size(x, self.norm_input.padding)

        return f, x


class Res_SGR_Net(nn.Module):
    def __init__(self, classes=[0, 1, 2, 3, 4,5], n_bands=136, nf_enc=[32, 64, 64, 64], nf_dec=[64, 64, 32, 32, 16, 8],
                 do_batchnorm=True, max_norm_val=None, n_heads=3, encoder_name='resnet50'):
        """
        Initialize a hsi_net instance
        :param classes: list of classes in the segmentation problem
        :param n_bands: number of bands in the hsi images
        :param nf_enc: list of number of filters for the encoder
        :param nf_dec: list of number of filters for the decoder
        :param do_batchnorm: boolean to implement batchnorm after every convolution layer
        :param max_norm_val: maximum value for normalization
        """
        super(Res_SGR_Net, self).__init__()

        self.classes = classes
        self.n_bands = n_bands
        self.nf_enc = nf_enc
        self.nf_dec = nf_dec
        self.do_batchnorm = do_batchnorm
        self.max_norm_val = max_norm_val
        self.n_heads = n_heads

        self.norm_input = NormLayer(mode='pad', max_norm_val=max_norm_val,
                                    divisible_number=2 ** len(nf_enc))
        self.encoder = get_encoder(encoder_name,in_channels=n_bands,depth=len(nf_enc),weights='imagenet')
        self.decoder = nn.ModuleList()
        self.post_layers = nn.ModuleList()
        self.recovery_size = NormLayer(mode='crop')
        self.softmax = nn.Softmax2d()
        self.feature_learning = nn.ModuleList()
        self.heads = nn.ModuleList()
        attention_module = AttentionOCR
        # Define the encoder
        # self.skip_conns = nn.ModuleList()
        # for (idx, n_filters) in enumerate(nf_enc):
        #     if idx == 0:  # first filter
        #         c_in = self.n_bands
        #     else:
        #         c_in = nf_enc[idx - 1]
        #     # self.encoder.append(conv_block(c_in, n_filters, stride=2,
        #     #                                do_batchnorm=do_batchnorm))
        #     #block = DilationBottleNeck(in_planes=c_in, planes=n_filters, dropout_rate=0.4, stride=2)
        #     if idx < len(nf_enc) - 1:
        #         dec_idx = len(nf_enc) - idx - 1
        #         skip = SemanticGuidingSpectralAttention(nf_enc[idx],nf_dec[dec_idx],rates=rates)
        #         self.skip_conns.append(skip)

        # Define the decoder
        n_encoder_layers = len(nf_enc)
        n_decoder_layers = len(nf_dec)
        start_aux = n_decoder_layers - n_heads
        self.start_aux = start_aux
        for idx in range(n_encoder_layers):
            if idx == 0:
                c_in = nf_enc[-1]
            else:
                c_in = nf_dec[idx - 1] + nf_enc[n_encoder_layers - idx - 1]
            #block = DilationBottleNeck(c_in, nf_dec[idx], dropout_rate=0.4, stride=1, do_upsample=True)
            block = conv_block(c_in, nf_dec[idx],do_upsample=True, do_batchnorm=do_batchnorm)
            self.decoder.append(block)
            # self.decoder.append(conv_block(c_in, nf_dec[idx],
            #                                do_upsample=True, do_batchnorm=do_batchnorm))
            hidden_channels = 16
            # Add aux head
            if idx == start_aux:
                self.heads.append(conv_block(nf_dec[idx], len(classes), do_upsample=False, do_batchnorm=do_batchnorm))
            if idx > start_aux:
                self.feature_learning.append(attention_module(num_classes=len(classes),
                                                          in_channels=nf_dec[idx], hidden_channels=hidden_channels,
                                                          out_channels=nf_dec[idx],
                                                          dropout=0.1
                                                          ))
                self.heads.append(conv_block(nf_dec[idx],len(classes),do_upsample=False, do_batchnorm=do_batchnorm))


        # Define the post_layer
        for idx in range(n_encoder_layers, n_decoder_layers):
            c_in = nf_dec[idx - 1]
            if idx == n_decoder_layers - 2:
                c_in = c_in + self.n_bands
            #block = DilationBottleNeck(c_in,nf_dec[idx],dropout_rate=0.4,stride=1,do_upsample=False)
            block = conv_block(c_in, nf_dec[idx],do_upsample=False, do_batchnorm=do_batchnorm)
            self.post_layers.append(block)
            # Add aux head
            if idx == start_aux:
                self.heads.append(conv_block(nf_dec[idx], len(classes), do_upsample=False, do_batchnorm=do_batchnorm))
            if idx > start_aux:
                self.feature_learning.append(AttentionOCR(len(classes),
                                                          in_channels=nf_dec[idx], hidden_channels=hidden_channels,
                                                          out_channels=nf_dec[idx],
                                                          scale=1,
                                                          dropout=0.1
                                                          ))
                self.heads.append(conv_block(nf_dec[idx], len(classes), do_upsample=False, do_batchnorm=do_batchnorm))

        # Append a conv_block that output the segmentation image

    def forward(self, x, early_exit=None):
        size = (x.size(2), x.size(3))
        # Normalize the input
        x = self.norm_input(x)
        input_org = x  # will be used by last post_layer via skip-connect

        # Add layers in the encoder (E1-4)
        enc_outputs = self.encoder(x)
        enc_outputs = enc_outputs[1:]
        x = enc_outputs[-1]
        f_results, o_results = [], []
        # for layer in self.encoder:
        #     x = layer(x)
        #     enc_outputs.append(x)  # store encoder output
        #enc_outputs =

        # Add layers in the decoder (D1-4)
        n_encoder_layers = len(self.nf_enc)
        cnt = 0
        for (idx, layer) in enumerate(self.decoder):
            # concatenate the output with the corresponding encoder layer if it is required
            if idx != 0:  # concatenate on the channel dimension
                enc_output = enc_outputs[n_encoder_layers - idx - 1]
                #enc_output = self.skip_conns[-idx](x_dec=x, x_enc=enc_output)
                x = torch.cat((x, enc_output), 1)

            # Add the decoder layer
            x = layer(x)
            # Student sub-networks
            if idx == self.start_aux:
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                o_results.append(F.interpolate(o, size))

            elif idx > self.start_aux:
                feat_size = (x.size(2), x.size(3))
                x, a = self.feature_learning[cnt](x,F.interpolate(o,feat_size))
                cnt += 1
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                o_results.append(F.interpolate(o, size))
                #o_results.append(o)
                f_results.append(x)
            # if early_exit == cnt and idx >= self.start_aux:
            #     return o

        # Add layers after the decoder (D5 and D6)
        for (idx, layer) in enumerate(self.post_layers):
            # Concatenate the input of second last layer with the original
            if idx == (len(self.post_layers) - 2):
                x = torch.cat((x, input_org), 1)

            # Add the layer
            x = layer(x)
            if idx + n_encoder_layers == self.start_aux:
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                o_results.append(F.interpolate(o, size))
            elif idx + n_encoder_layers > self.start_aux:
                x, a = self.feature_learning[cnt](x, o)
                cnt += 1
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                # if idx == len(self.post_layers):
                #     o = self.recovery_size(o, self.norm_input.padding)
                #if idx == len(self.post_layers)-1:
                #o_results.append(F.interpolate(o, size))
                o_results.append(o)
                f_results.append(x)
            # if early_exit == cnt and idx + n_encoder_layers >= self.start_aux:
            #     return o

        # # Recovery the size of the input
        # x = self.recovery_size(x, self.norm_input.padding)
        return f_results, o_results


class Teacher_Res_SGR_Net(nn.Module):
    def __init__(self, classes=[0, 1, 2, 3, 4,5], n_bands=136, nf_enc=[32, 64, 64, 64], nf_dec=[64, 64, 32, 32, 16, 8],
                 do_batchnorm=True, max_norm_val=None, n_heads=3, encoder_name='resnet50'):
        """
        Initialize a hsi_net instance
        :param classes: list of classes in the segmentation problem
        :param n_bands: number of bands in the hsi images
        :param nf_enc: list of number of filters for the encoder
        :param nf_dec: list of number of filters for the decoder
        :param do_batchnorm: boolean to implement batchnorm after every convolution layer
        :param max_norm_val: maximum value for normalization
        """
        super(Teacher_Res_SGR_Net, self).__init__()

        self.classes = classes
        self.n_bands = n_bands
        self.nf_enc = nf_enc
        self.nf_dec = nf_dec
        self.do_batchnorm = do_batchnorm
        self.max_norm_val = max_norm_val
        self.n_heads = n_heads

        self.norm_input = NormLayer(mode='pad', max_norm_val=max_norm_val,
                                    divisible_number=2 ** len(nf_enc))
        self.encoder = get_encoder(encoder_name,in_channels=n_bands,depth=len(nf_enc))
        self.decoder = nn.ModuleList()
        self.post_layers = nn.ModuleList()
        self.recovery_size = NormLayer(mode='crop')
        self.softmax = nn.Softmax2d()
        self.feature_learning = nn.ModuleList()
        self.heads = nn.ModuleList()
        attention_module = AttentionOCR
        # Define the encoder
        # self.skip_conns = nn.ModuleList()
        # for (idx, n_filters) in enumerate(nf_enc):
        #     if idx == 0:  # first filter
        #         c_in = self.n_bands
        #     else:
        #         c_in = nf_enc[idx - 1]
        #     # self.encoder.append(conv_block(c_in, n_filters, stride=2,
        #     #                                do_batchnorm=do_batchnorm))
        #     #block = DilationBottleNeck(in_planes=c_in, planes=n_filters, dropout_rate=0.4, stride=2)
        #     if idx < len(nf_enc) - 1:
        #         dec_idx = len(nf_enc) - idx - 1
        #         skip = SemanticGuidingSpectralAttention(nf_enc[idx],nf_dec[dec_idx],rates=rates)
        #         self.skip_conns.append(skip)

        # Define the decoder
        n_encoder_layers = len(nf_enc)
        n_decoder_layers = len(nf_dec)
        start_aux = n_decoder_layers - n_heads
        self.start_aux = start_aux
        for idx in range(n_encoder_layers):
            if idx == 0:
                c_in = nf_enc[-1]
            else:
                c_in = nf_dec[idx - 1] + nf_enc[n_encoder_layers - idx - 1]
            #block = DilationBottleNeck(c_in, nf_dec[idx], dropout_rate=0.4, stride=1, do_upsample=True)
            block = conv_block(c_in, nf_dec[idx],do_upsample=True, do_batchnorm=do_batchnorm)
            self.decoder.append(block)
            # self.decoder.append(conv_block(c_in, nf_dec[idx],
            #                                do_upsample=True, do_batchnorm=do_batchnorm))
            hidden_channels = 16
            # Add aux head
            if idx == start_aux:
                self.heads.append(conv_block(nf_dec[idx], len(classes), do_upsample=False, do_batchnorm=do_batchnorm))
            if idx > start_aux:
                self.feature_learning.append(attention_module(num_classes=len(classes),
                                                          in_channels=nf_dec[idx], hidden_channels=hidden_channels,
                                                          out_channels=nf_dec[idx],
                                                          dropout=0.1
                                                          ))
                self.heads.append(conv_block(nf_dec[idx],len(classes),do_upsample=False, do_batchnorm=do_batchnorm))


        # Define the post_layer
        for idx in range(n_encoder_layers, n_decoder_layers):
            c_in = nf_dec[idx - 1]
            if idx == n_decoder_layers - 2:
                c_in = c_in + self.n_bands
            #block = DilationBottleNeck(c_in,nf_dec[idx],dropout_rate=0.4,stride=1,do_upsample=False)
            block = conv_block(c_in, nf_dec[idx],do_upsample=False, do_batchnorm=do_batchnorm)
            self.post_layers.append(block)
            # Add aux head
            if idx == start_aux:
                self.heads.append(conv_block(nf_dec[idx], len(classes), do_upsample=False, do_batchnorm=do_batchnorm))
            if idx > start_aux:
                self.feature_learning.append(AttentionOCR(len(classes),
                                                          in_channels=nf_dec[idx], hidden_channels=hidden_channels,
                                                          out_channels=nf_dec[idx],
                                                          scale=1,
                                                          dropout=0.1
                                                          ))
                self.heads.append(conv_block(nf_dec[idx], len(classes), do_upsample=False, do_batchnorm=do_batchnorm))

        # Append a conv_block that output the segmentation image

    def forward(self, x, early_exit=None):
        size = (x.size(2), x.size(3))
        # Normalize the input
        x = self.norm_input(x)
        input_org = x  # will be used by last post_layer via skip-connect

        # Add layers in the encoder (E1-4)
        enc_outputs = self.encoder(x)
        enc_outputs = enc_outputs[1:]
        x = enc_outputs[-1]
        # for layer in self.encoder:
        #     x = layer(x)
        #     enc_outputs.append(x)  # store encoder output
        #enc_outputs =

        # Add layers in the decoder (D1-4)
        n_encoder_layers = len(self.nf_enc)
        cnt = 0
        for (idx, layer) in enumerate(self.decoder):
            # concatenate the output with the corresponding encoder layer if it is required
            if idx != 0:  # concatenate on the channel dimension
                enc_output = enc_outputs[n_encoder_layers - idx - 1]
                #enc_output = self.skip_conns[-idx](x_dec=x, x_enc=enc_output)
                x = torch.cat((x, enc_output), 1)

            # Add the decoder layer
            x = layer(x)
            # Student sub-networks
            if idx == self.start_aux:
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                #o_results.append(F.interpolate(o, size))
            elif idx > self.start_aux:
                feat_size = (x.size(2), x.size(3))
                x, a = self.feature_learning[cnt](x,F.interpolate(o,feat_size))
                cnt += 1
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                #o = F.interpolate(o, size)
                #o_results.append(F.interpolate(o, size))
                #f_results.append(x)
            # if early_exit == cnt and idx >= self.start_aux:
            #     return o

        # Add layers after the decoder (D5 and D6)
        for (idx, layer) in enumerate(self.post_layers):
            # Concatenate the input of second last layer with the original
            if idx == (len(self.post_layers) - 2):
                x = torch.cat((x, input_org), 1)

            # Add the layer
            x = layer(x)
            if idx + n_encoder_layers == self.start_aux:
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                #o_results.append(F.interpolate(o, size))
            elif idx + n_encoder_layers > self.start_aux:
                x, a = self.feature_learning[cnt](x, o)
                cnt += 1
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                # if idx == len(self.post_layers):
                #     o = self.recovery_size(o, self.norm_input.padding)
                #if idx == len(self.post_layers)-1:
                #o_results.append(F.interpolate(o, size))
                #f_results.append(x)
            # if early_exit == cnt and idx + n_encoder_layers >= self.start_aux:
            #     return o

        # # Recovery the size of the input
        # x = self.recovery_size(x, self.norm_input.padding)
        return x, o



if __name__ == "__main__":
    # Test the hsi_net model
    # from pytorch_memlab import LineProfiler
    # with LineProfiler() as prof:
    # prof.display()
    # from pytorch_memlab import Courtesy
    # iamcourtesy = Courtesy()
    x = torch.rand((1, 25, 217, 409))
    print(x.shape)
    model = HSINet(n_bands=25, classes=[0, 1, 2, 3, 4,5],
                   nf_enc=[16, 32, 32, 32], nf_dec=[32, 32, 32, 32, 8, 8],
                   do_batchnorm=False, max_norm_val=None)


    y = model(x)

    print(y.shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params: %d' % total_params)

    from functools import reduce
    total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in model.parameters())
    print('Number of trainable params: %d' % total_params)

    from thop import profile, clever_format
    # Profile model
    x = torch.rand((1, 25, 217, 409))
    flops, params = profile(model, inputs=(x,), verbose=False)
    macs, params = clever_format([flops, params], "%.3f")
    print("MACs: {}, FLOPs: {}, PARAMS: {}".format(macs, flops, params))

    # # Estimate the model size
    # from utils.utils import SizeEstimator
    # se = SizeEstimator(model, input_size=(1, 25, 217, 409))
    # print('Size estimation: ', se.estimate_size())
    #
    # print('Bits taken by params: ', se.param_bits)  # bits taken up by parameters
    # print('Bits for forward and backward: ', se.forward_backward_bits)  # bits stored for forward and backward
    # print('Bits for input: ', se.input_bits) # bits for input

    # Compute the size of the params
    bits = 32
    mods = list(model.modules())
    sizes = []
    for i in range(1, len(mods)):
        m = mods[i]
        p = list(m.parameters())
        for j in range(len(p)):
            sizes.append(np.array(p[j].size()))

    total_bits = 0
    for i in range(len(sizes)):
        s = sizes[i]
        bits = np.prod(np.array(s))*bits
        total_bits += bits

    print('Total bits of the model: %d' % total_bits)

# Step 1: Modify the existing hsi_net.py file
# Just add this class - no need to change anything else


class IgnoreBackgroundLayer(nn.Module):
    """
    Layer to ignore class 0 (background) in predictions and targets
    """
    def __init__(self, ignore_index=0):
        super(IgnoreBackgroundLayer, self).__init__()
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: predictions of shape (B, C, H, W)
            target: targets of shape (B, 1, H, W) or (B, H, W)
        Returns:
            masked_pred, masked_target with ignore_index areas set to appropriate values
        """
        if target.dim() == 4:
            target = target.squeeze(1)  # Remove channel dimension if present
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (target != self.ignore_index)
        
        # Clone predictions and targets
        masked_pred = pred.clone()
        masked_target = target.clone()
        
        # Set ignore areas in target to ignore_index for loss computation
        masked_target[~valid_mask] = self.ignore_index
        
        return masked_pred, masked_target, valid_mask

class SpatialTranscriptomicsNet(HSINet):
    def __init__(self, cell_types=[1, 2, 3, 4, 5],  # Remove 0 from active classes
                 n_genes=136, ignore_index=0, **kwargs):
        # Add ignore_index to classes for model architecture (total 6 classes)
        all_classes = [ignore_index] + cell_types
        super().__init__(classes=all_classes, n_bands=n_genes, **kwargs)
        
        self.n_genes = n_genes
        self.cell_types = cell_types
        self.active_classes = cell_types  # Only classes we care about
        self.ignore_index = ignore_index
        
        # Add ignore layer
        self.ignore_layer = IgnoreBackgroundLayer(ignore_index=ignore_index)
        
        # Gene attention mechanism
        reduction_factor = max(8, n_genes // 16)
        hidden_genes = max(16, n_genes // reduction_factor)
        
        self.gene_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_genes, hidden_genes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_genes, n_genes),
            nn.Sigmoid()
        )
        
    def forward(self, x, target=None):
        # Apply gene attention
        b, c, h, w = x.shape
        attention = self.gene_attention(x)
        attention = attention.view(b, c, 1, 1)
        x = x * attention
        
        # Get features and output from parent
        features, output = super().forward(x)
        
        # Apply ignore mask if target is provided (during training)
        if target is not None and self.training:
            output, target, valid_mask = self.ignore_layer(output, target)
            return features, output, target, valid_mask
        
        return features, output

class SpatialTranscriptomics_SGR_Net(Res_SGR_Net):
    """
    Knowledge Distillation version for Spatial Transcriptomics with ignore index support
    """
    def __init__(self, cell_types=[1, 2, 3, 4, 5],  # Only active classes
                 n_genes=136, encoder_name='resnet50', 
                 ignore_index=0, 
                 nf_enc=[64, 128, 256, 512, 1024],
                 nf_dec=[64, 32, 32, 16, 16, 8], 
                 do_batchnorm=True, 
                 max_norm_val=None, 
                 n_heads=5,
                 **kwargs):
        
        # Include ignore_index for architecture (total 6 classes)
        all_classes = [ignore_index] + cell_types  # [0, 1, 2, 3, 4, 5]
        
        # Initialize parent class with explicit parameters
        super().__init__(
            classes=all_classes,
            n_bands=n_genes,  # Map n_genes to n_bands
            nf_enc=nf_enc,
            nf_dec=nf_dec,
            do_batchnorm=do_batchnorm,
            max_norm_val=max_norm_val,
            n_heads=n_heads,
            encoder_name=encoder_name
        )
        
        # Update parameters
        self.n_genes = n_genes
        self.cell_types = cell_types
        self.all_classes = all_classes
        self.ignore_index = ignore_index
        
        # Add ignore layer
        self.ignore_layer = IgnoreBackgroundLayer(ignore_index=ignore_index)
        
        # Fix the encoder for gene expression input
        self._fix_encoder_for_genes(n_genes, encoder_name)
    
    def forward(self, x, early_exit=None, target=None):
        """
        Forward pass with ignore index support
        """
        size = (x.size(2), x.size(3))
        
        # Normalize the input
        x = self.norm_input(x)
        input_org = x  # will be used by last post_layer via skip-connect
        
        # Add layers in the encoder
        enc_outputs = self.encoder(x)
        enc_outputs = enc_outputs[1:]
        x = enc_outputs[-1]
        f_results, o_results = [], []
        
        # Add layers in the decoder
        n_encoder_layers = len(self.nf_enc)
        cnt = 0
        for (idx, layer) in enumerate(self.decoder):
            # concatenate the output with the corresponding encoder layer if it is required
            if idx != 0:  # concatenate on the channel dimension
                enc_output = enc_outputs[n_encoder_layers - idx - 1]
                x = torch.cat((x, enc_output), 1)
            
            # Add the decoder layer
            x = layer(x)
            
            # Student sub-networks
            if idx == self.start_aux:
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                o_results.append(F.interpolate(o, size))
            elif idx > self.start_aux:
                feat_size = (x.size(2), x.size(3))
                x, a = self.feature_learning[cnt](x, F.interpolate(o, feat_size))
                cnt += 1
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                o_results.append(F.interpolate(o, size))
                f_results.append(x)
        
        # Add layers after the decoder
        for (idx, layer) in enumerate(self.post_layers):
            # Concatenate the input of second last layer with the original
            if idx == (len(self.post_layers) - 2):
                x = torch.cat((x, input_org), 1)
            
            # Add the layer
            x = layer(x)
            if idx + n_encoder_layers == self.start_aux:
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                o_results.append(F.interpolate(o, size))
            elif idx + n_encoder_layers > self.start_aux:
                x, a = self.feature_learning[cnt](x, o)
                cnt += 1
                o = self.heads[cnt](x)
                if early_exit == cnt:
                    return x, o
                o_results.append(o)
                f_results.append(x)
        
        # Apply ignore mask if target is provided (during training)
        if target is not None and self.training:
            # Apply ignore mask to each head output
            masked_o_results = []
            for o in o_results:
                _, masked_o, _ = self.ignore_layer(o, target)
                masked_o_results.append(masked_o)
            return f_results, masked_o_results
        
        return f_results, o_results
    
    def _fix_encoder_for_genes(self, n_genes, encoder_name):
        """Fix ResNet encoder to accept gene expression channels"""
        # Get the encoder's first layer
        if hasattr(self.encoder, 'conv1'):
            old_conv = self.encoder.conv1
        else:
            # For segmentation_models_pytorch encoders
            for name, module in self.encoder.named_modules():
                if isinstance(module, nn.Conv2d):
                    old_conv = module
                    break
        
        # Create new first layer
        new_conv = nn.Conv2d(
            n_genes, 
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize weights
        with torch.no_grad():
            if old_conv.weight.shape[1] == 3 and n_genes != 3:
                # Adapt pretrained RGB weights to gene channels
                weight_mean = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.data = weight_mean.repeat(1, n_genes, 1, 1) / n_genes
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            if new_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()
        
        # Replace the conv layer
        if hasattr(self.encoder, 'conv1'):
            self.encoder.conv1 = new_conv
        else:
            # For segmentation_models_pytorch, replace in the model
            for name, module in self.encoder.named_children():
                if isinstance(module, nn.Conv2d):
                    setattr(self.encoder, name, new_conv)
                    break
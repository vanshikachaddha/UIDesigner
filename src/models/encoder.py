#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encoder.py

Simple CNN encoder for pix2code-style model.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Simple CNN encoder:
    - 3 conv blocks (conv + ReLU + maxpool)
    - Global average pool at the end
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.f1 = nn.Linear(in_features=32*32*128, out_features=512)
        self.f2 = nn.Linear(in_features=512, out_features=512)
    def block1(self, input):
        # input (256, 256, 3)
        conv1 = self.relu(self.conv1(input))
        # conv1 (256, 256, 32)
        conv2 = self.relu(self.conv2(conv1))
        # conv2 (256, 256, 32)
        self.b1_result =self.pool(conv2)
        # max pool (128, 128, 32)
        return self.b1_result
    def block2(self, input):
        conv3 = self.relu(self.conv3(self.b1_result))
        # conv3 (128, 128, 64)
        conv4 = self.relu(self.conv4(conv3))
        # conv4 (128, 128, 64)
        self.b2_result = self.pool(conv4)
        # max pool (64, 64, 64)
        return self.b2_result
    def block3(self, input):
        conv5 = self.relu(self.conv5(self.b2_result))
        # conv5 (64, 64, 128)
        conv6 = self.relu(self.conv6(conv5))
        # conv5 (64, 64, 128)
        self.b3_result = self.pool(conv6)
        # max pool (32, 32, 128)
        return self.b3_result
    def forward(self, x):
        block1_output = self.block1(x)
        block2_output = self.block2(block1_output)
        block3_output = self.block3(block2_output)
        # Flatten: (B, 128, 32, 32) -> (B, 32*32*128)
        flattened = block3_output.view(block3_output.size(0), -1)
        f1_output = self.relu(self.f1(flattened))
        f2_output = self.relu(self.f2(f1_output))
        return f2_output




    

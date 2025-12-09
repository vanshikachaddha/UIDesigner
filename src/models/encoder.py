import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    ResNet-based image encoder for GUI screenshots.
    Uses pretrained ResNet backbone and extracts spatial features.
    """
    def __init__(self, 
                 resnet_version='resnet50',
                 pretrained=True,
                 feature_dim=512,
                 freeze_backbone=False):
        """
        Args:
            resnet_version: Which ResNet to use ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained: Whether to use ImageNet pretrained weights
            feature_dim: Dimension of output features
            freeze_backbone: Whether to freeze ResNet weights during training
        """
        super(ResNetEncoder, self).__init__()
        
        # Load pretrained ResNet
        if resnet_version == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.backbone_dim = 512
        elif resnet_version == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.backbone_dim = 512
        elif resnet_version == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.backbone_dim = 2048
        elif resnet_version == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            self.backbone_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet version: {resnet_version}")
        
        # Remove the final fully connected layer and average pooling
        # We want spatial features, not a single vector
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Project to desired feature dimension
        self.feature_projection = nn.Sequential(
            nn.Conv2d(self.backbone_dim, feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            
        Returns:
            features: Tensor of shape (batch_size, num_patches, feature_dim)
            spatial_shape: Tuple (H', W') representing spatial dimensions
        """
        # Extract features from ResNet
        # Shape: (batch_size, backbone_dim, H', W')
        features = self.backbone(images)
        
        # Project to target dimension
        # Shape: (batch_size, feature_dim, H', W')
        features = self.feature_projection(features)
        
        batch_size, feature_dim, h, w = features.shape
        
        # Flatten spatial dimensions for transformer
        # Shape: (batch_size, feature_dim, H' * W')
        features = features.view(batch_size, feature_dim, h * w)
        
        # Transpose to (batch_size, num_patches, feature_dim)
        features = features.transpose(1, 2)
        
        return features, (h, w)
    
    def get_feature_dim(self):
        """Returns the output feature dimension"""
        return self.feature_dim


class PositionalEncoding2D(nn.Module):
    """
    2D Positional encoding for spatial image features.
    """
    def __init__(self, d_model, max_h=64, max_w=64):
        super(PositionalEncoding2D, self).__init__()
        
        # Create positional encodings
        pe = torch.zeros(max_h, max_w, d_model)
        
        # Compute the positional encodings
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * 
                           -(torch.log(torch.tensor(10000.0)) / d_model_half))
        
        pos_h = torch.arange(0., max_h).unsqueeze(1)
        pos_w = torch.arange(0., max_w).unsqueeze(1)
        
        pe[:, :, 0:d_model_half:2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, max_w, 1)
        pe[:, :, 1:d_model_half:2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, max_w, 1)
        pe[:, :, d_model_half::2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(max_h, 1, 1)
        pe[:, :, d_model_half+1::2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(max_h, 1, 1)
        
        # Flatten to (max_h * max_w, d_model)
        pe = pe.view(-1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x, spatial_shape):
        """
        Args:
            x: Tensor of shape (batch_size, num_patches, d_model)
            spatial_shape: Tuple (h, w)
            
        Returns:
            x with positional encoding added
        """
        h, w = spatial_shape
        num_patches = h * w
        
        # Add positional encoding
        x = x + self.pe[:num_patches, :].unsqueeze(0)
        return x
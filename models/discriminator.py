import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for Fantasy Art
    Classifies 256x256 RGB images as real or fake
    """
    
    def __init__(self, num_classes=0, embed_dim=50):
        super(Discriminator, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Convolutional layers
        self.main = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(3, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final classification layer
        if num_classes > 0:
            # Conditional discriminator
            self.embedding = nn.Embedding(num_classes, embed_dim)
            self.fc = nn.Sequential(
                nn.Linear(512 * 4 * 4 + embed_dim, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )
        else:
            # Unconditional discriminator
            self.fc = nn.Sequential(
                nn.Linear(512 * 4 * 4, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights according to DCGAN paper"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, input_img, labels=None):
        """
        Forward pass
        Args:
            input_img: Input images [batch_size, 3, 256, 256]
            labels: Optional class labels [batch_size]
        Returns:
            Probability of being real [batch_size, 1]
        """
        # Pass through conv layers
        features = self.main(input_img)
        features = features.view(features.size(0), -1)  # Flatten
        
        if labels is not None and self.num_classes > 0:
            # Embed labels and concatenate
            label_embed = self.embedding(labels)
            features = torch.cat([features, label_embed], dim=1)
        
        # Final classification
        output = self.fc(features)
        
        return output


class ConditionalDiscriminator(Discriminator):
    """
    Conditional Discriminator with predefined fantasy classes
    """
    
    def __init__(self):
        # Same classes as generator
        self.class_names = [
            'warrior', 'mage', 'archer', 'dragon', 
            'landscape', 'castle', 'forest', 'abstract'
        ]
        
        super().__init__(num_classes=len(self.class_names), embed_dim=50)


def test_discriminator():
    """Test function to verify discriminator works"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test unconditional discriminator
    disc = Discriminator()
    disc.to(device)
    
    fake_images = torch.randn(4, 3, 256, 256, device=device)
    output = disc(fake_images)
    print(f"Unconditional output shape: {output.shape}")
    
    # Test conditional discriminator
    cond_disc = ConditionalDiscriminator()
    cond_disc.to(device)
    
    labels = torch.randint(0, 8, (4,), device=device)
    output = cond_disc(fake_images, labels)
    print(f"Conditional output shape: {output.shape}")
    print("Discriminator test passed!")


if __name__ == "__main__":
    test_discriminator()

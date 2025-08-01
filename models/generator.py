import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator for Fantasy Art Generation
    Generates 256x256 RGB images from random noise and optional conditions
    """
    
    def __init__(self, noise_dim=100, num_classes=0, embed_dim=50):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # If we have classes, add embedding layer
        if num_classes > 0:
            self.embedding = nn.Embedding(num_classes, embed_dim)
            input_dim = noise_dim + embed_dim
        else:
            input_dim = noise_dim
        
        # Initial dense layer
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4 * 4 * 512),
            nn.BatchNorm1d(4 * 4 * 512),
            nn.ReLU(True)
        )
        
        # Transpose convolution layers
        self.main = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output between -1 and 1
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
    
    def forward(self, noise, labels=None):
        """
        Forward pass
        Args:
            noise: Random noise tensor [batch_size, noise_dim]
            labels: Optional class labels [batch_size]
        Returns:
            Generated images [batch_size, 3, 256, 256]
        """
        if labels is not None and self.num_classes > 0:
            # Embed labels and concatenate with noise
            label_embed = self.embedding(labels)
            input_tensor = torch.cat([noise, label_embed], dim=1)
        else:
            input_tensor = noise
        
        # Pass through dense layer and reshape
        x = self.fc(input_tensor)
        x = x.view(-1, 512, 4, 4)
        
        # Pass through conv layers
        output = self.main(x)
        
        return output


class ConditionalGenerator(Generator):
    """
    Conditional Generator with predefined fantasy classes
    """
    
    def __init__(self, noise_dim=100):
        # Fantasy character classes
        self.class_names = [
            'warrior', 'mage', 'archer', 'dragon', 
            'landscape', 'castle', 'forest', 'abstract'
        ]
        
        super().__init__(
            noise_dim=noise_dim, 
            num_classes=len(self.class_names), 
            embed_dim=50
        )
    
    def generate_by_class(self, batch_size, class_name, device='cuda'):
        """
        Generate images for a specific class
        Args:
            batch_size: Number of images to generate
            class_name: Name of the class ('warrior', 'mage', etc.)
            device: Device to generate on
        Returns:
            Generated images
        """
        if class_name not in self.class_names:
            raise ValueError(f"Class {class_name} not found. Available: {self.class_names}")
        
        # Create noise and labels
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        class_idx = self.class_names.index(class_name)
        labels = torch.full((batch_size,), class_idx, device=device, dtype=torch.long)
        
        with torch.no_grad():
            return self.forward(noise, labels)


def test_generator():
    """Test function to verify generator works"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test unconditional generator
    gen = Generator(noise_dim=100)
    gen.to(device)
    
    noise = torch.randn(4, 100, device=device)
    output = gen(noise)
    print(f"Unconditional output shape: {output.shape}")
    
    # Test conditional generator
    cond_gen = ConditionalGenerator(noise_dim=100)
    cond_gen.to(device)
    
    output = cond_gen.generate_by_class(4, 'warrior', device)
    print(f"Conditional output shape: {output.shape}")
    print("Generator test passed!")


if __name__ == "__main__":
    test_generator()

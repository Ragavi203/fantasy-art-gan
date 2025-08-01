import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
from datetime import datetime

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import ConditionalGenerator
from models.discriminator import ConditionalDiscriminator, SpectralNormDiscriminator


class FantasyGANTrainer:
    """
    Trainer for Fantasy Art GAN
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = ConditionalGenerator(noise_dim=config['noise_dim'])
        self.discriminator = ConditionalDiscriminator()
        
        # Move to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=config['lr_g'], 
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=config['lr_d'], 
            betas=(0.5, 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Fixed noise for monitoring progress
        self.fixed_noise = torch.randn(64, config['noise_dim'], device=self.device)
        self.fixed_labels = torch.randint(
            0, len(self.generator.class_names), 
            (64,), device=self.device
        )
        
        # Tracking
        self.G_losses = []
        self.D_losses = []
        
        # Create output directories
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
    
    def get_dataloader(self):
        """Create dataloader for training"""
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # Note: You'll need to organize your data into folders by class
        # data/processed/warrior/, data/processed/mage/, etc.
        try:
            dataset = ImageFolder(
                root='data/processed',
                transform=transform
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            print(f"Dataset loaded: {len(dataset)} images")
            print(f"Classes found: {dataset.classes}")
            
            return dataloader
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Please organize your data in data/processed/ with subfolders for each class")
            return None
    
    def train_discriminator(self, real_images, real_labels):
        """Train discriminator for one step"""
        batch_size = real_images.size(0)
        
        # Labels for real and fake
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)
        
        # Train with real images
        self.optimizer_D.zero_grad()
        output_real = self.discriminator(real_images, real_labels)
        loss_D_real = self.criterion(output_real, real_label)
        
        # Train with fake images
        noise = torch.randn(batch_size, self.config['noise_dim'], device=self.device)
        fake_labels = torch.randint(
            0, len(self.generator.class_names), 
            (batch_size,), device=self.device
        )
        
        fake_images = self.generator(noise, fake_labels)
        output_fake = self.discriminator(fake_images.detach(), fake_labels)
        loss_D_fake = self.criterion(output_fake, fake_label)
        
        # Total discriminator loss
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        self.optimizer_D.step()
        
        return loss_D.item()
    
    def train_generator(self, batch_size):
        """Train generator for one step"""
        # Generate fake images
        self.optimizer_G.zero_grad()
        
        noise = torch.randn(batch_size, self.config['noise_dim'], device=self.device)
        fake_labels = torch.randint(
            0, len(self.generator.class_names), 
            (batch_size,), device=self.device
        )
        
        fake_images = self.generator(noise, fake_labels)
        
        # Try to fool discriminator
        real_label = torch.ones(batch_size, 1, device=self.device)
        output = self.discriminator(fake_images, fake_labels)
        loss_G = self.criterion(output, real_label)
        
        loss_G.backward()
        self.optimizer_G.step()
        
        return loss_G.item()
    
    def save_samples(self, epoch):
        """Save sample images"""
        self.generator.eval()
        
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise, self.fixed_labels)
            
            # Denormalize
            fake_images = (fake_images + 1) / 2
            
            # Save grid
            vutils.save_image(
                fake_images,
                f'outputs/epoch_{epoch:04d}.png',
                nrow=8,
                padding=2,
                normalize=False
            )
        
        self.generator.train()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'G_losses': self.G_losses,
            'D_losses': self.D_losses,
            'config': self.config
        }
        
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, 'checkpoints/latest.pth')  # Always keep latest
    
    def train(self):
        """Main training loop"""
        dataloader = self.get_dataloader()
        if dataloader is None:
            return
        
        print(f"Starting training on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        for epoch in range(self.config['num_epochs']):
            epoch_G_loss = 0
            epoch_D_loss = 0
            
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
            
            for i, (real_images, labels) in enumerate(pbar):
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                
                # Train Discriminator
                d_loss = self.train_discriminator(real_images, labels)
                epoch_D_loss += d_loss
                
                # Train Generator (every other iteration)
                if i % 2 == 0:
                    g_loss = self.train_generator(real_images.size(0))
                    epoch_G_loss += g_loss
                else:
                    g_loss = epoch_G_loss / max(1, (i//2 + 1))
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{d_loss:.4f}',
                    'G_loss': f'{g_loss:.4f}'
                })
                
                # Log to wandb
                if self.config['use_wandb']:
                    wandb.log({
                        'D_loss': d_loss,
                        'G_loss': g_loss,
                        'epoch': epoch,
                        'step': epoch * len(dataloader) + i
                    })
            
            # Average losses for epoch
            avg_G_loss = epoch_G_loss / (len(dataloader) // 2)
            avg_D_loss = epoch_D_loss / len(dataloader)
            
            self.G_losses.append(avg_G_loss)
            self.D_losses.append(avg_D_loss)
            
            print(f'Epoch [{epoch+1}/{self.config["num_epochs"]}] - D_loss: {avg_D_loss:.4f}, G_loss: {avg_G_loss:.4f}')
            
            # Save samples and checkpoint
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_samples(epoch + 1)
                self.save_checkpoint(epoch + 1)
        
        print("Training completed!")
        self.save_checkpoint(self.config['num_epochs'])


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'batch_size': 32,  # Reduce if GPU memory issues
        'num_epochs': 50,  # Start small for testing
        'noise_dim': 100,
        'lr_g': 0.0002,
        'lr_d': 0.0002,
        'save_every': 5,
        'use_wandb': False  # Set to True if you want to use Weights & Biases
    }
    
    # Initialize wandb if requested
    if config['use_wandb']:
        wandb.init(
            project="fantasy-art-gan",
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Create trainer and start training
    trainer = FantasyGANTrainer(config)
    trainer.train()
    
    # Finish wandb run
    if config['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    main()
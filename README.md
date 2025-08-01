# 🎨 Fantasy Art Generator

A deep learning project that generates fantasy artwork and characters using Generative Adversarial Networks (GANs). Create unique warriors, mages, dragons, landscapes, and more with the power of AI!

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note**: This is a complete implementation ready for training. The project includes all components needed to train your own fantasy art generator, but pre-trained weights are not included due to file size constraints.

## ✨ Features

- **8 Fantasy Categories**: Generate warriors, mages, archers, dragons, landscapes, castles, forests, and abstract art
- **Conditional Generation**: Control what type of content to generate
- **Multiple Art Styles**: Realistic, painterly, anime, abstract, dark, and bright styles
- **Web Interface**: Easy-to-use Gradio interface for interactive generation
- **High Quality**: 256x256 resolution with option to scale to 512x512
- **Fast Training**: Optimized for 1-2 day development cycle

## 🏗️ Project Structure

```
fantasy-art-gan/
├── data/
│   ├── raw/              # Raw downloaded images
│   └── processed/        # Organized training data
├── models/
│   ├── generator.py      # GAN Generator architecture
│   └── discriminator.py  # GAN Discriminator architecture
├── training/
│   ├── train.py          # Main training script
│   └── prepare_data.py   # Data preparation utilities
├── interface/
│   └── gradio_app.py     # Web interface
├── notebooks/
│   └── experiments.ipynb # Jupyter experiments
├── outputs/              # Generated images during training
├── checkpoints/          # Saved model weights
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fantasy-art-gan
cd fantasy-art-gan

# Create virtual environment (choose one)
# Option A: Using conda
conda create -n fantasy-gan python=3.9 -y
conda activate fantasy-gan

# Option B: Using venv (Mac/Linux)
python3 -m venv fantasy-gan-env
source fantasy-gan-env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
# Test that everything is working
python3 test_setup.py
```

### 3. Try the Demo (Immediate)

```bash
# Launch the web interface (works immediately with random weights)
python3 interface/gradio_app.py

# Open http://127.0.0.1:7860 in your browser
# Try generating some fantasy art!
```

### 4. Train Your Own Model (Optional)

```bash
# Create sample training data
python3 training/prepare_data.py

# Start training (1-3 hours on GPU, longer on CPU)
python3 training/train.py

# Generated images will appear in outputs/ folder
# Trained models saved in checkpoints/
```

## 📸 Screenshots

<div align="center">
  <img src="docs/interface_screenshot.png" alt="Web Interface" width="600">
  <p><em>Interactive Gradio web interface for generating fantasy art</em></p>
</div>

## 🎯 Features Showcase

- **🎨 8 Fantasy Categories**: Warriors, mages, archers, dragons, landscapes, castles, forests, abstract art
- **🎛️ Interactive Controls**: Real-time generation with style and randomness controls  
- **🖥️ Web Interface**: Beautiful Gradio interface that works in any browser
- **⚡ Fast Training**: Optimized for quick experimentation and development
- **🔧 Modular Design**: Easy to extend with new character types or art styles
- **📊 Progress Monitoring**: Visual training progress with sample generation

## 🎯 Usage Examples

### Basic Generation
```python
from models.generator import ConditionalGenerator
import torch

# Load trained model
generator = ConditionalGenerator()
generator.load_state_dict(torch.load('checkpoints/latest.pth')['generator_state_dict'])

# Generate a warrior
images = generator.generate_by_class(batch_size=4, class_name='warrior')
```

### Custom Training
```python
# Modify training configuration
config = {
    'batch_size': 16,     # Reduce for smaller GPU
    'num_epochs': 100,    # Increase for better quality
    'lr_g': 0.0001,       # Generator learning rate
    'lr_d': 0.0001,       # Discriminator learning rate
}
```

### Web Interface Customization
```python
# Add new art styles or character types
# Modify interface/gradio_app.py
art_styles = ["Realistic", "Anime", "Oil Painting", "Sketch"]
character_types = ["Warrior", "Mage", "Dragon", "Your_New_Type"]
```

## 🛠️ Technical Implementation

### Model Architecture
- **Generator**: Deep Convolutional GAN (DCGAN) with conditional inputs
  - Input: 100D noise vector + class embedding  
  - Output: 256×256 RGB images
  - Parameters: ~4M parameters
- **Discriminator**: Convolutional classifier with batch normalization
  - Input: 256×256 RGB images + class labels
  - Output: Real/fake probability
  - Parameters: ~2.8M parameters

### Training Details
- **Framework**: PyTorch 2.0+
- **Loss Function**: Binary Cross-Entropy with label smoothing
- **Optimizer**: Adam (β₁=0.5, β₂=0.999)  
- **Learning Rate**: 0.0002 for both Generator and Discriminator
- **Batch Size**: 32 (configurable)
- **Training Time**: 1-3 hours on modern GPU

## 📁 Project Structure

```
fantasy-art-gan/
├── models/
│   ├── generator.py          # GAN Generator architecture
│   └── discriminator.py      # GAN Discriminator architecture  
├── training/
│   ├── train.py             # Main training script
│   └── prepare_data.py      # Data preparation utilities
├── interface/
│   └── gradio_app.py        # Interactive web interface
├── data/                    # Training data (not included)
├── outputs/                 # Generated images during training
├── checkpoints/             # Saved model weights
└── test_setup.py           # Setup verification script
```

## 🔧 Configuration

Easily customize the training by modifying `training/train.py`:

```python
config = {
    'batch_size': 32,        # Reduce for smaller GPU memory
    'num_epochs': 50,        # Increase for better quality  
    'noise_dim': 100,        # Latent space dimension
    'lr_g': 0.0002,         # Generator learning rate
    'lr_d': 0.0002,         # Discriminator learning rate
}
```

## 🎮 Usage Examples

### Programmatic Generation
```python
from models.generator import ConditionalGenerator
import torch

# Load the generator
generator = ConditionalGenerator()
# generator.load_state_dict(torch.load('checkpoints/latest.pth')['generator_state_dict'])

# Generate fantasy art
images = generator.generate_by_class(batch_size=4, class_name='warrior')
```

### Custom Training Data
1. Organize your fantasy art images into folders by category
2. Place in `data/raw/warrior/`, `data/raw/mage/`, etc.
3. Run `python3 training/prepare_data.py` to process them
4. Start training with `python3 training/train.py`

## 🚀 Performance & Results

### Training Progress
- **Epochs 1-10**: Basic shapes and color patterns emerge
- **Epochs 10-25**: Recognizable fantasy elements appear  
- **Epochs 25-50**: High-quality, detailed artwork
- **Epochs 50+**: Fine-tuning and style refinement

### Hardware Requirements
- **Minimum**: 8GB RAM, any modern CPU (CPU training supported)
- **Recommended**: 16GB RAM, GPU with 8GB VRAM
- **Training Time**: 1-3 hours (GPU) / 6-12 hours (CPU)

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- **🎨 New art styles**: Add neural style transfer capabilities
- **📸 Higher resolution**: Implement Progressive GAN for 512x512+ images  
- **🎯 New categories**: Add more fantasy character types
- **⚡ Performance**: Optimize training speed and memory usage
- **🌐 Deployment**: Add Docker support or cloud deployment scripts

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **Gradio Team**: For making ML demos incredibly easy to create
- **GAN Research Community**: For advancing generative modeling techniques
- **Fantasy Art Community**: For inspiration and reference

## 📞 Support

- **🐛 Bug Reports**: Open an issue with a detailed description
- **💡 Feature Requests**: Use GitHub Discussions to propose new features  
- **❓ Questions**: Check existing issues or start a new discussion

---

**⭐ Star this repository if you found it helpful!**

Built with ❤️ for the AI and fantasy art communities.
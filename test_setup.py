#!/usr/bin/env python3
"""
Quick setup test for Fantasy Art GAN
Tests that all components work correctly
"""

import sys
import os
import torch
import importlib
from pathlib import Path

def test_imports():
    """Test that all required packages are installed"""
    print("🔍 Testing imports...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'PIL', 'matplotlib', 
        'tqdm', 'gradio', 'albumentations'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Missing packages: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful!")
    return True

def test_pytorch_gpu():
    """Test PyTorch and GPU availability"""
    print("\n🖥️  Testing PyTorch and GPU...")
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️  No GPU detected. Training will be slow on CPU.")
    
    return True

def test_model_creation():
    """Test that models can be created"""
    print("\n🧠 Testing model creation...")
    
    try:
        # Test generator
        from models.generator import ConditionalGenerator
        generator = ConditionalGenerator(noise_dim=100)
        print(f"  ✅ Generator created: {sum(p.numel() for p in generator.parameters()):,} parameters")
        
        # Test discriminator
        from models.discriminator import ConditionalDiscriminator
        discriminator = ConditionalDiscriminator()
        print(f"  ✅ Discriminator created: {sum(p.numel() for p in discriminator.parameters()):,} parameters")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False

def test_model_forward():
    """Test model forward passes"""
    print("\n⚡ Testing model forward passes...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test generator
        from models.generator import ConditionalGenerator
        generator = ConditionalGenerator(noise_dim=100)
        generator.to(device)
        
        noise = torch.randn(2, 100, device=device)
        labels = torch.randint(0, 8, (2,), device=device)
        
        with torch.no_grad():
            generated = generator(noise, labels)
        
        print(f"  ✅ Generator forward pass: {generated.shape}")
        
        # Test discriminator
        from models.discriminator import ConditionalDiscriminator
        discriminator = ConditionalDiscriminator()
        discriminator.to(device)
        
        with torch.no_grad():
            output = discriminator(generated, labels)
        
        print(f"  ✅ Discriminator forward pass: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return False

def test_data_directories():
    """Test that data directories exist"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        'data/raw', 'data/processed', 'models', 'training', 
        'interface', 'outputs', 'checkpoints'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n⚠️  Missing directories: {', '.join(missing_dirs)}")
        print("Run the setup commands from the README")
        return False
    
    print("✅ All directories exist!")
    return True

def test_data_preparation():
    """Test data preparation"""
    print("\n🎨 Testing data preparation...")
    
    try:
        from training.prepare_data import DataPreparer
        preparer = DataPreparer()
        
        # Check if we have any processed data
        total_images = preparer.validate_data()
        
        if total_images == 0:
            print("  ℹ️  No training data found")
            print("  Run: python training/prepare_data.py")
        else:
            print(f"  ✅ Found {total_images} training images")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data preparation test failed: {e}")
        return False

def test_gradio_interface():
    """Test that Gradio interface can be created"""
    print("\n🌐 Testing Gradio interface...")
    
    try:
        from interface.gradio_app import FantasyArtGenerator
        
        # Create generator (without loading model)
        generator = FantasyArtGenerator(model_path=None)
        
        print("  ✅ Gradio interface created successfully")
        print("  📝 To launch: python interface/gradio_app.py")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Gradio interface test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Fantasy Art GAN - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_pytorch_gpu,
        test_data_directories,
        test_model_creation,
        test_model_forward,
        test_data_preparation,
        test_gradio_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready!")
        print("\n📝 Next steps:")
        print("   1. python training/prepare_data.py  # Setup training data")
        print("   2. python training/train.py         # Start training")
        print("   3. python interface/gradio_app.py   # Launch web interface")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
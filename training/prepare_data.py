"""
Quick data preparation for Fantasy Art GAN
"""
import os
from PIL import Image
import random
from tqdm import tqdm

class DataPreparer:
    """Handles data preparation"""
    
    def __init__(self):
        self.processed_dir = "data/processed"
        self.classes = [
            'warrior', 'mage', 'archer', 'dragon', 
            'landscape', 'castle', 'forest', 'abstract'
        ]
        
        # Create class directories
        for class_name in self.classes:
            os.makedirs(f"{self.processed_dir}/{class_name}", exist_ok=True)
    
    def validate_data(self):
        """Count existing images"""
        total = 0
        for class_name in self.classes:
            class_dir = f"{self.processed_dir}/{class_name}"
            if os.path.exists(class_dir):
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                count = len(images)
                total += count
                print(f"  {class_name}: {count} images")
        return total
    
    def create_sample_data(self):
        """Create sample colored images for testing"""
        print("🎨 Creating sample data...")
        
        for class_name in self.classes:
            class_dir = f"{self.processed_dir}/{class_name}"
            
            for i in range(10):  # 10 images per class
                # Create a simple colored image
                img = Image.new('RGB', (256, 256), 
                               color=(random.randint(50,200), 
                                     random.randint(50,200), 
                                     random.randint(50,200)))
                img.save(f"{class_dir}/sample_{i:03d}.jpg")
        
        print("✅ Sample data created!")

def main():
    preparer = DataPreparer()
    if preparer.validate_data() == 0:
        preparer.create_sample_data()
    preparer.validate_data()

if __name__ == "__main__":
    main()

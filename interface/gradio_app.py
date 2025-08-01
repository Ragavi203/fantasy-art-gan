"""
Simple Gradio interface for Fantasy Art Generation
"""
import sys
import os
import torch
import gradio as gr
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import ConditionalGenerator

class FantasyArtGenerator:
    """Simple web interface"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = ConditionalGenerator(noise_dim=100)
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                print(f"Model loaded from {model_path}")
            except:
                print("Using random weights (no trained model found)")
        
        self.generator.to(self.device)
        self.generator.eval()
    
    def generate_image(self, character_type, num_images=2):
        """Generate fantasy images"""
        try:
            with torch.no_grad():
                images = self.generator.generate_by_class(
                    batch_size=num_images,
                    class_name=character_type.lower(),
                    device=self.device
                )
                
                # Convert to PIL images
                pil_images = []
                for img in images:
                    img = (img + 1) / 2  # Denormalize
                    img = torch.clamp(img, 0, 1)
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(img_np))
                
                return pil_images
        except Exception as e:
            print(f"Generation error: {e}")
            # Return placeholder
            placeholder = Image.new('RGB', (256, 256), color='gray')
            return [placeholder]
    
    def create_interface(self):
        """Create simple Gradio interface"""
        with gr.Blocks(title="Fantasy Art Generator") as app:
            gr.HTML("<h1>🎨 Fantasy Art Generator</h1>")
            
            with gr.Row():
                character_type = gr.Dropdown(
                    choices=["Warrior", "Mage", "Archer", "Dragon", "Landscape"],
                    value="Warrior",
                    label="Character Type"
                )
                generate_btn = gr.Button("Generate Art", variant="primary")
            
            output_gallery = gr.Gallery(label="Generated Images", columns=2)
            
            generate_btn.click(
                fn=self.generate_image,
                inputs=[character_type],
                outputs=[output_gallery]
            )
        
        return app

def main():
    generator = FantasyArtGenerator()
    app = generator.create_interface()
    print("🚀 Launching Fantasy Art Generator...")
    app.launch(server_port=7860, share=False)

if __name__ == "__main__":
    main()

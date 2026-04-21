import os
import torch
import argparse
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def recognize_math(image_path, model_dir):
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model & Processor
    print(f"Loading model from {model_dir}...")
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()

    # Load Image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Generate LaTeX
    print("Generating LaTeX...")
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*40)
    print(" RESULTING LATEX FORMULA:")
    print("="*40)
    print(generated_text)
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test math handwriting recognition")
    parser.add_argument("image", help="Path to your handwritten math image (jpg/png)")
    parser.add_argument("--model", default="models/trocr-large-finetuned-math-captions", help="Path to the trained model directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found!")
    else:
        recognize_math(args.image, args.model)

import os
import torch
import argparse
from functools import lru_cache
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


@lru_cache(maxsize=2)
def load_trocr(model_dir):
    """Load TrOCR model once and cache it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()
    return processor, model, device


@lru_cache(maxsize=1)
def load_dino():
    """Load Grounding DINO once and cache it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model, device


def decode_formula(image, processor, model, device):
    """Convert a PIL image (single formula crop) to LaTeX string."""
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def detect_boxes(image, dino_processor, dino_model, device):
    """Detect formula bounding boxes in a full-page image."""
    inputs = dino_processor(images=image, text="math formula", return_tensors="pt").to(
        device
    )
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )[0]
    return results["boxes"]


def recognize(image_path, model_dir, is_page=False):
    """Recognize math formula(s) from an image."""

    # Load image once
    image = Image.open(image_path).convert("RGB")

    # Always load TrOCR
    trocr_proc, trocr_model, device = load_trocr(model_dir)

    print(f"Using device: {device}")
    print(f"Model: {model_dir}")
    print("=" * 40)

    if not is_page:
        # Single formula mode
        latex = decode_formula(image, trocr_proc, trocr_model, device)
        print("RESULTING LATEX FORMULA:")
        print("=" * 40)
        print(latex)

    else:
        # Full page mode: detect then OCR each box
        dino_proc, dino_model, _ = load_dino()
        boxes = detect_boxes(image, dino_proc, dino_model, device)

        print(f"FOUND {len(boxes)} MATH FORMULA(S)")
        print("=" * 40)

        for i, box in enumerate(boxes):
            box = box.cpu().numpy().astype(int)
            crop = image.crop((box[0], box[1], box[2], box[3]))
            latex = decode_formula(crop, trocr_proc, trocr_model, device)
            print(f"[{i+1}] {latex}")

    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Math handwriting recognition")
    parser.add_argument("image", help="Path to image")
    parser.add_argument(
        "--model",
        default="models/trocr-large-finetuned-math-captions",
        help="Path to TrOCR model directory",
    )
    parser.add_argument(
        "--page",
        action="store_true",
        help="Process full page with multiple formulas",
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found!")
        exit(1)

    recognize(args.image, args.model, args.page)

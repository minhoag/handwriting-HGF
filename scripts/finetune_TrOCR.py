from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch as t
import argparse

parser = argparse.ArgumentParser(
    description="Finetune TrOCR for Math handwritten recognition"
)
parser.add_argument("--gpu", type=int, default=0, help="GPU index for CUDA")
args = parser.parse_args()

print("Select GPU Backend:")
print("1: AMD (DirectML)")
print("2: NVIDIA (CUDA)")
print("3: CPU")
choice = input("Enter choice (1, 2, or 3): ")

if choice == "1":
    import torch_directml

    device = torch_directml.device()
    print("Using AMD GPU (DirectML)")
elif choice == "2":
    device = t.device(f"cuda:{args.gpu}")
    print(f"Using NVIDIA GPU: cuda:{args.gpu}")
elif choice == "3":
    device = t.device("cpu")
    print("Using CPU")
else:
    device = t.device("cpu")
    print("No valid choice. Defaulting to CPU.")

import sys, os

# Get the absolute path of the directory containing the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from scripts/)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

sys.path.append(ROOT_DIR)

# Load LST files
import pandas as pd
import numpy as np
from tqdm import tqdmfrom 
from data.datasets import set_seed


# Load BASE model
hf_model_id = "microsoft/trocr-base-handwritten"
model = VisionEncoderDecoderModel.from_pretrained(hf_model_id, use_safetensors=True).to(
    device
)
processor = TrOCRProcessor.from_pretrained(hf_model_id)


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.05):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + t.randn(tensor.size()).to(tensor.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class MathCaptionsDataset(Dataset):
    """
    A dataset object that loads in images from img_data and captions from labels.csv.

    - Processor: A Huggingface processor object that will be used to process the images and captions.
    - img_dir: The directory containing the images and labels.csv.
    - csv_file: The file containing the captions.
    - transform: A torchvision transform to be applied to the images.
    """

    def __init__(
        self,
        processor,
        img_dir=os.path.join(ROOT_DIR, "img_data"),
        csv_file="labels.csv",
        transform=None,
        use_float16=False,
        device=device,
        partition="train",
        test_split=0.2,
        random_seed=0,
    ):

        self.img_dir = img_dir
        self.csv_path = os.path.join(img_dir, csv_file)
        self.labels = pd.read_csv(self.csv_path)
        # Randomly select data_split of the data for training and the rest for validation
        self.labels, self.labels_val = train_test_split(
            self.labels, test_size=test_split, random_state=random_seed
        )
        self.partition = partition
        self.transform = transform
        self.processor = processor
        self.use_float16 = use_float16
        self.device = device

    def __len__(self):
        if self.partition == "train":
            return len(self.labels)
        else:
            return len(self.labels_val)

    def train(self):
        self.partition = "train"

    def val(self):
        self.partition = "val"

    def __getitem__(self, idx):
        if t.is_tensor(idx):
            idx = idx.tolist()

        if self.partition == "train":
            img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 1])
        else:
            img_name = os.path.join(self.img_dir, self.labels_val.iloc[idx, 1])

        image = Image.open(img_name).convert("RGB")

        # Apply data augmentation only for training
        if self.partition == "train" and self.transform is not None:
            image = self.transform(image)

        # Process image - keep on CPU during dataset fetch to avoid per-item GPU overhead
        inputs = self.processor(images=image, padding="max_length", return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].squeeze()

        if self.partition == "train":
            caption = self.labels.iloc[idx, 0]
        else:
            caption = self.labels_val.iloc[idx, 0]
        caption = (
            self.processor.tokenizer.encode(
                caption,
                return_tensors="pt",
                padding="max_length",
                max_length=256,  # Tweak this, longest length in current dataset is 156
            )
            .to(self.device)
            .squeeze()
        )

        return inputs, caption


import torch as t
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

# Hyperparams - Optimized for extremely fast Proof of Concept (PoC)
NUM_EPOCHS = 1
LEARNING_RATE = 1e-5
BATCH_SIZE = 2  # Very safe for VRAM
SHUFFLE_DATASET = True

# Define Augmentation Transform - Simplified for better CPU performance
train_transforms = transforms.Compose(
    [
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=2, translate=(0.01, 0.01)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    ]
)

set_seed(0)
optimizer = t.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
train_ds = MathCaptionsDataset(processor, transform=train_transforms)
train_ds.train()
train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATASET, num_workers=0
)
val_ds = MathCaptionsDataset(processor)
val_ds.val()
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
assert (train_ds.labels_val.values == val_ds.labels_val.values).all()

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.to(device)
model.train()

history = []
val_history = []
val_timesteps = []
ema_loss = None
ema_alpha = 0.95
scaler = t.cuda.amp.GradScaler(enabled=True)
for epoch in range(NUM_EPOCHS):
    with tqdm(train_dl, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}") as pbar:
        for batch, captions in pbar:
            # Move batch to device here (more efficient for some backends)
            pixel_values = batch["pixel_values"].to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            # AMP (Autocast) works best on NVIDIA, but we'll leave it or disable if on DirectML
            autocast_enabled = device.type == "cuda"
            with t.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=autocast_enabled,
            ):
                outputs = model(pixel_values=pixel_values, labels=captions)
                loss = outputs.loss
                history.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = ema_loss * ema_alpha + loss.item() * (1 - ema_alpha)
            pbar.set_postfix(loss=ema_loss)

            # Explicitly delete tensors to help with memory reclamation
            del pixel_values, captions, outputs, loss

    model.eval()
    with t.no_grad():
        val_losses = []
        for batch, captions in tqdm(val_dl):
            pixel_values = batch["pixel_values"]
            outputs = model(pixel_values=pixel_values, labels=captions)
            val_losses.append(outputs.loss.item())
        print(f"Validation loss: {np.mean(val_losses)}")
        val_history.append(np.mean(val_losses))
        val_timesteps.append(len(history) - 1)

output_dir = os.path.join(ROOT_DIR, "models", "trocr-large-finetuned-math-captions")
os.makedirs(output_dir, exist_ok=True)

import json

model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

# Save history as JSON instead of torch.save to avoid security blocks
with open(os.path.join(output_dir, "history.json"), "w") as f:
    json.dump(history, f)
with open(os.path.join(output_dir, "val_history.json"), "w") as f:
    json.dump(val_history, f)
with open(os.path.join(output_dir, "val_timesteps.json"), "w") as f:
    json.dump(val_timesteps, f)

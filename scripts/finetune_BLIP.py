import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import BlipProcessor, BlipForQuestionAnswering
import argparse
import sys

# Get the absolute path of the directory containing the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from scripts/)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from data.datasets import set_seed

parser = argparse.ArgumentParser(
    description="Finetune BLIP for Math handwritten recognition VQA"
)
parser.add_argument("--gpu", type=int, default=0, help="GPU index for CUDA")
args = parser.parse_args()

print("Select GPU Backend:")
print("1: AMD (DirectML)")
print("2: NVIDIA (CUDA)")
choice = input("Enter choice (1 or 2): ")

if choice == "1":
    import torch_directml

    device = torch_directml.device()
    print("Using AMD GPU (DirectML)")
elif choice == "2":
    device = t.device(f"cuda:{args.gpu}")
    print(f"Using NVIDIA GPU: cuda:{args.gpu}")
else:
    device = t.device("cpu")
    print("No valid choice. Defaulting to CPU.")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

class MathCaptionsDataset(Dataset):
    """
    A dataset object that loads in images from img_data and captions from labels.csv.
    """
    def __init__(self, processor,
                 img_dir = os.path.join(ROOT_DIR, "img_data"), 
                 csv_file = "labels.csv",
                 transform = None, use_float16 = False,
                 device = device,
                 prefix = "What does the formula above say?", partition = "train",
                 test_split = 0.2, random_seed = 0):
        
        self.img_dir = img_dir
        self.csv_path = os.path.join(img_dir, csv_file)
        self.labels = pd.read_csv(self.csv_path)
        self.labels, self.labels_val = train_test_split(self.labels, test_size = test_split, random_state = random_seed)
        self.partition = partition
        self.transform = transform
        self.processor = processor
        self.use_float16 = use_float16
        self.device = device
        self.prefix = prefix

    def __len__(self):
        if self.partition == "train": return len(self.labels)
        else: return len(self.labels_val)
    
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
            
        image = Image.open(img_name).convert('RGB')
        
        if self.partition == "train" and self.transform is not None:
            image = self.transform(image)
            
        inputs = self.processor(image, self.prefix, padding = "max_length", return_tensors="pt").to(self.device)
        if self.use_float16:
            inputs = inputs.to(t.float16)
        for key in inputs:
            inputs[key] = inputs[key].squeeze()

        if self.partition == "train": caption = self.labels.iloc[idx, 0]
        else: caption = self.labels_val.iloc[idx, 0]
        caption = self.processor.tokenizer.encode(
            caption, return_tensors="pt", padding = "max_length", max_length = 256,
            ).to(self.device).squeeze()

        return inputs, caption

# Hyperparams
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
SHUFFLE_DATASET = True

set_seed(0)
optimizer = t.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
train_ds = MathCaptionsDataset(processor); train_ds.train()
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATASET, num_workers=0)
val_ds = MathCaptionsDataset(processor); val_ds.val()
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model.to(device)
model.train()

history = []; val_history = []; val_timesteps = []
ema_loss = None; ema_alpha = 0.99
scaler = t.cuda.amp.GradScaler(enabled = True)
for epoch in range(NUM_EPOCHS):
    with tqdm(train_dl, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}") as pbar:
        for batch, captions in pbar:
            pixel_values = batch["pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            
            optimizer.zero_grad()
            autocast_enabled = device.type == "cuda"
            with t.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled = autocast_enabled):
                outputs = model(pixel_values = pixel_values,
                                attention_mask = attention_mask,
                                input_ids = input_ids,
                                labels = captions)
                loss = outputs.loss
                history.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema_loss is None: ema_loss = loss.item()
            else: ema_loss = ema_loss * ema_alpha + loss.item() * (1 - ema_alpha)
            pbar.set_postfix(loss=ema_loss)
    
    model.eval()
    with t.no_grad():
        val_losses = []
        for batch, captions in tqdm(val_dl):
            pixel_values = batch["pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = model(pixel_values = pixel_values,
                            attention_mask = attention_mask,
                            input_ids = input_ids,
                            labels = captions)
            val_losses.append(outputs.loss.item())
        print(f"Validation loss: {np.mean(val_losses)}")
        val_history.append(np.mean(val_losses))
        val_timesteps.append(len(history) - 1)

output_dir = os.path.join(ROOT_DIR, "models", "blip-finetuned")
os.makedirs(output_dir, exist_ok=True)

import json
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

with open(os.path.join(output_dir, "history.json"), "w") as f:
    json.dump(history, f)
with open(os.path.join(output_dir, "val_history.json"), "w") as f:
    json.dump(val_history, f)
with open(os.path.join(output_dir, "val_timesteps.json"), "w") as f:
    json.dump(val_timesteps, f)

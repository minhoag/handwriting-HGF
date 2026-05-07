import os
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
import argparse

class LatexBoundingBoxDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = []
        self.masks = []
        
        # Typically you would load the list of files here
        if os.path.exists(os.path.join(root, "images")):
            self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        if os.path.exists(os.path.join(root, "masks")):
            self.masks = sorted(os.listdir(os.path.join(root, "masks")))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        
        img = torchvision.io.read_image(img_path)
        mask = torchvision.io.read_image(mask_path)
        
        # In a real scenario, you'd convert masks to bounding boxes here
        # obj_ids = torch.unique(mask)
        # obj_ids = obj_ids[1:]
        # masks = mask == obj_ids[:, None, None]
        # boxes = torchvision.ops.masks_to_boxes(masks)
        # ...
        
        target = {}
        # target["boxes"] = boxes
        # target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        # target["masks"] = masks
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def get_model(num_classes):
    # use the mobilenet for mobile detection
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN for LaTeX bounding boxes")
    parser.add_argument("--data_dir", type=str, default="data/LatexBoundingBoxDataset", help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    num_classes = 2 # background + latex
    
    if not os.path.exists(args.data_dir):
        print(f"Warning: Data directory {args.data_dir} does not exist. Please prepare the dataset.")
        return

    dataset = LatexBoundingBoxDataset(args.data_dir, get_transform(train=True))
    dataset_test = LatexBoundingBoxDataset(args.data_dir, get_transform(train=False))

    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        print(f"Epoch: {epoch+1}/{args.epochs}")
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
        lr_scheduler.step()

    print("Training complete!")

if __name__ == "__main__":
    main()

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os

# ============================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ============================================================================
BASE_TRAIN_PATH = '/kaggle/input/techtonicddc/Offroad_Segmentation_Training_Dataset/train'
BASE_VAL_PATH = '/kaggle/input/techtonicddc/Offroad_Segmentation_Training_Dataset/val'

# High-Res Configuration: Doubling the resolution for better small-object IoU
H_RES, W_RES = 448, 896 
PATCH_SIZE = 14
EPOCHS = 20
BATCH_SIZE = 6  # Reduced to accommodate high resolution in VRAM

# Aggressive weights for the "hard" classes (Logs, Rocks, Clutter)
# Order: [BG, Trees, Lush, Grass, DryBush, Clutter, Logs, Rocks, Ground, Sky]
CLASS_WEIGHTS = torch.tensor([1.2, 1.5, 1.5, 1.5, 2.5, 3.5, 7.5, 6.5, 1.0, 0.3])

# ============================================================================
# 2. DATASET LOGIC (FIXED)
# ============================================================================
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = convert_mask(Image.open(os.path.join(self.masks_dir, data_id)))
        
        if self.transform: image = self.transform(image)
        if self.mask_transform: mask = self.mask_transform(mask) * 255
        
        # RETURN ONLY TWO VALUES TO FIX THE VALUEERROR
        return image, mask

# ============================================================================
# 3. ADVANCED SEGMENTATION HEAD
# ============================================================================
class SegmentationHeadExtreme(nn.Module):
    def __init__(self, in_channels, out_classes, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        # Deepened decoder with BatchNorm to stabilize high-res learning
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, out_classes, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.decoder(x)

# ============================================================================
# 4. LOSS FUNCTION: MULTI-CLASS FOCAL LOSS
# ============================================================================
class MultiClassFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================================
# 5. TRAINING LOOP
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Heavy Augmentation for robustness
    t_img = transforms.Compose([
        transforms.Resize((H_RES, W_RES)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    t_mask = transforms.Compose([
        transforms.Resize((H_RES, W_RES), interpolation=Image.NEAREST), 
        transforms.ToTensor()
    ])

    train_loader = DataLoader(MaskDataset(BASE_TRAIN_PATH, t_img, t_mask), batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Models
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    model = SegmentationHeadExtreme(384, 10, W_RES//PATCH_SIZE, H_RES//PATCH_SIZE).to(device)

    # Optimizer & OneCycleLR (Modern training recipe for speed)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=EPOCHS)
    
    criterion = MultiClassFocalLoss(weight=CLASS_WEIGHTS.to(device))

    print(f"🚀 Launching 0.9 Quest: Res={W_RES}x{H_RES}, Epochs={EPOCHS}")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.squeeze(1).long().to(device)
            
            with torch.no_grad():
                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
            
            logits = model(tokens)
            preds = F.interpolate(logits, size=(H_RES, W_RES), mode="bilinear", align_corners=False)
            
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    torch.save(model.state_dict(), "techtonic_model_gold.pth")
    print("✅ Training Complete. Gold Weights Saved.")

if __name__ == "__main__":
    main()

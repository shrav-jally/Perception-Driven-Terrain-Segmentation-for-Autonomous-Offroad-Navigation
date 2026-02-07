import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ==========================================
# 1. SETUP
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '/kaggle/working/techtonic_model.pth' 
DATA_DIR = '/kaggle/input/techtonicddc/Offroad_Segmentation_Training_Dataset/val'
OUTPUT_DIR = '/kaggle/working/eval_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# UPDATED: Matches your 256-channel enhanced architecture
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_classes, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 7, padding=3), nn.GELU()) # Changed to 256
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256),
            nn.GELU(), nn.Conv2d(256, 256, 1), nn.GELU(),
        )
        self.classifier = nn.Conv2d(256, out_classes, 1)
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.classifier(self.block(self.stem(x)))

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform, self.mask_transform = transform, mask_transform
        self.data_ids = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
    def __len__(self): return len(self.data_ids)
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask_raw = np.array(Image.open(os.path.join(self.masks_dir, data_id)))
        v_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}
        mask_new = np.zeros_like(mask_raw, dtype=np.uint8)
        for r, n in v_map.items(): mask_new[mask_raw == r] = n
        mask = Image.fromarray(mask_new)
        if self.transform: img = self.transform(img)
        if self.mask_transform: mask = self.mask_transform(mask) * 255
        return img, mask, data_id

# ==========================================
# 2. LOAD & EVALUATE
# ==========================================
w, h = 448, 224
t_img = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), 
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
t_mask = transforms.Compose([transforms.Resize((h, w), interpolation=Image.NEAREST), transforms.ToTensor()])

val_loader = DataLoader(MaskDataset(DATA_DIR, t_img, t_mask), batch_size=4, shuffle=False)

print(f"🧠 Loading Weights...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
model = SegmentationHeadConvNeXt(384, 10, w//14, h//14).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

palette = [[0,0,0], [34,139,34], [0,255,0], [210,180,140], [139,90,43], [128,128,0], [139,69,19], [128,128,128], [160,82,45], [135,206,235]]
class_names = ["BG", "Trees", "Lush Bush", "Dry Grass", "Dry Bush", "Clutter", "Logs", "Rocks", "Ground", "Sky"]

# Per-class metric tracking
class_inter = np.zeros(10)
class_union = np.zeros(10)

print("📊 Final Evaluation in Progress...")
with torch.no_grad():
    for i, (imgs, labels, data_ids) in enumerate(tqdm(val_loader)):
        imgs, labels = imgs.to(device), labels.squeeze(1).long().to(device)
        tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
        logits = model(tokens)
        preds = torch.argmax(F.interpolate(logits, size=(h, w), mode="bilinear"), dim=1)

        # Better IoU Calculation (Accumulating for more accuracy)
        for cls in range(10):
            class_inter[cls] += ((preds == cls) & (labels == cls)).sum().item()
            class_union[cls] += ((preds == cls) | (labels == cls)).sum().item()

        # Save Visuals (First 5 batches)
        if i < 5:
            for b in range(imgs.shape[0]):
                img_np = (imgs[b].cpu().numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
                pred_color = np.zeros((h, w, 3), dtype=np.uint8)
                gt_color = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(10): 
                    pred_color[preds[b].cpu().numpy() == c] = palette[c]
                    gt_color[labels[b].cpu().numpy() == c] = palette[c]
                
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(img_np); ax[0].set_title("Input")
                ax[1].imshow(gt_color); ax[1].set_title("Target")
                ax[2].imshow(pred_color); ax[2].set_title("Techtonic AI")
                for a in ax: a.axis('off')
                plt.savefig(f"{OUTPUT_DIR}/final_{data_ids[b]}")
                plt.close()

# Calculate Final Metrics
iou_per_class = class_inter / (class_union + 1e-6)
mean_iou = np.mean(iou_per_class)

print("\n--- 🏆 PERFORMANCE SUMMARY ---")
print(f"OVERALL MEAN IoU: {mean_iou:.4f}\n")
print("PER-CLASS IoU:")
for name, iou in zip(class_names, iou_per_class):
    print(f"  {name:10}: {iou:.4f}")

print(f"\n✅ High-res results saved in {OUTPUT_DIR}")

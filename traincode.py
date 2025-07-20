import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import datetime
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"shadow_removal_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Paths - Using the actual paths from your error message
TRAIN_SHADOW_PATH = '/home/ccbd/Downloads/dataset4/SRD/Train/shadow'  # Path to shadow images
TRAIN_SHADOW_FREE_PATH = '/home/ccbd/Downloads/dataset4/SRD/Train/shadow_free'  # Path to shadow-free images
TRAIN_MASK_PATH = '/home/ccbd/Downloads/dataset4/SRD/Train/trainmask'  # Path to mask images (optional)

# Model parameters
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = './shadowformer_model.pth'
CHECKPOINT_DIR = './checkpoints'

logger.info(f"Using device: {DEVICE}")

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Robust dataset class that handles different naming conventions and missing files
class ShadowRemovalDataset(Dataset):
    def __init__(self, shadow_dir, shadow_free_dir, mask_dir=None, transform=None):
        """
        Initialize the dataset with flexible file handling.
        
        Args:
            shadow_dir: Directory containing shadow images
            shadow_free_dir: Directory containing shadow-free images
            mask_dir: Directory containing mask images (optional)
            transform: Image transformations to apply
        """
        self.shadow_dir = shadow_dir
        self.shadow_free_dir = shadow_free_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.pairs = []
        
        # Get all shadow image paths
        shadow_files = sorted(glob.glob(os.path.join(shadow_dir, "*.jpg")) + 
                              glob.glob(os.path.join(shadow_dir, "*.png")) + 
                              glob.glob(os.path.join(shadow_dir, "*.jpeg")))
        
        # Get all shadow-free image paths
        shadow_free_files = glob.glob(os.path.join(shadow_free_dir, "*.jpg")) + \
                            glob.glob(os.path.join(shadow_free_dir, "*.png")) + \
                            glob.glob(os.path.join(shadow_free_dir, "*.jpeg"))
        
        # Convert to basenames for easier matching
        shadow_free_basenames = [os.path.basename(f) for f in shadow_free_files]
        
        # Find valid pairs
        for shadow_path in shadow_files:
            shadow_basename = os.path.basename(shadow_path)
            shadow_name = os.path.splitext(shadow_basename)[0]
            
            # Try different naming conventions
            possible_free_names = [
                f"{shadow_name}_no_shadow{os.path.splitext(shadow_basename)[1]}",  # IMG_XXX_no_shadow.jpg
                f"{shadow_name}_free{os.path.splitext(shadow_basename)[1]}",       # IMG_XXX_free.jpg
                f"{shadow_name.replace('shadow', 'free')}{os.path.splitext(shadow_basename)[1]}", # shadow_XXX -> free_XXX
                shadow_basename  # Same name in both directories
            ]
            
            # Find matching shadow-free image
            shadow_free_path = None
            for possible_name in possible_free_names:
                if possible_name in shadow_free_basenames:
                    shadow_free_path = os.path.join(shadow_free_dir, possible_name)
                    break
            
            # Find matching mask if mask_dir is provided
            mask_path = None
            if mask_dir:
                mask_possible_paths = [
                    os.path.join(mask_dir, shadow_basename),
                    os.path.join(mask_dir, f"{shadow_name}_mask{os.path.splitext(shadow_basename)[1]}")
                ]
                for path in mask_possible_paths:
                    if os.path.exists(path):
                        mask_path = path
                        break
            
            # Add to dataset if shadow and shadow-free images exist
            if shadow_free_path:
                self.pairs.append((shadow_path, shadow_free_path, mask_path))
        
        logger.info(f"Found {len(self.pairs)} valid image pairs for training")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        shadow_path, shadow_free_path, mask_path = self.pairs[idx]
        
        # Load images
        try:
            shadow_img = Image.open(shadow_path).convert('RGB')
            shadow_free_img = Image.open(shadow_free_path).convert('RGB')
            
            # Create or load mask
            if mask_path and os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
            else:
                # Create dummy mask if not available
                mask_img = Image.new('L', shadow_img.size, 255)
            
            # Apply transformations
            if self.transform:
                shadow_img = self.transform(shadow_img)
                shadow_free_img = self.transform(shadow_free_img)
                
                # Make sure mask is resized to match the transformed image size
                mask_transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor()
                ])
                mask_img = mask_transform(mask_img)
                
                # Normalize mask to be binary (0 or 1)
                mask_img = (mask_img > 0.5).float()
            
            return shadow_img, shadow_free_img, mask_img, os.path.basename(shadow_path)
        except Exception as e:
            logger.error(f"Error loading image pair: {shadow_path}, {shadow_free_path}, {str(e)}")
            # Return a dummy sample in case of errors
            dummy_img = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            dummy_mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            return dummy_img, dummy_img, dummy_mask, "error.jpg"

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        return x

# PatchEmbed module
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, C, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x

# ShadowFormer Model
class ShadowFormer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Encoder - processes shadow image
        self.encoder_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.encoder_embed.num_patches, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # Decoder for generating shadow-free image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),  # H/8, W/8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # H/4, W/4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # H/2, W/2
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # H, W
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Initialize weights
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x, mask=None):
        # Save input shape for later use
        B, C, H, W = x.shape
        
        # Get embedded patches
        x = self.encoder_embed(x)  # B, N, C
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Reshape for decoder
        B, N, C = x.shape
        grid_size = int(np.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, grid_size, grid_size)
        
        # Decode to shadow-free image
        output = self.decoder(x)
        
        # Ensure output size matches input size
        if output.shape[-2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output

# Loss functions
class ShadowRemovalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target, mask=None):
        # Ensure pred and target have same size
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
        
        # Calculate L1 loss
        l1 = self.l1_loss(pred, target)
        
        # Calculate MSE loss
        mse = self.mse_loss(pred, target)
        
        # Calculate perceptual loss using gradients
        # For x direction: use same size for both tensors
        pred_x = pred[:, :, 1:, :]
        pred_prev_x = pred[:, :, :-1, :]
        target_x = target[:, :, 1:, :]
        target_prev_x = target[:, :, :-1, :]
        
        # For y direction: use same size for both tensors
        pred_y = pred[:, :, :, 1:]
        pred_prev_y = pred[:, :, :, :-1]
        target_y = target[:, :, :, 1:]
        target_prev_y = target[:, :, :, :-1]
        
        # Compute gradients
        pred_grad_x = pred_x - pred_prev_x
        pred_grad_y = pred_y - pred_prev_y
        target_grad_x = target_x - target_prev_x
        target_grad_y = target_y - target_prev_y
        
        # Compute gradient losses
        grad_loss_x = self.mse_loss(pred_grad_x, target_grad_x)
        grad_loss_y = self.mse_loss(pred_grad_y, target_grad_y)
        grad_loss = grad_loss_x + grad_loss_y
        
        # Combine losses
        total_loss = l1 + 0.5 * mse + 0.5 * grad_loss
        
        return total_loss

# Function to handle errors in data loading
class CustomBatchSampler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.valid_indices = list(range(len(dataset)))
        
    def __iter__(self):
        random.shuffle(self.valid_indices)
        batches = [self.valid_indices[i:i + self.batch_size] 
                   for i in range(0, len(self.valid_indices), self.batch_size)]
        for batch in batches:
            yield batch
            
    def __len__(self):
        return (len(self.valid_indices) + self.batch_size - 1) // self.batch_size

# Custom collate function to handle potential size mismatches and errors
def custom_collate(batch):
    # Filter out any None or error samples
    valid_batch = [(shadow, shadow_free, mask, name) for shadow, shadow_free, mask, name in batch 
                  if shadow is not None and shadow_free is not None]
    
    # If batch is empty after filtering, create a dummy batch
    if not valid_batch:
        dummy_img = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        dummy_mask = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)
        return dummy_img, dummy_img, dummy_mask, ["dummy.jpg"]
    
    # Stack the valid samples
    shadow_imgs = torch.stack([item[0] for item in valid_batch])
    shadow_free_imgs = torch.stack([item[1] for item in valid_batch])
    masks = torch.stack([item[2] for item in valid_batch])
    img_names = [item[3] for item in valid_batch]
    
    return shadow_imgs, shadow_free_imgs, masks, img_names

# Transform for data preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Helper function to denormalize images
def denormalize(img):
    return (img * 0.5 + 0.5).clamp(0, 1)

# Function to save example outputs
def save_example_outputs(epoch, model, dataloader, num_examples=3):
    model.eval()
    
    with torch.no_grad():
        for i, (shadow_imgs, shadow_free_imgs, masks, img_names) in enumerate(dataloader):
            try:
                if i >= num_examples:
                    break
                    
                # Move data to device
                shadow_imgs = shadow_imgs.to(DEVICE)
                shadow_free_imgs = shadow_free_imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                
                # Forward pass
                outputs = model(shadow_imgs, masks)
                
                # Convert to numpy for visualization
                shadow_img = denormalize(shadow_imgs[0]).cpu().permute(1, 2, 0).numpy()
                shadow_free_img = denormalize(shadow_free_imgs[0]).cpu().permute(1, 2, 0).numpy()
                output_img = denormalize(outputs[0]).cpu().permute(1, 2, 0).numpy()
                
                # Create visualization
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(shadow_img)
                plt.title("Shadow Image")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(output_img)
                plt.title("Predicted Shadow-Free")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(shadow_free_img)
                plt.title("Ground Truth")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"outputs/epoch_{epoch}_sample_{i}_{img_names[0]}")
                plt.close()
            except Exception as e:
                logger.error(f"Error saving example {i}: {e}")
                continue

# Training function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    valid_batches = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for i, (shadow_imgs, shadow_free_imgs, masks, _) in enumerate(pbar):
            try:
                # Skip empty batches
                if shadow_imgs.size(0) == 0:
                    continue
                
                # Move data to device
                shadow_imgs = shadow_imgs.to(DEVICE)
                shadow_free_imgs = shadow_free_imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(shadow_imgs, masks)
                
                # Calculate loss
                loss = criterion(outputs, shadow_free_imgs, masks)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix(loss=running_loss / max(1, valid_batches))
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue
    
    return running_loss / max(1, valid_batches)  # Avoid division by zero

# Main training function
def main_train():
    # Create dataset and dataloader
    logger.info("Creating training dataset...")
    train_dataset = ShadowRemovalDataset(
        TRAIN_SHADOW_PATH, 
        TRAIN_SHADOW_FREE_PATH, 
        TRAIN_MASK_PATH, 
        transform=transform
    )
    
    # Check if dataset is empty
    if len(train_dataset) == 0:
        logger.error("No valid image pairs found for training. Please check your dataset paths.")
        return
    
    # Create data loader with custom batch sampler and collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=CustomBatchSampler(train_dataset, BATCH_SIZE),
        collate_fn=custom_collate,
        num_workers=4
    )
    
    # Initialize model, loss, and optimizer
    logger.info("Initializing model...")
    model = ShadowFormer(
        img_size=IMAGE_SIZE, 
        patch_size=16, 
        in_chans=3, 
        embed_dim=512, 
        depth=8, 
        num_heads=8
    ).to(DEVICE)
    
    criterion = ShadowRemovalLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Train
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        
        # Log training progress
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save example outputs every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_example_outputs(epoch, model, train_loader)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            logger.info(f"New best model at epoch {epoch+1} with loss {best_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, MODEL_SAVE_PATH)
            logger.info(f"Best model saved to {MODEL_SAVE_PATH}")
        
        # Update learning rate
        scheduler.step()
    
    logger.info("Training completed!")
    logger.info(f"Best model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    # Check if directories exist
    for path in [TRAIN_SHADOW_PATH, TRAIN_SHADOW_FREE_PATH]:
        if not os.path.exists(path):
            logger.error(f"Directory does not exist: {path}")
            logger.info(f"Please update the path variables in the script or create the directory.")
            exit(1)
    
    # Run training
    main_train()

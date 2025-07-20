import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse
import glob
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("shadow_removal_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define default paths as function parameters rather than globals
def get_default_paths():
    return {
        'TEST_SHADOW_PATH': '/home/ccbd/Downloads/dataset6/SRD/test/shadow',
        'TEST_SHADOW_FREE_PATH': '/home/ccbd/Downloads/dataset6/SRD/test/shadow_free',
        'TEST_MASK_PATH': '/home/ccbd/Downloads/dataset6/SRD/test/testmask',
        'MODEL_PATH': './shadowformer_model.pth',
        'RESULTS_DIR': './test_results'
    }

# Model parameters
IMAGE_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"Using device: {DEVICE}")

# Safe image loading function with error handling
def safe_load_image(path, mode='RGB'):
    try:
        return Image.open(path).convert(mode)
    except UnidentifiedImageError:
        logger.warning(f"Could not identify image file: {path}")
        return None
    except OSError as e:
        logger.warning(f"Error opening image {path}: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error opening {path}: {str(e)}")
        return None

# Robust test dataset
class ShadowRemovalTestDataset(Dataset):
    def __init__(self, shadow_dir, shadow_free_dir, mask_dir=None, transform=None):
        """
        Initialize the test dataset with flexible file handling.
        
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
            
            # Try different naming conventions for test data
            possible_free_names = [
                f"{shadow_name}_free{os.path.splitext(shadow_basename)[1]}",       # IMG_XXX_free.jpg
                f"{shadow_name}_no_shadow{os.path.splitext(shadow_basename)[1]}",  # IMG_XXX_no_shadow.jpg
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
        
        logger.info(f"Found {len(self.pairs)} valid image pairs for testing")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        shadow_path, shadow_free_path, mask_path = self.pairs[idx]
        img_name = os.path.basename(shadow_path)
        
        try:
            # Load images with safe loader
            shadow_img = safe_load_image(shadow_path, 'RGB')
            shadow_free_img = safe_load_image(shadow_free_path, 'RGB')
            
            # Skip if either image couldn't be loaded
            if shadow_img is None or shadow_free_img is None:
                logger.warning(f"Skipping invalid image pair: {shadow_path}, {shadow_free_path}")
                return None, None, None, img_name, (0, 0)
            
            # Store original size for later
            original_size = shadow_img.size
            
            # Create or load mask
            if mask_path and os.path.exists(mask_path):
                mask_img = safe_load_image(mask_path, 'L')
                if mask_img is None:
                    # Create dummy mask if loading failed
                    mask_img = Image.new('L', shadow_img.size, 255)
            else:
                # Create dummy mask if not available
                mask_img = Image.new('L', shadow_img.size, 255)
            
            # Apply transformations
            if self.transform:
                shadow_img_tensor = self.transform(shadow_img)
                shadow_free_img_tensor = self.transform(shadow_free_img)
                
                # Make sure mask is resized to match the transformed image size
                mask_transform = transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor()
                ])
                mask_img_tensor = mask_transform(mask_img)
                
                # Normalize mask to be binary (0 or 1)
                mask_img_tensor = (mask_img_tensor > 0.5).float()
            
            return shadow_img_tensor, shadow_free_img_tensor, mask_img_tensor, img_name, original_size
        
        except Exception as e:
            logger.error(f"Error loading test image pair: {shadow_path}, {shadow_free_path}, {str(e)}")
            # Return empty tensors in case of errors
            return None, None, None, img_name, (0, 0)

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

# Transform for data preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Helper function to denormalize images
def denormalize(img):
    return (img * 0.5 + 0.5).clamp(0, 1)

# Custom collate function to handle potential size mismatches
def custom_collate(batch):
    # Filter out None values (failed loads)
    valid_batch = [(s, sf, m, name, size) for s, sf, m, name, size in batch if s is not None]
    
    # If no valid items, return empty batch
    if not valid_batch:
        return None, None, None, [], []
    
    shadow_imgs = torch.stack([item[0] for item in valid_batch])
    shadow_free_imgs = torch.stack([item[1] for item in valid_batch])
    masks = torch.stack([item[2] for item in valid_batch])
    img_names = [item[3] for item in valid_batch]
    original_sizes = [item[4] for item in valid_batch]
    
    return shadow_imgs, shadow_free_imgs, masks, img_names, original_sizes

# Function to save an image safely with error handling
def safe_save_image(img, save_path):
    try:
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        cv2.imwrite(save_path, img)
        return True
    except Exception as e:
        logger.error(f"Error saving image to {save_path}: {str(e)}")
        return False

# Function to evaluate model performance
def evaluate_model(model, test_loader, results_dir):
    model.eval()
    
    # Metrics
    psnr_scores = []
    ssim_scores = []
    
    # Create directory for output images
    os.makedirs(os.path.join(results_dir, "comparison"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "output_only"), exist_ok=True)
    
    # Results tables
    results_data = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Skip empty batches
            if batch[0] is None:
                continue
            
            shadow_imgs, shadow_free_imgs, masks, img_names, original_sizes = batch
            
            # Process batch
            try:
                # Move to device
                shadow_imgs = shadow_imgs.to(DEVICE)
                shadow_free_imgs = shadow_free_imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                
                # Forward pass
                outputs = model(shadow_imgs, masks)
                
                # Process each image in batch
                for i in range(shadow_imgs.size(0)):
                    img_name = img_names[i]
                    original_size = original_sizes[i]
                    
                    # Convert tensors to numpy arrays
                    shadow_img = denormalize(shadow_imgs[i]).cpu().permute(1, 2, 0).numpy()
                    shadow_free_img = denormalize(shadow_free_imgs[i]).cpu().permute(1, 2, 0).numpy()
                    output_img = denormalize(outputs[i]).cpu().permute(1, 2, 0).numpy()
                    
                    # Calculate metrics - with data_range parameter
                    img_psnr = psnr(shadow_free_img, output_img, data_range=1.0)
                    img_ssim = ssim(shadow_free_img, output_img, data_range=1.0, channel_axis=2)
                    
                    # Append to overall metrics
                    psnr_scores.append(img_psnr)
                    ssim_scores.append(img_ssim)
                    
                    # Store results
                    results_data.append({
                        "image": img_name,
                        "psnr": img_psnr,
                        "ssim": img_ssim
                    })
                    
                    # Save comparison visualization with error handling
                    try:
                        plt.figure(figsize=(15, 5))
                        
                        plt.subplot(1, 3, 1)
                        plt.imshow(shadow_img)
                        plt.title("Shadow Image")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 2)
                        plt.imshow(output_img)
                        plt.title(f"Predicted (PSNR: {img_psnr:.2f}, SSIM: {img_ssim:.4f})")
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.imshow(shadow_free_img)
                        plt.title("Ground Truth")
                        plt.axis('off')
                        
                        plt.tight_layout()
                        comparison_path = os.path.join(results_dir, "comparison", f"comparison_{img_name}")
                        plt.savefig(comparison_path)
                        plt.close()
                    except Exception as e:
                        logger.error(f"Error saving comparison visualization for {img_name}: {str(e)}")
                    
                    # Save output image alone with error handling
                    try:
                        output_save = (output_img * 255).astype(np.uint8)
                        output_save_path = os.path.join(results_dir, "output_only", f"output_{img_name}")
                        
                        # Convert to BGR for OpenCV
                        output_save_bgr = cv2.cvtColor(output_save, cv2.COLOR_RGB2BGR)
                        
                        # Resize to original size if needed
                        if original_size != (0, 0) and (original_size[0] != IMAGE_SIZE or original_size[1] != IMAGE_SIZE):
                            output_save_bgr = cv2.resize(output_save_bgr, original_size)
                        
                        # Save safely
                        safe_save_image(output_save_bgr, output_save_path)
                    except Exception as e:
                        logger.error(f"Error saving output image for {img_name}: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error processing images {img_names}: {str(e)}")
                continue
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
    
    # Save metrics to file
    try:
        with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Total processed images: {len(psnr_scores)}\n")
            f.write("\nIndividual image metrics:\n")
            
            # Sort by PSNR
            results_data.sort(key=lambda x: x["psnr"], reverse=True)
            
            for item in results_data:
                f.write(f"Image: {item['image']}, PSNR: {item['psnr']:.4f}, SSIM: {item['ssim']:.4f}\n")
    except Exception as e:
        logger.error(f"Error saving metrics file: {str(e)}")
    
    logger.info(f"Testing complete. Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")
    logger.info(f"Successfully processed {len(psnr_scores)} images")
    logger.info(f"Results saved to {results_dir}")
    
    return avg_psnr, avg_ssim

# Function to test a specific image
def test_single_image(model, image_path, output_path=None, results_dir='./test_results'):
    """Test the model on a single image without ground truth."""
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None
    
    # Create single image output directory
    single_out_dir = os.path.join(results_dir, "single")
    os.makedirs(single_out_dir, exist_ok=True)
    
    # Load and process the image
    try:
        shadow_img = safe_load_image(image_path, 'RGB')
        if shadow_img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
            
        original_size = shadow_img.size
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        shadow_img_tensor = transform(shadow_img).unsqueeze(0).to(DEVICE)
        
        # Create dummy mask (all ones)
        mask = torch.ones(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        
        # Generate shadow-free image
        model.eval()
        with torch.no_grad():
            output = model(shadow_img_tensor, mask)
        
        # Convert output to image
        output_img = denormalize(output[0]).cpu().permute(1, 2, 0).numpy()
        output_save = (output_img * 255).astype(np.uint8)
        
        # Resize to original dimensions
        output_save = cv2.resize(output_save, original_size)
        
        # Create output path if not provided
        if output_path is None:
            basename = os.path.basename(image_path)
            output_path = os.path.join(single_out_dir, f"output_{basename}")
        
        # Save output image
        output_save_bgr = cv2.cvtColor(output_save, cv2.COLOR_RGB2BGR)
        if safe_save_image(output_save_bgr, output_path):
            logger.info(f"Single image processed: {os.path.basename(image_path)}")
            logger.info(f"Output saved to: {output_path}")
        
        # Visualize input and output
        try:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(shadow_img))
            plt.title("Input Shadow Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(output_img)
            plt.title("Predicted Shadow-Free Image")
            plt.axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(os.path.dirname(output_path), f"comparison_{os.path.basename(image_path)}")
            plt.savefig(comparison_path)
            plt.close()
        except Exception as e:
            logger.error(f"Error saving comparison for single image: {str(e)}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing single image: {str(e)}")
        return None

# Main function
def main():
    # Get default paths
    default_paths = get_default_paths()
    
    parser = argparse.ArgumentParser(description="Test Shadow Removal Model")
    parser.add_argument("--model_path", type=str, default=default_paths['MODEL_PATH'], help="Path to trained model")
    parser.add_argument("--test_dir", type=str, default=default_paths['TEST_SHADOW_PATH'], help="Path to test directory")
    parser.add_argument("--test_free_dir", type=str, default=default_paths['TEST_SHADOW_FREE_PATH'], help="Path to shadow-free test directory")
    parser.add_argument("--test_mask_dir", type=str, default=default_paths['TEST_MASK_PATH'], help="Path to test mask directory")
    parser.add_argument("--results_dir", type=str, default=default_paths['RESULTS_DIR'], help="Directory to save results")
    parser.add_argument("--single_image", type=str, default=None, help="Path to a single image to test")
    parser.add_argument("--output_path", type=str, default=None, help="Output path for single image test")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    
    args = parser.parse_args()
    
    # Use parsed arguments
    model_path = args.model_path
    test_shadow_path = args.test_dir
    test_shadow_free_path = args.test_free_dir
    test_mask_path = args.test_mask_dir
    results_dir = args.results_dir
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    
    # Check test directories exist
    if not os.path.exists(test_shadow_path):
        logger.error(f"Test shadow directory not found: {test_shadow_path}")
        return
    
    if not os.path.exists(test_shadow_free_path):
        logger.error(f"Test shadow-free directory not found: {test_shadow_free_path}")
        return
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = ShadowFormer(
        img_size=IMAGE_SIZE, 
        patch_size=16, 
        in_chans=3, 
        embed_dim=512, 
        depth=8, 
        num_heads=8
    ).to(DEVICE)
    
    # Load model weights with error handling
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        return
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # Test single image if provided
    if args.single_image:
        if not os.path.exists(args.single_image):
            logger.error(f"Single image not found: {args.single_image}")
            return
            
        test_single_image(model, args.single_image, args.output_path, results_dir)
        return
    
    # Test on dataset
    logger.info("Creating test dataset...")
    test_dataset = ShadowRemovalTestDataset(
        test_shadow_path,
        test_shadow_free_path,
        test_mask_path,
        transform=transform
    )
    
    # Check if dataset is empty
    if len(test_dataset) == 0:
        logger.error("No valid image pairs found for testing")
        return
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    avg_psnr, avg_ssim = evaluate_model(model, test_loader, results_dir)
    
    logger.info("Testing complete")
    logger.info(f"Average PSNR: {avg_psnr:.4f}")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")

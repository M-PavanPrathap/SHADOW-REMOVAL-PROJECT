# Shadow Removal from Images using Deep Learning

A PyTorch implementation of ShadowFormer, a Vision Transformer-based approach for automatic shadow removal from images. This project was developed as part of a summer internship focusing on Machine Learning and Deep Learning techniques for image processing.

## 🎯 Project Overview

This repository contains a complete implementation of a shadow removal system that can:
- Train a Vision Transformer model to remove shadows from images
- Test the trained model on new images or datasets
- Evaluate performance using PSNR and SSIM metrics
- Handle various image formats and naming conventions
- Process single images or entire datasets

## 🏗️ Architecture

The project implements **ShadowFormer**, a Vision Transformer-based architecture consisting of:

### Core Components:
- **Patch Embedding**: Converts input images into patch tokens
- **Transformer Encoder**: Multi-head self-attention blocks for feature extraction
- **Decoder Network**: Convolutional transpose layers for image reconstruction
- **Multi-scale Loss Function**: Combines L1, MSE, and gradient losses

### Model Specifications:
- **Input Size**: 256×256 RGB images
- **Patch Size**: 16×16 pixels
- **Embedding Dimension**: 512
- **Transformer Depth**: 8 layers
- **Attention Heads**: 8 heads
- **Parameters**: ~50M parameters

## 📁 Project Structure

```
shadow-removal/
├── traincode.py          # Training script
├── testcode.py           # Testing and evaluation script
├── checkpoints/          # Model checkpoints (auto-created)
├── outputs/             # Training visualization outputs
├── test_results/        # Test results and metrics
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/shadow-removal.git
cd shadow-removal

# Create virtual environment
python -m venv shadow_env
source shadow_env/bin/activate  # On Windows: shadow_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install opencv-python pillow matplotlib
pip install numpy scikit-image tqdm
pip install argparse logging glob
```

## 📊 Dataset Structure

The code expects datasets in the following structure:

```
dataset/
├── Train/
│   ├── shadow/          # Shadow images
│   ├── shadow_free/     # Corresponding shadow-free images
│   └── trainmask/       # Shadow masks (optional)
└── Test/
    ├── shadow/          # Test shadow images
    ├── shadow_free/     # Test shadow-free images (for evaluation)
    └── testmask/        # Test masks (optional)
```

### Supported Naming Conventions:
- `image_001.jpg` → `image_001_free.jpg` or `image_001_no_shadow.jpg`
- `shadow_001.jpg` → `free_001.jpg`
- Identical names in different directories

### Supported Formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- Automatic format detection

## 🚀 Usage

### Training

```bash
# Basic training with default paths
python traincode.py

# Custom training paths
python traincode.py \
    --shadow_dir /path/to/shadow/images \
    --shadow_free_dir /path/to/shadow_free/images \
    --mask_dir /path/to/masks \
    --epochs 200 \
    --batch_size 8 \
    --learning_rate 0.0001
```

#### Training Parameters:
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--image_size`: Input image size (default: 256)
- `--checkpoint_dir`: Checkpoint save directory

### Testing

```bash
# Test on dataset
python testcode.py \
    --model_path ./shadowformer_model.pth \
    --test_dir /path/to/test/shadow \
    --test_free_dir /path/to/test/shadow_free \
    --results_dir ./test_results

# Test single image
python testcode.py \
    --model_path ./shadowformer_model.pth \
    --single_image /path/to/shadow/image.jpg \
    --output_path /path/to/output.jpg
```

#### Testing Parameters:
- `--model_path`: Path to trained model
- `--test_dir`: Test shadow images directory
- `--test_free_dir`: Test shadow-free images directory (for metrics)
- `--single_image`: Path to single test image
- `--batch_size`: Testing batch size (default: 1)

## 📈 Model Performance

### Evaluation Metrics:
- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality
- **SSIM** (Structural Similarity Index): Measures structural similarity

### Expected Performance:
- Training typically converges within 100-150 epochs
- PSNR: 25-30 dB on standard datasets
- SSIM: 0.85-0.95 on standard datasets

### Output Files:
- `metrics.txt`: Detailed per-image and average metrics
- `comparison/`: Side-by-side visual comparisons
- `output_only/`: Generated shadow-free images only
- Training visualizations saved every 5 epochs

## 🔧 Key Features

### Robust Data Handling:
- Automatic file format detection
- Multiple naming convention support
- Graceful error handling for corrupted images
- Custom collate functions for batch processing

### Training Features:
- Multi-component loss function (L1 + MSE + Gradient)
- Automatic checkpointing every 10 epochs
- Best model saving based on loss
- Learning rate scheduling
- Progress visualization

### Testing Features:
- Comprehensive evaluation metrics
- Visual comparison generation
- Single image processing capability
- Original image size preservation
- Batch and individual processing modes

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   python traincode.py --batch_size 4
   ```

2. **No Valid Image Pairs Found**:
   - Check dataset directory structure
   - Verify image naming conventions
   - Ensure shadow and shadow-free images exist

3. **Model Loading Errors**:
   ```python
   # Check model path and ensure file exists
   if not os.path.exists(model_path):
       print(f"Model not found: {model_path}")
   ```

4. **Memory Issues**:
   - Reduce image size: `--image_size 128`
   - Use CPU instead of GPU: Set `CUDA_VISIBLE_DEVICES=""`

## 📝 Implementation Details

### Loss Function:
```python
total_loss = L1_loss + 0.5 * MSE_loss + 0.5 * gradient_loss
```

### Data Augmentation:
- Resize to 256×256
- Normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
- Random horizontal/vertical flips (can be added)

### Optimization:
- **Optimizer**: Adam
- **Initial LR**: 1e-4
- **Scheduler**: StepLR (γ=0.5, step_size=30)
- **Weight Decay**: Can be added for regularization

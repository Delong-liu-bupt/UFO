# **EasyAnimate UFO Plugin Training Guide**
> This guide provides comprehensive instructions for training LoRA models with EasyAnimate using concepts from UFO, as well as generating videos.


## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training Configuration](#training-configuration)
- [Training Process](#training-process)
- [Inference with Trained LoRA](#inference-with-trained-lora)
- [Troubleshooting](#troubleshooting)

## Overview

EasyAnimate supports LoRA (Low-Rank Adaptation) training for fine-tuning video generation models. This approach allows you to:
- Train on custom datasets with limited GPU memory
- Create style-specific video generation models
- Maintain the base model's capabilities while adding new features

**Key UFO Training Strategy:**
- **Training**: Use `lora_weight = 1.0` during training for maximum consistency preservation
- **Inference**: Use smaller `lora_weight` values (0.05-0.2) during inference to enhance video aesthetics and consistency

## Environment Setup

### System Requirements

**Minimum Requirements:**
- OS: Ubuntu 20.04+ or CentOS 7+
- Python: 3.10 or 3.11
- CUDA: 11.8 or 12.1
- CUDNN: 8+
- GPU Memory: 16GB+ (24GB+ recommended)
- Storage: 60GB+ available space

**Recommended Hardware:**
- GPU: NVIDIA A10 24GB, A100 40GB/80GB, or V100 16GB
- RAM: 32GB+
- Storage: SSD with 100GB+ free space

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/aigc-apps/EasyAnimate.git
cd EasyAnimate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download Base Model**
Download the EasyAnimateV5.1 model and place it in the correct directory:
```bash
mkdir -p models/Diffusion_Transformer
# Download EasyAnimateV5.1-12b-zh-InP to models/Diffusion_Transformer/
```

## Data Preparation

### Dataset Structure

Organize your training data as follows:
```
datasets/
├── internal_datasets/
│   ├── train/
│   │   ├── video001.mp4
│   │   ├── video002.mp4
│   │   ├── image001.jpg
│   │   └── ...
│   └── metadata.json
```

### Metadata Format

Create a `metadata.json` file with the following structure:
```json
[
    {
        "file_path": "train/video001.mp4",
        "text": "A detailed description of the video content",
        "type": "video"
    },
    {
        "file_path": "train/image001.jpg", 
        "text": "A detailed description of the image content",
        "type": "image"
    }
]
```

**Important Notes:**
- Use detailed, descriptive captions for better training results
- Mix both videos and images in your dataset
- Ensure video files are in supported formats (mp4, avi, mov)
- Recommended video length: 2-6 seconds at 8 fps

## Training Configuration

### Basic Configuration

The training script `scripts/train_lora.sh` contains the following key parameters:

```bash
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5.1-12b-zh-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"
```

### Key Training Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `image_sample_size` | 1024 | Resolution for image training |
| `video_sample_size` | 256 | Token size for video processing |
| `video_sample_n_frames` | 49 | Number of frames per video |
| `train_batch_size` | 1 | Batch size per GPU |
| `learning_rate` | 1e-04 | Learning rate |
| `num_train_epochs` | 100 | Number of training epochs |
| `checkpointing_steps` | 100 | Save checkpoint every N steps |

### Memory Optimization

For limited GPU memory, use these flags:
- `--low_vram`: Enables VRAM optimization
- `--gradient_checkpointing`: Reduces memory usage
- `--mixed_precision="bf16"`: Uses mixed precision training

## Training Process

### 1. Prepare Training Script

Edit `scripts/train_lora.sh` to match your dataset paths:
```bash
export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5.1-12b-zh-InP"
export DATASET_NAME="datasets/your_dataset/"
export DATASET_META_NAME="datasets/your_dataset/metadata.json"
```

### 2. Start Training

Execute the training script:
```bash
bash scripts/train_ufo.sh
```

### 3. Monitor Training

Training outputs will be saved to `output_dir/`:
- **Checkpoints**: `checkpoint-{step}.safetensors`
- **Training logs**: Console output and tensorboard logs
- **Validation samples**: Generated during training (if validation enabled)

### 4. Training Completion

Upon completion, you'll find:
- Final LoRA weights: `output_dir/checkpoint-{final_step}.safetensors`
- Training state: `output_dir/checkpoint-{final_step}/` (if `--save_state` enabled)

## Inference with Trained UFO LoRA

### Text-to-Video Generation

1. **Edit `predict_t2v.py`:**
```python
# Set your LoRA path
lora_path = "output_dir/checkpoint-5000.safetensors"

# Use smaller weight for better aesthetics (0.05-0.2 range)
lora_weight = 0.15  # Instead of default 0.60

# Set your prompt
prompt = "Your custom prompt here"
```

2. **Run inference:**
```bash
python predict_t2v.py
```

### Image-to-Video Generation

1. **Edit `predict_i2v.py`:**
```python
# Set your LoRA path
lora_path = "output_dir/checkpoint-5000.safetensors"

# Use smaller weight for better aesthetics
lora_weight = 0.15

# Set your input image and prompt
validation_image_start = "path/to/your/image.jpg"
prompt = "Your custom prompt here"
```

2. **Run inference:**
```bash
python predict_i2v.py
```


## Troubleshooting

### Common Issues

**1. GPU Memory Errors**
```bash
# Solution: Enable memory optimizations
--low_vram --gradient_checkpointing --mixed_precision="bf16"
```

**2. Training Loss Not Decreasing**
- Check data quality and captions
- Reduce learning rate (try 5e-05)
- Increase batch size if possible

**3. Poor Generation Quality**
- Try different LoRA weights (0.05-0.2 range)
- Check if training converged properly
- Verify prompt quality and relevance

**4. CUDA Out of Memory During Inference**
- Use `GPU_memory_mode = "model_cpu_offload_and_qfloat8"`
- Reduce `video_length` or `sample_size`
- Close other GPU processes

### Performance Optimization

**For 16GB GPU:**
- Use `model_cpu_offload_and_qfloat8` mode
- Reduce batch size to 1
- Enable gradient checkpointing

**For 24GB+ GPU:**
- Use `model_cpu_offload` mode
- Can increase batch size to 2-4
- Full precision training possible

## Acknowledgements and Citation

> This codebase mainly reuses [https://github.com/aigc-apps/EasyAnimate](https://github.com/aigc-apps/EasyAnimate), and we sincerely thank them for their contribution.
>
> **If you use this code, please cite:**
>
> ```bibtex
> @inproceedings{liu2025ufo,
>   title={UFO: Enhancing Diffusion-Based Video Generation with a Uniform Frame Organizer},
>   author={Liu, Delong and Hou, Zhaohui and Zhan, Mingjie and Han, Shihao and Zhao, Zhicheng and Su, Fei},
>   booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
>   volume={39},
>   number={5},
>   pages={5388--5396},
>   year={2025}
> }
> ```

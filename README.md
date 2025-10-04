# Privacy-Preserving Blurring Models for Sports Video Analytics

A comprehensive investigation into privacy-preserving techniques for sports video analysis, focusing on maintaining player dependency information while anonymizing facial features.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Approaches Implemented](#approaches-implemented)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Experimental Results](#experimental-results)
- [Computational Requirements & Limitations](#computational-requirements--limitations)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## Overview

This project explores three distinct approaches to face anonymization in sports videos while preserving critical analytical features such as player positioning, pose, and movement patterns. The goal is to enable privacy-compliant sports analytics without sacrificing the quality of downstream tasks like pose estimation and player tracking.

### Research Question
**Can we anonymize player identities in sports footage while maintaining sufficient visual information for accurate pose estimation and movement analysis?**

---

### Key Directories

- **`data/`**: Contains sample images for testing and demonstration
- **`models_/`**: Stores pre-trained model weights and architecture files
- **`val2017_subset/`**: Subset of COCO 2017 validation set (~950 images) used for evaluation
- **`val2017_subset_regular_approach/`**: Output directory for traditionally blurred images
- **`results/`**: Contains evaluation metrics, comparison charts, and analysis reports

---

## Approaches Implemented

### 1. Traditional Detection-Based Approach (`regular_approach.py`)

**Method**: Classical computer vision techniques using OpenCV and NumPy

**Key Features**:
- Multiple Haar cascade classifiers (frontal, alternative frontal, profile)
- Multi-scale face detection with varied scale factors (1.05, 1.1, 1.2, 1.3)
- Profile detection (both left and right orientations via image flipping)
- Non-maximum suppression to eliminate duplicate detections
- Gaussian blur with 10% padding around detected faces
- Histogram equalization for improved detection in varied lighting

**Operational Modes**:
- Single image processing
- Batch video processing
- Real-time webcam feed

**Advantages**:
- ✅ Lightweight and fast (runs on CPU)
- ✅ No training required
- ✅ Works on any hardware
- ✅ Real-time capable

**Limitations**:
- ❌ Lower detection accuracy (especially for partially occluded or rotated faces)
- ❌ Aggressive blurring may remove too much visual information
- ❌ No preservation of facial structure or expression
- ❌ Can blur non-face regions or miss faces entirely

**Usage Example**:
```python
from regular_approach import RegularApproach

anonymizer = RegularApproach(output_dir="./output")

# Process single image
anonymizer.run(mode="image", file_path="data/player.jpg")

# Process video
anonymizer.run(mode="video", file_path="video/match.mp4")

# Real-time webcam
anonymizer.run(mode="webcam")
```

---

### 2. GAN-Based Face Replacement (`gan_approach_1.py`)

**Method**: Generative Adversarial Network for identity-preserving face generation

**Repository**: [GANonymization](https://github.com/hcmlab/GANonymization)

**Key Features**:
- Pix2Pix GAN architecture for face generation
- Facial landmark detection (478 points) for structural preservation
- Face segmentation for precise region isolation
- Classifier training for identity verification
- Supports image and directory batch processing

**Pipeline**:
1. Face detection and alignment
2. Facial landmark extraction
3. Face segmentation mask generation
4. GAN-based face synthesis
5. Seamless blending back into original image

**Advantages**:
- ✅ Preserves facial structure and expressions
- ✅ Maintains head positioning and eye direction
- ✅ More natural-looking results than blurring
- ✅ Retains analytical value (pose, orientation)

**Limitations**:
- ❌ **Requires GPU for practical use** (extremely slow on CPU)
- ❌ Limited to image processing (frame-by-frame for video)
- ❌ Heavy computational requirements
- ❌ Uses a pre-trained model

**License**: MIT

**Usage Example**:
```python
from gan_approach_1 import FaceAnonymizer

# Preprocess dataset
FaceAnonymizer.preprocess(
    input_path="data/dataset/",
    img_size=512,
    align=True
)

# Anonymize single image
FaceAnonymizer.anonymize_image(
    model_file="models_/pre_trained_models/pix2pix.pth",
    input_file="data/player.jpg",
    output_file="output/player_anon.jpg",
    device=0  # GPU device ID
)

# Batch process directory
anonymizer = FaceAnonymizer()
anonymizer.anonymize_directory(
    model_file="models_/pre_trained_models/pix2pix.pth",
    input_directory="data/dataset/",
    output_directory="output/anon_dataset/",
    device=0
)
```

---

### 3. Advanced Diffusion-Based Anonymization (`demo.ipynb`)

**Method**: Stable Diffusion with ReferenceNet for identity-preserving facial replacement

**Repository**: [face_anon_simple](https://github.com/hanweikung/face_anon_simple)

**Key Features**:
- Uses Stable Diffusion 2.1 as backbone
- ReferenceNet architecture for facial feature control
- CLIP vision encoder for semantic understanding
- Adjustable anonymization degree (0.0 = face swap, 1.25+ = full anonymization)
- Supports both aligned and unaligned faces
- Face swap capability between images

**Technical Components**:
- **UNet**: Main denoising network
- **ReferenceNet**: Preserves facial structure and attributes
- **Conditioning ReferenceNet**: Controls expression and pose
- **VAE**: Image encoding/decoding
- **CLIP**: Semantic feature extraction

**Advantages**:
- ✅ State-of-the-art quality and realism
- ✅ Fine control over anonymization level
- ✅ Preserves expressions, gaze, and pose
- ✅ Can handle multiple unaligned faces
- ✅ Face swapping capability for creative applications

**Limitations**:
- ❌ **Extremely GPU-intensive** (requires significant VRAM)
- ❌ Very slow inference (5-200 steps per image)
- ❌ Model weights are very large (several GB)
- ❌ Complex setup with multiple dependencies
- ❌ Impractical for real-time or video processing without high-end hardware

**License**: AGPL v3.0 (if deployed as web service, modifications must be open-sourced)

**Usage Example**:
```python
# See demo.ipynb for full setup

# Quick anonymization (aligned face)
anon_image = pipe(
    source_image=original_image,
    conditioning_image=original_image,
    num_inference_steps=5,
    guidance_scale=3.0,
    anonymization_degree=1.25,
    width=256,
    height=256,
).images[0]

# Unaligned faces (with face detection)
anon_image = anonymize_faces_in_image(
    image=original_image,
    face_alignment=fa,
    pipe=pipe,
    generator=generator,
    num_inference_steps=25,
    anonymization_degree=1.25,
)

# Face swap between two images
swap_image = pipe(
    source_image=person_a,
    conditioning_image=person_b,
    anonymization_degree=0.0,  # Pure swap, no anonymization
).images[0]
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for GAN approaches, 8GB+ VRAM)
- 16GB+ RAM recommended

### Basic Setup (Traditional Approach Only)

```bash
# Clone the repository
git clone <repository-url>
cd privacy-preserving-blurring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install basic requirements
pip install opencv-python numpy

# Run traditional approach
python main.py
```

### Full Setup (All Approaches)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate face-anon

# Or use pip
pip install -r requirements.txt

# Additional dependencies for diffusion model
pip install torch torchvision transformers diffusers huggingface_hub
pip install face-alignment

# Download pre-trained models (see model documentation)
```

### Model Downloads

**GAN Approach**:
- Download pre-trained Pix2Pix weights from [GANonymization repo]
- Place in `models_/pre_trained_models/`

**Diffusion Approach**:
- Models auto-download from HuggingFace on first run
- Requires ~6GB storage space
- Model ID: `hkung/face-anon-simple`

---

## Usage Guide

### Quick Start: Traditional Approach

```python
from regular_approach import RegularApproach

# Initialize
anonymizer = RegularApproach(output_dir="./output")

# Process single image
anonymizer.run(mode="image", file_path="data/erling.jpg")

# Process video file
anonymizer.run(mode="video", file_path="your_file.mp4")

# Test with webcam
anonymizer.run(mode="webcam")
```

### Running Evaluations

```bash
# Process COCO subset with traditional approach
python main.py --mode batch --input val2017_subset/ --output val2017_subset_regular_approach/

# Run pose estimation evaluation
python evaluate_pose.py --original val2017_subset/ --blurred val2017_subset_regular_approach/
```

### Using the Jupyter Notebook (Diffusion Approach)

```bash
# Start Jupyter
jupyter notebook

# Open demo.ipynb
# Follow cells sequentially:
# 1. Load models
# 2. Choose anonymization mode (aligned/unaligned/face swap)
# 3. Process images
# 4. View results
```

---

## Experimental Results

### Evaluation Dataset

**Dataset**: COCO 2017 Keypoints [val2017 subset](https://drive.google.com/drive/u/1/folders/1K0kLk9CSD6NQRDag5pZYNIgd521GszDx)
- **Size**: ~950 images (sampled from 5,000 image validation set)
- **Annotations**: 17 human keypoints per person (nose, eyes, shoulders, elbows, hips, knees, ankles)
- **Purpose**: Evaluate pose estimation performance on anonymized images

### Evaluation Metrics

Performance measured using COCO keypoints benchmark:

- **AP@[0.50:0.95]**: Mean Average Precision across IoU thresholds 0.50-0.95 (primary metric)
- **AP@0.50**: Average Precision at IoU threshold 0.50 (lenient matching)
- **AP@0.75**: Average Precision at IoU threshold 0.75 (strict matching)
- **AR@[0.50:0.95]**: Mean Average Recall across IoU thresholds

### Pose Estimation Model

**MoveNet Lightning**: Fast, efficient pose estimation model
- Chosen for CPU-friendly evaluation
- Inference performed on CPU only

### Results Summary

#### Original Dataset Performance

| Metric | Score | Notes |
|--------|-------|-------|
| AP@[0.50:0.95] | 0.010 | Primary accuracy metric |
| AP@0.50 | 0.010 | Lenient threshold |
| AP@0.75 | 0.010 | Strict threshold |
| AR@[0.50:0.95] | 0.008 | Detection recall |

**Note**: These scores are significantly lower than MoveNet's reported baseline (~0.20-0.30 AP) due to:
- Small random subset of COCO dataset
- CPU-only inference (no GPU acceleration)
- Serves as functional baseline for comparison

#### Blurred Dataset Performance (Traditional Approach)

| Metric | Score | Change from Original |
|--------|-------|---------------------|
| AP@[0.50:0.95] | 0.007 | -30% |
| AP@0.50 | 0.015 | +50% |
| AP@0.75 | 0.005 | -50% |
| AR@[0.50:0.95] | 0.006 | -25% |

### Key Findings

1. **Performance Degradation**: Blurring caused a 30% decrease in overall AP, with strict threshold (AP@0.75) dropping by 50%

2. **Inconsistent Results**: Lenient threshold (AP@0.50) surprisingly improved, suggesting detection variability

3. **Blurring Accuracy Issues**:
   - Pipeline sometimes failed to detect and blur all faces
   - Occasionally blurred non-face regions (false positives)
   - Imperfections likely contributed to performance variability

4. **Literature Support**: 
   - Research shows realistic face anonymization (GANs/diffusion) substantially reduces performance loss compared to traditional blurring
   - Studies indicate models trained on face-blurred images achieve 3D keypoint accuracy comparable to unblurred data

### Supporting Research

#### DeepPrivacy2 Study
**Finding**: Realistic face anonymization techniques (like GANs) significantly mitigate performance degradation compared to conventional blurring methods, though they don't fully substitute unmodified data.

**Source**: [Realistic Face Anonymization Performance](https://arxiv.org/pdf/2306.05135)

#### Human Pose Estimation Impact
**Finding**: Empirical results show models trained on face-blurred images achieve 3D keypoint localization accuracy comparable to models trained on original unblurred datasets.

**Source**: [Impact on Human Pose Estimation](https://pmc.ncbi.nlm.nih.gov/articles/PMC9739378/pdf/sensors-22-09376.pdf)

---

## 💻 Computational Requirements & Limitations

### Hardware Requirements by Approach

| Approach | CPU | GPU | RAM | VRAM | Real-time Capable |
|----------|-----|-----|-----|------|-------------------|
| Traditional (Haar) | ✅ Any | ❌ Not needed | 4GB | N/A | ✅ Yes |
| GAN (Pix2Pix) | ⚠️ Very slow | ✅ Required | 8GB+ | 4GB+ | ❌ No |
| Diffusion (ReferenceNet) | ❌ Impractical | ✅ Required | 16GB+ | 8GB+ | ❌ No |


**Impact on Testing**:
- Even with optimizations (attention slicing, VAE slicing, 256x256 resolution, 5 inference steps), processing was prohibitively slow
- Could only test on handful of sample images
- Full dataset evaluation estimated at 50+ hours on CPU
- **This approach is effectively unusable without GPU access**

### Optimization Attempts

We implemented several CPU optimization strategies:

```python
# Enable memory-efficient attention
pipe.enable_attention_slicing(1)

# Enable VAE slicing
pipe.enable_vae_slicing()

# Use half-precision (float16) instead of float32
torch_dtype=torch.float16

# Reduce image resolution
width=256, height=256  # Instead of 512x512

# Minimize inference steps
num_inference_steps=5  # Instead of 25-200
```

**Result**: Even with all optimizations, diffusion approach remained too slow for practical CPU use.

### Testing Limitations

Due to computational constraints, we were unable to:

1. ✗ Run full quantitative evaluation on GAN approach
2. ✗ Test diffusion approach on COCO subset at all
3. ✗ Compare all three approaches on identical test sets
4. ✗ Perform video processing tests on GAN/diffusion methods
5. ✗ Measure real-world inference time on large datasets

### What We Could Test

✓ Traditional approach: Full evaluation on 950-image subset  
✓ Diffusion approach: Qualitative assessment on ~5 sample images  
✓ Visual quality comparison on small sample set  

---

## References

### Repositories Used

1. **GANonymization** (GAN Approach 1)
   - Repository: [https://github.com/hcmlab/GANonymization](https://github.com/hcmlab/GANonymization)
   - License: MIT
   - Used for: Pix2Pix-based face replacement

2. **face_anon_simple** (Diffusion Approach)
   - Repository: [https://github.com/hanweikung/face_anon_simple](https://github.com/hanweikung/face_anon_simple)
   - License: AGPL v3.0
   - Used for: Identity-preserving facial replacement with Stable Diffusion

### Academic Papers

3. **DeepPrivacy2**: Realistic Face Anonymization Performance
   - Paper: [https://arxiv.org/pdf/2306.05135](https://arxiv.org/pdf/2306.05135)
   - Key Finding: GAN-based anonymization reduces performance loss vs. blurring

4. **Impact on Human Pose Estimation**
   - Paper: [https://pmc.ncbi.nlm.nih.gov/articles/PMC9739378/pdf/sensors-22-09376.pdf](https://pmc.ncbi.nlm.nih.gov/articles/PMC9739378/pdf/sensors-22-09376.pdf)
   - Key Finding: Face-blurred images maintain comparable pose estimation accuracy

### Datasets

5. **COCO 2017 Keypoints Dataset**
   - Source: [https://www.kaggle.com/datasets/sabahesaraki/2017-2017](https://www.kaggle.com/datasets/sabahesaraki/2017-2017)
   - Used for: Pose estimation evaluation benchmark

---

## License

This project incorporates code from multiple sources with different licenses:

- **Traditional Approach** (regular_approach.py): Original implementation, MIT License
- **GAN Approach** (gan_approach_1.py): Based on GANonymization (MIT License)
- **Diffusion Approach** (demo.ipynb): Based on face_anon_simple (AGPL v3.0)

Please refer to individual source repositories for specific licensing terms.

---

## Acknowledgments

- Thanks to the authors of GANonymization and face_anon_simple for open-sourcing their implementations
- COCO dataset maintainers for providing comprehensive pose estimation annotations
- DeepPrivacy2 and related research teams for advancing privacy-preserving computer vision

---


### Summary of Findings

This project demonstrates that **privacy-preserving face anonymization in sports video analytics is feasible**, but comes with significant tradeoffs:

1. **Traditional blurring** is fast and accessible but causes measurable accuracy loss and may not preserve sufficient analytical detail

2. **GAN-based approaches** offer better preservation of facial structure and analytical features, but require substantial computational resources

3. **Diffusion-based methods** provide state-of-the-art quality and control, but are currently impractical without high-end GPU hardware

4. **Research evidence** supports that sophisticated anonymization methods can maintain pose estimation accuracy comparable to original images
---

*Last updated: October 2025*
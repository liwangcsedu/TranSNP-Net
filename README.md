# TranSNP-Net

TranSNP-Net is a deep learning model for salient object detection, combining Transformer architecture with depth information for enhanced performance.

## Project Structure

```
TranSNP-Net/
├── models/                 # Model architecture files
│   ├── TranSNP_Net.py     # Main model implementation
│   ├── SwinTransformers.py # Swin Transformer implementation
│   └── swin_base_patch4_window12_384_22k.pth # Pre-trained weights
├── data/                  # Data loading and processing
├── utils/                 # Utility functions
├── cpts/                  # Checkpoints directory
├── evaluation/           # Evaluation metrics and tools
├── test_maps/           # Test results
├── score/               # Scoring results
├── tempdata/           # Temporary data storage
├── train.py            # Training script
└── test.py             # Testing script
```

## Features

- Implements a novel architecture combining Transformer with depth information
- Uses Swin Transformer as backbone
- Deep supervision for better training
- Multi-scale feature fusion
- Depth-aware feature extraction

## Requirements

- Python 3.x
- PyTorch
- CUDA (for GPU acceleration)
- TensorboardX
- NumPy
- OpenCV

## Training

To train the model:

```bash
python train.py
```

Key training parameters can be configured in `utils/options.py`.

## Testing

To test the model:

```bash
python test.py
```

## Model Architecture

The model consists of:
- Swin Transformer backbone
- Depth-aware feature extraction
- Multi-scale feature fusion
- Deep supervision mechanism

## Performance

The model achieves competitive results on salient object detection benchmarks using both RGB and depth information.


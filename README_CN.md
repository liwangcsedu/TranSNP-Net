# TranSNP-Net

TranSNP-Net是一个用于显著性目标检测的深度学习模型，它结合了Transformer架构和深度信息以提升性能。

## 项目结构

```
TranSNP-Net/
├── models/                 # 模型架构文件
│   ├── TranSNP_Net.py     # 主要模型实现
│   ├── SwinTransformers.py # Swin Transformer实现
│   └── swin_base_patch4_window12_384_22k.pth # 预训练权重
├── data/                  # 数据加载和处理
├── utils/                 # 工具函数
├── cpts/                  # 检查点目录
├── evaluation/           # 评估指标和工具
├── test_maps/           # 测试结果
├── score/               # 评分结果
├── tempdata/           # 临时数据存储
├── train.py            # 训练脚本
└── test.py             # 测试脚本
```

## 特性

- 实现了结合Transformer和深度信息的新型架构
- 使用Swin Transformer作为骨干网络
- 深度监督以实现更好的训练效果
- 多尺度特征融合
- 深度感知特征提取

## 环境要求

- Python 3.x
- PyTorch
- CUDA（用于GPU加速）
- TensorboardX
- NumPy
- OpenCV

## 训练

训练模型：

```bash
python train.py
```

主要训练参数可在`utils/options.py`中配置。

## 测试

测试模型：

```bash
python test.py
```

## 模型架构

模型包含以下组件：
- Swin Transformer骨干网络
- 深度感知特征提取
- 多尺度特征融合
- 深度监督机制

## 性能

该模型在显著性目标检测基准测试中，通过结合RGB和深度信息，取得了具有竞争力的结果。

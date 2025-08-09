# [AAAI 2025] UFO: Enhancing Diffusion-Based Video Generation with a Uniform Frame Organizer

## News
* **[2025.08.10]** Code released! Complete training and inference implementation now available.
* [2024.12.12] Repo is created. Code will come soon.

## Abstract

Recently, diffusion-based video generation models have achieved significant success. However, existing models often suffer from issues like weak consistency and declining image quality over time. To overcome these challenges, inspired by aesthetic principles, we propose a non-invasive plug-in called **Uniform Frame Organizer (UFO)**, which is compatible with any diffusion-based video generation model. The UFO comprises a series of adaptive adapters with adjustable intensities, which can significantly enhance the consistency between the foreground and background of videos and improve image quality without altering the original model parameters when integrated.

## Method Summary

**UFO** is a lightweight, non-invasive plug-in that addresses video generation consistency issues through:

- **Adaptive Adapters**: Minimal-parameter adapters (only 0.005× of original model size) that learn to identify and correct inconsistencies
- **Adjustable Intensity**: Tunable parameter α for balancing consistency enhancement and motion preservation
- **Training Strategy**: Use α=1.0 during training for maximum consistency, then α=0.05-0.2 during inference for optimal aesthetics
- **Model Agnostic**: Compatible with any diffusion-based video generation model without parameter modification
- **Direct Transferability**: Trained UFO can be directly applied across different models of the same specification

## Demo Results

![UFO Results](asset/UFO_ResultsVideo.mp4)

*UFO significantly enhances video consistency and quality while preserving natural motion dynamics.*

## Implementation

While the original paper validated UFO on older video generation models, **this open-source implementation uses the latest EasyAnimate model** for better performance and compatibility. EasyAnimate provides superior baseline quality compared to the models used in the paper's experiments.

### Quick Start

For detailed training and inference instructions, please refer to our comprehensive guide:

**📖 [Complete Training & Inference Guide](EasyAnimate/README.md)**

The guide includes:
- Environment setup and requirements
- Data preparation and formatting
- Training configuration and execution
- Inference with trained UFO models
- Troubleshooting and optimization tips

### Key Training Commands

```bash
# Navigate to EasyAnimate directory
cd EasyAnimate

# Start UFO training
bash scripts/train_ufo.sh

# Run inference with trained UFO
python predict_t2v.py  # Text-to-video
python predict_i2v.py  # Image-to-video
```

## Key Features

✅ **Consistency Enhancement**: Significantly improves foreground-background consistency  
✅ **Quality Preservation**: Maintains original model capabilities while fixing artifacts  
✅ **Efficient Training**: Requires only 3000 steps on a single GPU  
✅ **Modular Design**: Multiple UFOs can be combined for customized effects  
✅ **Resource Friendly**: Minimal computational overhead during inference  

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{liu2025ufo,
  title={UFO: Enhancing Diffusion-Based Video Generation with a Uniform Frame Organizer},
  author={Liu, Delong and Hou, Zhaohui and Zhan, Mingjie and Han, Shihao and Zhao, Zhicheng and Su, Fei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={5388--5396},
  year={2025}
}
```

## Acknowledgements

This implementation is built upon [EasyAnimate](https://github.com/aigc-apps/EasyAnimate). We sincerely thank the EasyAnimate team for their excellent contribution to the open-source community.

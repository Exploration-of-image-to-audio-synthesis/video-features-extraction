
<div align="center">

# Video Feature Extraction Using ResNet and EfficientNetV2

<img src="./docs/_assets/base.png" width="300" />
</div>

This repository provides scripts to extract frame-wise features from videos using either [ResNet](https://arxiv.org/abs/1512.03385) or [EfficientNetV2](https://arxiv.org/abs/2104.00298) models, both pre-trained on the 1k ImageNet dataset. The features are extracted from the pre-classification layer.

The extraction scripts are based on the [torchvision models](https://pytorch.org/vision/0.19/models.html#classification). For each frame, the script outputs feature vectors, timestamps in ms, and the video’s fps. Frame-wise transformations and augmentations are also applied as part of the extraction process.

## Model-specific information
> [!IMPORTANT]
> Note that the output dimensions for these two models differ from each other: 
> * For [ResNet](https://arxiv.org/abs/1512.03385) models: `num_frames * 2048`
> * For [EfficientNetV2-S](https://arxiv.org/abs/2104.00298) models: `num_frames * 1280`


# Installation
This guide assumes you already have Python installed on your system.

The installation process is the same whether you will use the ResNet models or EfficientNetV2

## Set up the Environment for ResNet and EfficientNetV2
Setup `conda` environment. Requirements are in file `conda_env.yml`
```bash
# it will create a new conda environment called 'video_features' on your machine
conda env create -f conda_env.yml
```

> [!NOTE]
> `conda_env.yml` file uses PyTorch with CUDA version 12.1. If you're using a different version, or will use ROCm or CPU training, you might need to install them manually according to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

# Examples
> [!TIP] 
> If you experience a TypeError `'NoneType object is not subscriptable` when launching `main.py`, make sure that you're providing the list of `device_ids` as an argument when launching the script.

Start by activating the environment
```bash
conda activate video_features
```

The example is provided for the ResNet-50 flavour, but the following examples also work for ResNet-18,34,101,152 as well as EfficientNetV2-S.
```bash
python main.py \
    --feature_type resnet50 \
    --device_ids 0 2 \
    --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
If you would like to save the features, use `--on_extraction save_numpy` (or `save_pickle`) – by default, the features are saved in `./output/` or where `--output_path` specifies. In the case of frame-wise features, besides features, it also saves timestamps in ms and the original fps of the video into the same folder with features.
```bash
python main.py \
    --feature_type resnet50 \
    --device_ids 0 2 \
    --on_extraction save_numpy \
    --file_with_video_paths ./sample/sample_video_paths.txt
```
Since these features are so fine-grained and light-weight we may increase the extraction speed with batching. Therefore, frame-wise features have `--batch_size` argument, which defaults to `1`.
```bash
python main.py \
    --feature_type resnet50 \
    --device_ids 0 2 \
    --batch_size 128 \
    --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```
If you would like to extract features at a certain fps, add `--extraction_fps` argument
```bash
python main.py \
    --feature_type resnet50 \
    --device_ids 0 2 \
    --extraction_fps 5 \
    --video_paths ./sample/v_ZNVhz7ctTq0.mp4 ./sample/v_GGSY1Qvo990.mp4
```

# All arguments

The list of available arguments can be obtained by launching `python main.py --help`.

Some options are left from [the original repository](https://github.com/v-iashin/video_features/tree/specvqgan) and have no effect on the behavior of the script due to the removal of unused models.

# Credits
The code is based on work from [Video Features](https://github.com/v-iashin/video_features/tree/specvqgan) repository (branch `specvqgan`) by [Vladimir Iashin](https://github.com/v-iashin) and other contributors.

We added the ability to extract the features from [EfficientNetV2](https://arxiv.org/abs/2104.00298), and removed any models we didn't use, thus keeping only the [ResNet](https://arxiv.org/abs/1512.03385) models.

Additional credits:
1. The [TorchVision implementation](https://pytorch.org/vision/0.19/models.html#classification)
2. The [ResNet paper](https://arxiv.org/abs/1512.03385)
3. The [EfficientNetV2 paper](https://arxiv.org/abs/2104.00298)

# License

This repository is licensed under the terms of the GNU General Public License v3.0, as the base `specvqgan` branch from [the original repository](https://github.com/v-iashin/video_features/tree/specvqgan) has been licensed under GPL v3.

The model wrappers are licensed under the MIT License, as noted in their file headers. The use of these files complies with both the MIT and GPL v3 licenses.
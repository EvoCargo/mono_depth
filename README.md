# Monocular depth estimation 🔍

Hi! In this repo we will try to solve monocular depth estimation problem for our autonomous selfdriving car ([Check it out!](https://evocargo.com/eng/)).

All results would be introduced in our [paper](https://www.overleaf.com/read/hzvkhgckssjz) (view only).

Don't forget to set your environment with [this guide](HOW_TO_SET_ENV.md).

In this repo we try to implement next networks for our task:

1. lightweight:

    * FastDepth [[paper](https://arxiv.org/pdf/1903.03273.pdf)] [[code](https://github.com/dwofk/fast-depth)]

2. networks with unsupervised approach:

    * MonoDepth2 [[paper](https://arxiv.org/pdf/1806.01260.pdf)] [[code](https://github.com/nianticlabs/monodepth2)]

    * FeatDepth [[paper](https://arxiv.org/pdf/2007.10603v1.pdf)] [[code](https://github.com/sconlyshootery/FeatDepth)]

3. SOTA networks:

    * BTS [[paper](https://arxiv.org/pdf/1907.10326v5.pdf)] [[code](https://github.com/cogaplex-bts/bts)]

    * DORN [[paper](https://arxiv.org/pdf/1806.02446.pdf)] [[code](https://github.com/dontLoveBugs/SupervisedDepthPrediction)]

    * AdaBins [[paper](https://arxiv.org/pdf/2011.14141v1.pdf)] [[code](https://github.com/shariqfarooq123/AdaBins)]

# Datasets

There are 3 datasets available for our task:

* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

* [CityScapes](https://www.cityscapes-dataset.com/downloads/)

* Our own

# Train 🚂

You can train any network with one command:

```bash
python train.py --nn adabins --train_dir /path/to/dir --pretrained kitti --config config_file.txt
```
where

* `nn` is one of introduced networks;

* `train_dir` is directory with training and ground truth data;

* `pretrained` is flag to use one of pretrained models;

* `config` is file with specific configs for specific network.

Detailed information about each model is available in its folder in [src](./src).

All pretrained models are [here](https://drive.google.com/drive/folders/184grgoiV4IqAgJ_M0_Fkk9FB_v975RSq?usp=sharing).

Here is the table of pretrains which you can use in train script with `--pretrained` arg.

All this pretrains you can download via

```bash
python download_model.py --network monodepth2 --model mono_640x192
```

__PLEASE__ use this sript from root dir!!!

Network | Model | Description
--- | --- | ---
Adabins | [kitti](https://drive.google.com/u/0/uc?export=download&confirm=l0dw&id=15dE5uF7lG__lx8H8fXaZBymOC041QTEQ) | Model pretrained on KITTI data
BTS | [densenet121](https://drive.google.com/u/0/uc?export=download&confirm=83-1&id=1gYD3ZhfLTbxYon6NPaWRE7UsZJ7eKjG7) | Model with densenet121 backbone
BTS | [densenet161](https://drive.google.com/u/0/uc?export=download&confirm=BBd3&id=1rlT_L6K5FyL35pH9oogLYh8qNVnOc4Iq) | Model with densenet161 backbone
BTS | [resnet50](https://drive.google.com/u/0/uc?export=download&confirm=Q9hh&id=1QM3DOQCU0HmdFXSVEjbt3nQWa2-BAH9n) | Model with resnet50 backbone
BTS | [resnet101](https://drive.google.com/u/0/uc?export=download&confirm=EbdG&id=1dNC7AtGVgS627AxcXmm5B-UXY2wXqGRB) | Model with resnet101 backbone
BTS | [resnext50](https://drive.google.com/u/0/uc?export=download&confirm=WXub&id=1IR3sONAj3lbPajbor3hjOZ8hvlyvtWzt) | Model with resnext50 backbone
BTS | [resnext101](https://drive.google.com/u/0/uc?export=download&confirm=A1bl&id=1Lf-FcJwE-A51XtwcqAZs3ja4OG0pn6-n) | Model with resnext101 backbone
DORN | [resnet](https://drive.google.com/u/0/uc?export=download&confirm=7bQE&id=1pOHRZB6a0IJUE3cFzPWYrSMA0UgIfQmQ) | Model with pretrained Resnet
FastDepth | [mobilenet-nnconv5](https://drive.google.com/u/0/uc?export=download&confirm=uE58&id=1k3D5sr88LwMMRyfSfSAA2EyjOi57U5GT) | Model with pretrained MobileNet
FastDepth | [mobilenet-nnconv5-dw](https://drive.google.com/u/0/uc?id=12n25k8e5qF4l61Wgw5Fw788a4ROA4azy&export=download) | Model with pretrained MobileNet as encoder and smth else
FastDepth | [mobilenet-nnconv5-dw-sc](https://drive.google.com/u/0/uc?id=1dB6J6x_vrsDo4-M1fO5HxO8Z0sgUFcpN&export=download) | Model with pretrained MobileNet as encoder, smth else and skip-connections
FastDepth | [mobilenet-nnconv5-dw-sc-pn](https://drive.google.com/u/0/uc?id=1G2ZyS63FMwR9uX-criPC0IVDLYSfW6xK&export=download) | Model with pretrained MobileNet as encoder, smth else, skip-connections and pruned
FeatDepth | [autoencoder](https://drive.google.com/u/0/uc?export=download&confirm=i2Xd&id=1TZ-piXUlLfJhiN-OUC-sDoICUXlzXknn) | Autoencoder trained on the kitti raw data
FeatDepth | [depth_refine](https://drive.google.com/u/0/uc?export=download&confirm=t9pL&id=1vIh9NnwvgsnMyjHLsSbLhbvtgSIHalZz) | Model finetuned on test split of KITTI by online refinement
FeatDepth | [depth](https://drive.google.com/u/0/uc?export=download&confirm=ogKI&id=1EQdJAF6Ew64_kFGmKwKMP2r7wnKnJuWn) | Model trained on KITTI
MonoDepth2 | [mono_640x192](https://drive.google.com/u/0/uc?export=download&confirm=hlYX&id=1gVv4kb1_9H_boQAVTd3BzFmWxzbivS6P) | Imagenet pretrained model with resolution 640x192 and mono as source
MonoDepth2 | [mono_1024x320](https://drive.google.com/u/0/uc?export=download&confirm=62us&id=1_p7T4BKKSIbJ_92cV_9LzbXdgWCut1Ay) | Imagenet pretrained model with resolution 1024x320 and mono as source
MonoDepth2 | [mono_no_pt_640x192](https://drive.google.com/u/0/uc?export=download&confirm=8SMG&id=1ubu-AAoxr3wVmKS77wEGrB56Anb8mmxO) | Model with resolution 640x192 and mono as source without pretrained Imagenet
MonoDepth2 | [mono_resnet50_640x192](https://drive.google.com/u/0/uc?export=download&confirm=B8hW&id=1fwWnoHNhippOPKvAs0Wv3L1vzliJyYBj) | Model with resolution 640x192 and mono as source on resnet50 with Imagenet
MonoDepth2 | [mono_resnet50_no_pt_640x192](https://drive.google.com/u/0/uc?export=download&confirm=vMBg&id=1se52I8K5cyEuB_vXtMmGJFkwlTHYywRH) | Model with resolution 640x192 and mono and stereo as source on resnet50 without Imagenet

# Test 🏁

You can test model on your data. It can be one image (**PNG ONLY**) or directory (**STILL PNG**).

To test network on image use next syntax:

```bash
python test.py --model_path /path/to/model.h5 --img /path/to/img.png
```

To test network on dir of iamges use next syntax:

```bash
python test.py --model_path /path/to/model.h5 -dir /path/to/dir
```

# Perfomance ⚙️

Here we compare perfomance and consumptions of our networks

Network | Modification | Inference time | RAM consumption
--- | --- | --- | ---
Adabins | NaN | NaN | NaN
BTS | NaN | NaN | NaN
DORN | NaN | NaN | NaN | NaN
FastDepth | NaN | NaN | NaN
FeatDepth | NaN | NaN | NaN
Monodepth2 | NaN | NaN | NaN

# Results 📊

Here we compare results

Network | Modification | Accuracy | MAE
--- | --- | --- | ---
Adabins | NaN | NaN | NaN
BTS | NaN | NaN | NaN
DORN | NaN | NaN | NaN | NaN
FastDepth | NaN | NaN | NaN
FeatDepth | NaN | NaN | NaN
Monodepth2 | NaN | NaN | NaN

# TODO ❗❗❗

* __Adabins__: правильно установить pytorch3d + дообучить + eval

* __Monodepth__: eval

* __BTS__: evo датасет + дообучение + eval

* __DORN__: train + finetune + eval

* __Featdepth__:

* __Fastdepth__: train kitti/use pretrained nyu2 + finetune + eval

* __Global__: fix downloads + global scripts like `train.py`

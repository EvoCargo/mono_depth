# Monocular depth estimation 🔍

Hi! In this repo we will try to solve monocular depth estimation problem for our autonomous selfdriving car ([Check it out!](https://evocargo.com/eng/)).

All results are introduced in our [paper](https://drive.google.com/file/d/1DCW-Ywv0ISlMtBxqlFlMSEWG69QKtBMr/view?usp=sharing).

Don't forget to set your environment with [this guide](HOW_TO_SET_ENV.md).

In this repo we try to implement next networks for our task:

1. unsupervised approach:

    * MonoDepth2 [[paper](https://arxiv.org/pdf/1806.01260.pdf)] [[code](https://github.com/nianticlabs/monodepth2)]

3. supervised approach:

    * BTS [[paper](https://arxiv.org/pdf/1907.10326v5.pdf)] [[code](https://github.com/cogaplex-bts/bts)]

    * DORN [[paper](https://arxiv.org/pdf/1806.02446.pdf)] [[code](https://github.com/dontLoveBugs/SupervisedDepthPrediction)]

    * AdaBins [[paper](https://arxiv.org/pdf/2011.14141v1.pdf)] [[code](https://github.com/shariqfarooq123/AdaBins)]

# Datasets

There are 3 datasets available for our task:

* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

* [CityScapes](https://www.cityscapes-dataset.com/downloads/)

* Our own

# Train 🚂

Detailed information about train and other is available in [networks](./networks) folder.

All pretrained models are [here](https://drive.google.com/drive/folders/184grgoiV4IqAgJ_M0_Fkk9FB_v975RSq?usp=sharing). You can download it with

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
MonoDepth2 | [mono_640x192](https://drive.google.com/u/0/uc?export=download&confirm=hlYX&id=1gVv4kb1_9H_boQAVTd3BzFmWxzbivS6P) | Imagenet pretrained model with resolution 640x192 and mono as source
MonoDepth2 | [mono_1024x320](https://drive.google.com/u/0/uc?export=download&confirm=62us&id=1_p7T4BKKSIbJ_92cV_9LzbXdgWCut1Ay) | Imagenet pretrained model with resolution 1024x320 and mono as source
MonoDepth2 | [mono_no_pt_640x192](https://drive.google.com/u/0/uc?export=download&confirm=8SMG&id=1ubu-AAoxr3wVmKS77wEGrB56Anb8mmxO) | Model with resolution 640x192 and mono as source without pretrained Imagenet
MonoDepth2 | [mono_resnet50_640x192](https://drive.google.com/u/0/uc?export=download&confirm=B8hW&id=1fwWnoHNhippOPKvAs0Wv3L1vzliJyYBj) | Model with resolution 640x192 and mono as source on resnet50 with Imagenet
MonoDepth2 | [mono_resnet50_no_pt_640x192](https://drive.google.com/u/0/uc?export=download&confirm=vMBg&id=1se52I8K5cyEuB_vXtMmGJFkwlTHYywRH) | Model with resolution 640x192 and mono and stereo as source on resnet50 without Imagenet

# Results 📊

Here we compare results

Network | RMSE | RMSLE | AbsRel | SqRel | Acc < 1.25 | Acc < 1.25^2 | Acc < 1.25^3
--- | --- | --- | --- | --- | --- | --- | ---
Adabins | 7.018 | 0.175 | 0.099 | 1.054 | 0.883 | 0.9649 | 0.985
BTS | 6.296 | 0.1675 | 0.115 | 0.9324 | 0.858 | 0.969 | 0.991
DORN | 8.912 | 0.8417 | 1.401 | 1.854 | 0.772 | 0.905 | 0.933
Monodepth2 | 5.465 | 0.158 | 0.134 | 0.660 | 0.877 | 0.997 | 0.996

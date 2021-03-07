# Monocular depth estimation

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

# Train

All pretrained models are [here](https://drive.google.com/drive/folders/184grgoiV4IqAgJ_M0_Fkk9FB_v975RSq?usp=sharing).

Here is the table of pretrains which you can use in train script with `--pretrained` arg.

Network | Model | Description
--- | --- | ---
Adabins | [model](https://drive.google.com/u/0/uc?export=download&confirm=bS7j&id=15NEu5bjOn2ABseVpHVgOpDRpjhsb626m) | Model pretrained on KITTI data
FastDepth | [imagenet](https://drive.google.com/u/0/uc?id=1aqHzLwSLDqCIDtAvqEvOStivv2oX5Tgm&export=download) | Best model introduced by authors
FastDepth | [mobilenet-nnconv5](https://drive.google.com/u/0/uc?export=download&confirm=uE58&id=1k3D5sr88LwMMRyfSfSAA2EyjOi57U5GT) | Model with pretrained MobileNet
FastDepth | [mobilenet-nnconv5-dw](https://drive.google.com/u/0/uc?id=12n25k8e5qF4l61Wgw5Fw788a4ROA4azy&export=download) | Model with pretrained MobileNet as encoder and smth else
FastDepth | [mobilenet-nnconv5-dw-sc](https://drive.google.com/u/0/uc?id=1dB6J6x_vrsDo4-M1fO5HxO8Z0sgUFcpN&export=download) | Model with pretrained MobileNet as encoder, smth else and skip-connections
FastDepth | [mobilenet-nnconv5-dw-sc-pn](https://drive.google.com/u/0/uc?id=1G2ZyS63FMwR9uX-criPC0IVDLYSfW6xK&export=download) | Model with pretrained MobileNet as encoder, smth else, skip-connections and pruned
DORN | [resnet](https://drive.google.com/u/0/uc?export=download&confirm=7bQE&id=1pOHRZB6a0IJUE3cFzPWYrSMA0UgIfQmQ) | Model with pretrained Resnet
FeatDepth | [autoencoder](https://drive.google.com/u/0/uc?export=download&confirm=i2Xd&id=1TZ-piXUlLfJhiN-OUC-sDoICUXlzXknn) | Unknown
FeatDepth | [depth_odom](https://drive.google.com/u/0/uc?export=download&confirm=KRD3&id=1rsZ7SgjNEmwEXufKh8PAlooZ5gNTEKsX) | Unknown
FeatDepth | [depth_refine](https://drive.google.com/u/0/uc?export=download&confirm=t9pL&id=1vIh9NnwvgsnMyjHLsSbLhbvtgSIHalZz) | Unknown
FeatDepth | [depth](https://drive.google.com/u/0/uc?export=download&confirm=ogKI&id=1EQdJAF6Ew64_kFGmKwKMP2r7wnKnJuWn) | Unknown
BTS | [densenet121](https://drive.google.com/u/0/uc?export=download&confirm=83-1&id=1gYD3ZhfLTbxYon6NPaWRE7UsZJ7eKjG7) | U know
BTS | [densenet161](https://drive.google.com/u/0/uc?export=download&confirm=BBd3&id=1rlT_L6K5FyL35pH9oogLYh8qNVnOc4Iq) | U know
BTS | [resnet50](https://drive.google.com/u/0/uc?export=download&confirm=Q9hh&id=1QM3DOQCU0HmdFXSVEjbt3nQWa2-BAH9n) | U know
BTS | [resnet101](https://drive.google.com/u/0/uc?export=download&confirm=EbdG&id=1dNC7AtGVgS627AxcXmm5B-UXY2wXqGRB) | U know
BTS | [resnext50](https://drive.google.com/u/0/uc?export=download&confirm=WXub&id=1IR3sONAj3lbPajbor3hjOZ8hvlyvtWzt) | U know
BTS | [resnext101](https://drive.google.com/u/0/uc?export=download&confirm=A1bl&id=1Lf-FcJwE-A51XtwcqAZs3ja4OG0pn6-n) | U know
MonoDepth2 | [mono_640x192](https://drive.google.com/u/0/uc?export=download&confirm=hlYX&id=1gVv4kb1_9H_boQAVTd3BzFmWxzbivS6P) | U know
MonoDepth2 | [stereo_640x192](https://drive.google.com/u/0/uc?export=download&confirm=neJi&id=1-aWu7lKQRNnygr3vAta8-vZx_ahYExlI) | U know
MonoDepth2 | [mono+stereo_640x192](https://drive.google.com/u/0/uc?export=download&confirm=MTKo&id=1DziaSK4oT01D2ug038JvfkJIUIOLcbt8) | U know
MonoDepth2 | [mono_1024x320](https://drive.google.com/u/0/uc?export=download&confirm=62us&id=1_p7T4BKKSIbJ_92cV_9LzbXdgWCut1Ay) | U know
MonoDepth2 | [stereo_1024x320](https://drive.google.com/u/0/uc?export=download&confirm=G-Oy&id=1z4q4xo1sI2Qyukxbwv8E_hYeWvarNfQ8) | U know
MonoDepth2 | [mono+stereo_1024x320](https://drive.google.com/u/0/uc?export=download&confirm=8nq8&id=1KmtNclGufmFq-XoKqL3dy2Uppwcfkj4e) | U know
MonoDepth2 | [mono_no_pt_640x192](https://drive.google.com/u/0/uc?export=download&confirm=8SMG&id=1ubu-AAoxr3wVmKS77wEGrB56Anb8mmxO) | U know
MonoDepth2 | [stereo_no_pt_640x192](https://drive.google.com/u/0/uc?export=download&confirm=Pmut&id=1tDpF5qVgWFdOkbeWRDTNCZx3wCCPEec_) | U know
MonoDepth2 | [mono+stereo_no_pt_640x192](https://drive.google.com/u/0/uc?export=download&confirm=vn7L&id=1v9wBGVKvm75LSmrys3vmmiSeHmU1xC4o) | U know
MonoDepth2 | [mono_odom_640x192](https://drive.google.com/u/0/uc?export=download&confirm=2iWf&id=16TxTfVc7E90rQqrSWaB53Fa-U58arKLT) | U know
MonoDepth2 | [mono+stereo_odom_640x192](https://drive.google.com/u/0/uc?export=download&confirm=s2NN&id=1RzwNhlecp7nPx_ul992GRhLw58ammRkg) | U know
MonoDepth2 | [mono_resnet50_640x192](https://drive.google.com/u/0/uc?export=download&confirm=B8hW&id=1fwWnoHNhippOPKvAs0Wv3L1vzliJyYBj) | U know
MonoDepth2 | [mono_resnet50_no_pt_640x192](https://drive.google.com/u/0/uc?export=download&confirm=vMBg&id=1se52I8K5cyEuB_vXtMmGJFkwlTHYywRH) | U know




You can train any network with one command:

```bash
python train.py --nn adabins --pretrained imagenet --train_dir /path/to/dir
```


# Evaluate

```bash
python eval.py
```

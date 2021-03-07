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

Here is the table of pretrains which you can use

Network | Model | Description
--- | --- | ---
Adabins | [model](https://drive.google.com/u/0/uc?export=download&confirm=p99H&id=15NEu5bjOn2ABseVpHVgOpDRpjhsb626m) | Model pretrained on KITTI data
FastDepth | [imagenet](https://drive.google.com/u/0/uc?id=1aqHzLwSLDqCIDtAvqEvOStivv2oX5Tgm&export=download) | Best model introduced by authors
FastDepth | [mobilenet-nnconv5](https://drive.google.com/u/0/uc?export=download&confirm=8Ba_&id=1k3D5sr88LwMMRyfSfSAA2EyjOi57U5GT) | Model with pretrained MobileNet
FastDepth | [mobilenet-nnconv5-dw](https://drive.google.com/u/0/uc?id=12n25k8e5qF4l61Wgw5Fw788a4ROA4azy&export=download) | Model with pretrained MobileNet as encoder and smth else
FastDepth | [mobilenet-nnconv5-dw-sc](https://drive.google.com/u/0/uc?id=1dB6J6x_vrsDo4-M1fO5HxO8Z0sgUFcpN&export=download) | Model with pretrained MobileNet as encoder, smth else and skip-connections
FastDepth | [mobilenet-nnconv5-dw-sc-pn](https://drive.google.com/u/0/uc?id=1G2ZyS63FMwR9uX-criPC0IVDLYSfW6xK&export=download) | Model with pretrained MobileNet as encoder, smth else, skip-connections and pruned
DORN | [resnet](https://drive.google.com/u/0/uc?export=download&confirm=w7B6&id=1pOHRZB6a0IJUE3cFzPWYrSMA0UgIfQmQ) | Model with pretrained Resnet
FeatDepth | [autoencoder](https://drive.google.com/u/0/uc?export=download&confirm=52fx&id=1TZ-piXUlLfJhiN-OUC-sDoICUXlzXknn) | Unknown
FeatDepth | [depth_odom](https://drive.google.com/u/0/uc?export=download&confirm=Xjg0&id=1rsZ7SgjNEmwEXufKh8PAlooZ5gNTEKsX) | Unknown
FeatDepth | [depth_refine](https://drive.google.com/u/0/uc?export=download&confirm=Mci_&id=1vIh9NnwvgsnMyjHLsSbLhbvtgSIHalZz) | Unknown
FeatDepth | [depth](https://drive.google.com/u/0/uc?export=download&confirm=IthB&id=1EQdJAF6Ew64_kFGmKwKMP2r7wnKnJuWn) | Unknown
BTS | [densenet121](https://drive.google.com/u/0/uc?export=download&confirm=cpAB&id=1gYD3ZhfLTbxYon6NPaWRE7UsZJ7eKjG7) | U know
BTS | []() | U know
BTS | []() | U know
BTS | []() | U know
BTS | []() | U know
BTS | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know
MonoDepth2 | []() | U know




You can train any network with one command:

'''bash
python train.py --nn adabins --pretrained imagenet --train_dir /path/to/dir
'''

Use 'pretrained' arg for it


# Evaluate

'''bash
python eval.py
'''

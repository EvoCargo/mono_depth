# Monocular depth estimation

Hi! In this repo we will try to solve monocular depth estimation problem for our autonomous selfdriving car ([Check it out!](https://evocargo.com/eng/)).

All results would be introduced in our [paper](https://www.overleaf.com/read/hzvkhgckssjz) (view only).

Don't forget to set your environment with [this guide](HOW_TO_SET_ENV.md).

In this repo we try to implement next networks for our task:

1. lightweight:

    * FastDepth [paper](https://arxiv.org/pdf/1903.03273.pdf) [code](https://github.com/dwofk/fast-depth)

2. networks with unsupervised approach:

    * MonoDepth2 [paper](https://arxiv.org/pdf/1806.01260.pdf) [code](https://github.com/nianticlabs/monodepth2)

    * FeatDepth [paper](https://arxiv.org/pdf/2007.10603v1.pdf) [code](https://github.com/sconlyshootery/FeatDepth)

3. networks with specific layers:

    * BTS [paper](https://arxiv.org/pdf/1907.10326v5.pdf) [code](https://github.com/cogaplex-bts/bts)

    * DORN [paper](https://arxiv.org/pdf/1806.02446.pdf) [code](https://github.com/dontLoveBugs/SupervisedDepthPrediction)

4. SOTA networks:

    * AdaBins [paper](https://arxiv.org/pdf/2011.14141v1.pdf) [code](https://github.com/shariqfarooq123/AdaBins)

    * ViP-DeepLab [paper](https://arxiv.org/pdf/2012.05258.pdf) [code](https://github.com/joe-siyuan-qiao/ViP-DeepLab)

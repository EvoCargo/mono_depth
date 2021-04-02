# FeatDepth

## Setup

### Requirements:
- PyTorch1.1+, Python3.5+, Cuda10.0+
- mmcv==0.4.4

Our codes are based on mmcv for distributed learning.

## KITTI training data

Our training data is the same with other self-supervised monocular depth estimation methods, please refer to [monodepth2](https://github.com/nianticlabs/monodepth2) to prepare the training data.

## API
We provide an API interface for you to predict depth and pose from an image sequence and visulize some results.
They are stored in folder 'scripts'.
```
draw_odometry.py is used to provide several analytical curves and obtain standard kitti odometry evaluation results.
```

```
eval_pose.py is used to obtain kitti odometry evaluation results.
```

```
eval_depth.py is used to obtain kitti depth evaluation results.
```

```
infer.py is used to generate depth maps from given models.
```

```
infer_singleimage.py is used to test a single image for view.
```
## Training
You can use following command to launch distributed learning of our model:
```shell
/path/to/python -m torch.distributed.launch --master_port=9900 --nproc_per_node=1 train.py --config /path/to/cfg_kitti_fm.py --work_dir /dir/for/saving/weights/and/logs'
```
Here nproc_per_node refers to GPU number you want to use.

## Configurations
We provide a variety of config files for training on different datasets.
They are stored in config folder.

For example:
(1) 'cfg_kitti_fm.py' is used to train our model on kitti dataset, where the weights of autoencoder are loaded from the pretrained weights we provide and fixed during the traing.
This mode is prefered when your GPU memory is lower than 16 GB;
(2) 'cfg_kitti_fm_joint.py' is used to train our model on kitti dataset, where the autoencoder is jointly trained with depthnet and posenet.
We rescale the input resolution of our model to ensure training with 12 GB GPU memory, slightly reducing the performance.
You can modify the input resolution according to your computational resource.

For modifying config files, please refer to cfg_kitti_fm.py.

## Online refinement
We provide cfg file for online refinement, you can use cfg_kitti_fm_refine.py to refine your model trained on kitti raw data by keeping training on test data.
For settings of online refinement, please refer to details in cfg_kitti_fm_refine.py in the folder config.

## Finetuning
If you want to finetune on a given weights, you can modify the 'resume_from' term from 'None' to an existing path to a pre-trained weight in the config files.

## Notes
Our model predicts inverse depths.
If you want to get real depth when training stereo model, you have to convert inverse depth to depth, and then multiply it by 36.

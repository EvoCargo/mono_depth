# BTS

## Testing with [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
First make sure that you have prepared the test set using instructions in README.md at root of this repo.
```shell
$ cd ~/workspace/bts/pytorch
$ mkdir models
### Get BTS model trained with NYU Depth V2
$ cd models
$ wget https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_densenet161.zip
$ unzip bts_nyu_v2_pytorch_densenet161.zip
```
Once the preparation steps completed, you can test BTS using following commands.
```
$ cd ~/workspace/bts/pytorch
$ python bts_test.py arguments_test_nyu.txt
```
This will save results to ./result_bts_nyu_v2_pytorch_densenet161. With a single RTX 2080 Ti it takes about 41 seconds for processing 654 testing images.

## Evaluation
Following command will evaluate the prediction results for NYU Depvh V2.
```
$ cd ~/workspace/bts/pytorch
$ python ../utils/eval_with_pngs.py --pred_path result_bts_nyu_v2_pytorch_densenet161/raw/ --gt_path ../../dataset/nyu_depth_v2/official_splits/test/ --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
```

You should see outputs like this:
```
Raw png files reading done
Evaluating 654 files
GT files reading done
0 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.885,   0.978,   0.994,   0.110,   0.066,   0.392,   0.142,  11.533,   0.047
Done.
```

## Preparation for Training

### KITTI
You can also train BTS with KITTI dataset by following procedures.
First, make sure that you have prepared the ground truth depthmaps from [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip).
If you have not, please follow instructions on README.md at root of this repo.
Then, download and unzip the raw dataset using following commands.
```
$ cd ~/workspace/dataset/kitti_dataset
$ aria2c -x 16 -i ../../bts/utils/kitti_archives_to_download.txt
$ parallel unzip ::: *.zip
```
Finally, we can train our network with
```
$ cd ~/workspace/bts/pytorch
$ python bts_main.py arguments_train_eigen.txt
```

## Testing and Evaluation with [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
Once you have KITTI dataset and official ground truth depthmaps, you can test and evaluate our model with following commands.
```
### Get model trained with KITTI Eigen split
$ cd ~/workspace/bts/pytorch/models
$ wget https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_densenet161.zip
$ cd unzip bts_eigen_v2_pytorch_densenet161.zip
```
Test and save results.
```
$ cd ~/workspace/bts/pytorch
$ python bts_test.py arguments_test_eigen.txt
```
This will save results to ./result_bts_eigen_v2_pytorch_densenet161.
Finally, we can evaluate the prediction results with
```
$ cd ~/workspace/bts/pytorch
$ python ../utils/eval_with_pngs.py --pred_path result_bts_eigen_v2_pytorch_densenet161/raw/ --gt_path ../../dataset/kitti_dataset/data_depth_annotated/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --do_kb_crop --garg_crop
```
You should see outputs like this:
```
GT files reading done
45 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.955,   0.993,   0.998,   0.060,   0.249,   2.798,   0.096,   8.933,   0.027
Done.
```

Also you can use various base networks with pretrained weights as the encoder for bts.\
Available options are: `resnet50_bts`, `resnet101_bts`, `resnext50_bts`, `resnext101_bts`, `densenet121_bts` and `densenet161_bts`\
Simply change the argument `--encoder` in arguments_train_*.txt with your choice.

## Model Zoo
### KITTI Eigen Split

| Base Network |   d1  |   d2  |   d3  | AbsRel | SqRel |  RMSE | RMSElog | SILog | log10 | #Params |          Model Download          |
|:------------:|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|:-------:|:-----:|:-----:|:-------:|:--------------------------------:|
| ResNet50     | 0.954 | 0.992 | 0.998 |  0.061 | 0.250 | 2.803 |   0.098 | 9.030 | 0.027 |   49.5M | [bts_eigen_v2_pytorch_resnet50](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_resnet50.zip)  |
| ResNet101    | 0.954 | 0.992 | 0.998 |  0.061 | 0.261 | 2.834 |   0.099 | 9.075 | 0.027 |   68.5M | [bts_eigen_v2_pytorch_resnet101](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_resnet101.zip) |
| ResNext50    | 0.954 | 0.993 | 0.998 |  0.061 | 0.245 | 2.774 |   0.098 | 9.014 | 0.027 |   49.0M | [bts_eigen_v2_pytorch_resnext50](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_resnext50.zip)  |
| ResNext101   | 0.956 | 0.993 | 0.998 |  0.059 | 0.241 | 2.756 |   0.096 | 8.781 | 0.026 |  112.8M | [bts_eigen_v2_pytorch_resnext101](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_resnext101.zip)  |
| DenseNet121  | 0.951 | 0.993 | 0.998 |  0.063 | 0.256 | 2.850 |   0.100 | 9.221 | 0.028 |   21.2M | [bts_eigen_v2_pytorch_densenet121](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_densenet121.zip) |
| DenseNet161  | 0.955 | 0.993 | 0.998 |  0.060 | 0.249 | 2.798 |   0.096 | 8.933 | 0.027 |   47.0M | [bts_eigen_v2_pytorch_densenet161](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_densenet161.zip) |


Note: Modify arguments `--encoder`, `--model_name`, `--checkpoint_path` and `--pred_path` accordingly.


python ./src/networks/bts/model/bts_test.py --encoder densenet121 --dataset kitti --data_path autosave.png --checkpoint_path ./src/networks/bts/pretrained/densenet121/densenet121/bts_eigen_v2_pytorch_densenet121/model --filenames_file ./src/networks/bts/train_test_inputs/eigen_test_files_with_gt.txt

# Monodepth2

## How to

We trained our network with such arguments:

```bash
python train.py --model_name evo_scratch --height 288 --width 512 --data_path /media/data/datasets/bag_depth --split evo --dataset evo --num_epochs 30 --batch_size 4 --num_layers 50
```

For more info about training and evaluation arguments check [options](./options.py) file.

To evaluate evo:

```bash
python evaluate_depth.py --load_weights_folder result/... --eval_mono --eval_split evo --data_path /media/data/datasets --save_pred_disps --eval_from_file
```

## Inference

You can predict depth for a single image with:
```bash
python test_simple.py --image assets/test_image.jpg --model pretrained/mono_640x192
```

But we used [notebook](./depth_prediction_example.ipynb) for it.

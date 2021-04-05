# AdaBins

## Pretrained

Download pretrained model with `download_model.py` in root

```
python download_model.py --network adabins --model kitti
```

## Inference

Инференс заводится через

```bash
python infer.py --model ./pretrained/kitti/kitti.pth --image /path/to/image
```

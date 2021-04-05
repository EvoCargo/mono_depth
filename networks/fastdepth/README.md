# FastDepth

## Requirements
- Install [PyTorch](https://pytorch.org/) on a machine with a CUDA GPU. Our code was developed on a system running PyTorch v0.4.1.
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format libraries. Files in our pre-processed datasets are in HDF5 format.
  ```bash
  sudo apt-get update
  sudo apt-get install -y libhdf5-serial-dev hdf5-tools
  pip3 install h5py matplotlib imageio scikit-image opencv-python
  ```

## Train

Parameters

## Inference

Инференс заводится через

```bash
python test.py --image /path/to/image --model ./pretrained/mobilenet-nnconv5/mobilenet-nnconv5.pth
```

## Evaluation

This step requires a valid PyTorch installation and a saved copy of the NYU Depth v2 dataset. It is meant to be performed on a host machine with a CUDA GPU, not on an embedded platform. Deployment on an embedded device is discussed in the [next section](#deployment).

To evaluate a model, navigate to the repo directory and run:

```bash
python3 main.py --evaluate [path_to_trained_model]
```

The evaluation code will report model accuracy in terms of the delta1 metric as well as RMSE in millimeters.

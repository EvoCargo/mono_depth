import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--image', '-i', required=True, help='image path (.png or .jpg)')
    parser.add_argument('--model', '-m', required=True, help='model path (.pth)')

    args = parser.parse_args()
    return args


def test(image, model, **args):

    mo = torch.load(model)

    im = cv2.imread(image)
    tr_im = cv2.resize(im, (224 * 5, 224 * 3))

    real_mo = mo['model']

    ten_im = torch.unsqueeze(torch.from_numpy(tr_im / 255.0), 0).permute((0, 3, 1, 2))
    ten_im = ten_im.cuda().float()

    with torch.no_grad():
        cpu_pred = torch.clip(real_mo(ten_im), 0, 255).cpu()

    res_im = np.squeeze(cpu_pred.numpy(), 0).transpose((1, 2, 0))
    plt.imsave(
        '.'.join(image.split('.')[:-1]) + '_di.jpg',
        (res_im * 255.0 / res_im.max()).squeeze(),
    )


if __name__ == '__main__':
    test(**vars(parse_args()))

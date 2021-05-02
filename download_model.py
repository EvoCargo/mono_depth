import argparse
import os
import zipfile

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--network', '-n', required=True, help='Network to download')
    parser.add_argument('--model', '-m', required=True, help='Model to download')

    args = parser.parse_args()
    return args


net2model = {
    'adabins': {'kitti': '15dE5uF7lG__lx8H8fXaZBymOC041QTEQ'},
    'bts': {
        'densenet121': '1gYD3ZhfLTbxYon6NPaWRE7UsZJ7eKjG7',
        'densenet161': '1rlT_L6K5FyL35pH9oogLYh8qNVnOc4Iq',
        'resnet50': '1QM3DOQCU0HmdFXSVEjbt3nQWa2-BAH9n',
        'resnet101': '1dNC7AtGVgS627AxcXmm5B-UXY2wXqGRB',
        'resnext50': '1IR3sONAj3lbPajbor3hjOZ8hvlyvtWzt',
        'resnext101': '1Lf-FcJwE-A51XtwcqAZs3ja4OG0pn6-n',
    },
    'dorn': {'resnet': '1pOHRZB6a0IJUE3cFzPWYrSMA0UgIfQmQ'},
    'fastdepth': {
        'mobilenet-nnconv5': '1k3D5sr88LwMMRyfSfSAA2EyjOi57U5GT',
        'mobilenet-nnconv5-dw': '12n25k8e5qF4l61Wgw5Fw788a4ROA4azy',
        'mobilenet-nnconv5-dw-sc': '1dB6J6x_vrsDo4-M1fO5HxO8Z0sgUFcpN',
        'mobilenet-nnconv5-dw-sc-pn': '1G2ZyS63FMwR9uX-criPC0IVDLYSfW6xK',
    },
    'featdepth': {
        'autoencoder': '1TZ-piXUlLfJhiN-OUC-sDoICUXlzXknn',
        'depth_odom': '1rsZ7SgjNEmwEXufKh8PAlooZ5gNTEKsX',
        'depth_refine': '1vIh9NnwvgsnMyjHLsSbLhbvtgSIHalZz',
        'depth': '1EQdJAF6Ew64_kFGmKwKMP2r7wnKnJuWn',
    },
    'monodepth2': {
        'mono_640x192': '1gVv4kb1_9H_boQAVTd3BzFmWxzbivS6P',
        'stereo_640x192': '1-aWu7lKQRNnygr3vAta8-vZx_ahYExlI',
        'mono+stereo_640x192': '1DziaSK4oT01D2ug038JvfkJIUIOLcbt8',
        'mono_1024x320': '1_p7T4BKKSIbJ_92cV_9LzbXdgWCut1Ay',
        'stereo_1024x320': '1z4q4xo1sI2Qyukxbwv8E_hYeWvarNfQ8',
        'mono+stereo_1024x320': '1KmtNclGufmFq-XoKqL3dy2Uppwcfkj4e',
        'mono_no_pt_640x192': '1ubu-AAoxr3wVmKS77wEGrB56Anb8mmxO',
        'stereo_no_pt_640x192': '1tDpF5qVgWFdOkbeWRDTNCZx3wCCPEec_',
        'mono+stereo_no_pt_640x192': '1v9wBGVKvm75LSmrys3vmmiSeHmU1xC4o',
        'mono_odom_640x192': '16TxTfVc7E90rQqrSWaB53Fa-U58arKLT',
        'mono+stereo_odom_640x192': '1RzwNhlecp7nPx_ul992GRhLw58ammRkg',
        'mono_resnet50_640x192': '1fwWnoHNhippOPKvAs0Wv3L1vzliJyYBj',
        'mono_resnet50_no_pt_640x192': '1se52I8K5cyEuB_vXtMmGJFkwlTHYywRH',
    },
}


def main(**args):

    if args['network'] not in net2model:
        print('Network not found')
        raise Exception

    if args['model'] not in net2model[args['network']]:
        print('Model not found')
        raise Exception

    file_id = net2model[args['network']][args['model']]
    if not os.path.exists('networks/{}/pretrained'.format(args['network'])):
        os.mkdir('networks/{}/pretrained'.format(args['network']))
    os.mkdir('networks/{}/pretrained/{}'.format(args['network'], args['model']))

    file_id = net2model[args['network']][args['model']]

    destination = 'networks/{}/pretrained/{}/{}.'.format(
        args['network'], args['model'], args['model']
    )

    if args['network'] not in ['bts', 'monodepth2']:
        destination += 'pth'
        download_file_from_google_drive(file_id, destination)
    else:
        destination += 'zip'
        download_file_from_google_drive(file_id, destination)
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(
                './networks/{}/pretrained/{}/.'.format(args['network'], args['model'])
            )
        os.remove(destination)


if __name__ == '__main__':
    main(**vars(parse_args()))

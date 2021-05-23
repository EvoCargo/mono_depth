DEPTH_LAYERS = 50  # resnet50
POSE_LAYERS = 18  # resnet18
FRAME_IDS = [0, -1, 1]
IMGS_PER_GPU = 3  # the number of images fed to each GPU
HEIGHT = 288  # input image height
WIDTH = 512  # input image width

data = dict(
    name='evo',  # dataset name
    split='evo',  # training split name
    height=HEIGHT,
    width=WIDTH,
    frame_ids=FRAME_IDS,
    in_path='/media/data/datasets/bag_depth',  # path to raw data
    # gt_depth_path='/media/sconly/harddisk/data/kitti/kitti_raw/rawdata/gt_depths.npz',  # path to gt data
    png=True,  # image format
)

model = dict(
    name='mono_fm',  # select a model by name
    depth_num_layers=DEPTH_LAYERS,
    pose_num_layers=POSE_LAYERS,
    frame_ids=FRAME_IDS,
    imgs_per_gpu=IMGS_PER_GPU,
    height=HEIGHT,
    width=WIDTH,
    scales=[0, 1, 2, 3],  # output different scales of depth maps
    min_depth=2.0,  # minimum of predicted depth value
    max_depth=117.0,  # maximum of predicted depth value
    depth_pretrained_path='/home/penitto/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth',
    pose_pretrained_path='/home/penitto/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth',
    extractor_pretrained_path='/media/data/datasets/penitto/networks/featdepth/autoencoder/20210522_233542_512x288/epoch_30.pth',  # pretrained weights for autoencoder
    automask=True,
    disp_norm=True,
    perception_weight=1e-3,
    smoothness_weight=1e-3,
)

resume_from = None
finetune = None
total_epochs = 40
imgs_per_gpu = IMGS_PER_GPU
learning_rate = 1e-4
workers_per_gpu = IMGS_PER_GPU
validate = True

optimizer = dict(type='Adam', lr=learning_rate, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20, 30],
    gamma=0.5,
)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
workflow = [('train', 1)]

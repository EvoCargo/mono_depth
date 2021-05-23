DEPTH_LAYERS = 50
POSE_LAYERS = 18
FRAME_IDS = [0]
IMGS_PER_GPU = 10
HEIGHT = 288
WIDTH = 512

data = dict(
    name='evo',
    split='evo',
    height=HEIGHT,
    width=WIDTH,
    frame_ids=FRAME_IDS,
    in_path='/media/data/datasets/bag_depth',
    # gt_depth_path = '/node01_data5/monodepth2-test/monodepth2/gt_depths.npz',
    png=False,
    stereo_scale=False,
)

model = dict(
    name='autoencoder',
    depth_num_layers=DEPTH_LAYERS,
    pose_num_layers=POSE_LAYERS,
    frame_ids=FRAME_IDS,
    imgs_per_gpu=IMGS_PER_GPU,
    height=HEIGHT,
    width=WIDTH,
    scales=[0, 1, 2, 3],
    min_depth=2.0,
    max_depth=117.0,
    depth_pretrained_path='/home/penitto/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth',
    pose_pretrained_path='/home/penitto/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth',
    automask=True,
    disp_norm=True,
    use_min_construct=True,
    dis=0.001,
    cvt=0.001,
)

# resume_from = '/node01_data5/monodepth2-test/model/ms/ms.pth'
resume_from = None
finetune = None
total_epochs = 30
imgs_per_gpu = IMGS_PER_GPU
learning_rate = 1e-4
workers_per_gpu = 4
validate = False

optimizer = dict(type='Adam', lr=learning_rate, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[10, 20],
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
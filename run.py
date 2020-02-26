# LICENSE file in the root directory of this source tree.
# All rights reserved.
#
# This source code is licensed under the license found in the
#

"""
import
"""
import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.m import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Defines
"""

"""
Parse Data
"""
args = parse_args()
print(args)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

"""
bm 
"""
torch.autograd.set_detect_anomaly(True)
DEBUG = False
UNUSED = False
NEW_LOSS = args.bm

SCALARS_SET = False
k_pmpjpe = 0

# velocity error
#VEL_LOSS = NEW_LOSS # cuda
VEL_LOSS = UNUSED # cuda

# acceleration error
#ACCEL_LOSS = NEW_LOSS # cuda
ACCEL_LOSS = UNUSED # cuda

# joint limits, symmetry error, angular..
#MJT_LOSS = UNUSED # cuda
MJT_LOSS = NEW_LOSS # cuda

# scaling and posing fixing error Protocol 2
#PMPJPE_LOSS = UNUSED # cuda
PMPJPE_LOSS = NEW_LOSS # cuda

# bone ratio error
MRPJ_LOSS = UNUSED # UNUSED

"""
data
"""
print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
else:
    raise KeyError('Invalid dataset')

#
print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        
        positions_3d = []
        for cam in anim['cameras']:
            pos_3d = world_to_camera(anim['positions'], 
                                     R=cam['orientation'], 
                                     t=cam['translation'])
            """            
            Remove global offset, but keep trajectory in first 
            position
            """
            pos_3d[:, 1:] -= pos_3d[:, :1] 
            positions_3d.append(pos_3d)
        anim['positions_3d'] = positions_3d

#
print('Loading 2D detections...')
keypoints = np.load('data/data_2d_' + 
                    args.dataset + 
                    '_' + 
                    args.keypoints + 
                    '.npz')
keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), \
                           list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

#print("joints", joints_left, joints_right)
keypoints = keypoints['positions_2d'].item()

#
for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        for cam_idx in range(len(keypoints[subject][action])):
            """
            We check for >= instead of == because some videos in 
            H3.6M contain extra frames
            """
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

# 
for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

#
subjects_train = args.subjects_train.split(',')
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')
subjects_test = args.subjects_test.split(',')

# semi supervised
semi_supervised = len(subjects_semi) > 0
if semi_supervised and not dataset.supports_semi_supervised():
    raise RuntimeError('Semi-supervised training is not implemented for this dataset')

"""
functions
"""
            
def fetch(subjects, 
          action_filter=None, 
          subset=1, 
          parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
                
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                # Iterate across cameras
                for i in range(len(poses_3d)): 
                    out_poses_3d.append(poses_3d[i])
    
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    
    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('=== Selected actions:', action_filter, "===")
    print("")
    
cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

filter_widths = [int(x) for x in args.architecture.split(',')]


def weights_init_uniform_rule(m):
        print("init_uniform")
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)
"""
cli mode options
"""

if not args.disable_optimizations and not args.dense and args.stride == 1:
    # Use optimized model for single-frame predictions
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], 
                                     poses_valid_2d[0].shape[-1], 
                                     poses_valid[0].shape[-2],
                                     filter_widths=filter_widths, 
                                     causal=args.causal, 
                                     dropout=args.dropout, 
                                     channels=args.channels)

else:
    """
    When incompatible settings are detected (stride > 1, dense 
    filters, or disabled optimization) fall back to normal model
    """
    model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], 
                                    poses_valid_2d[0].shape[-1], 
                                    poses_valid[0].shape[-2],
                                    filter_widths=filter_widths, 
                                    causal=args.causal, 
                                    dropout=args.dropout, 
                                    channels=args.channels,
                                    dense=args.dense)
if(MJT_LOSS):
  model_pos_train.apply(weights_init_uniform_rule)
   
# inference? 
model_pos = TemporalModel(poses_valid_2d[0].shape[-2], 
                          poses_valid_2d[0].shape[-1], 
                          poses_valid[0].shape[-2],
                          filter_widths=filter_widths, 
                          causal=args.causal, 
                          dropout=args.dropout, 
                          channels=args.channels,
                          dense=args.dense)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()
    
if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])
    
test_generator = UnchunkedGenerator(cameras_valid, 
                                    poses_valid, poses_valid_2d,
                                    pad=pad, 
                                    causal_shift=causal_shift,  
                                    augment=False,
                                    kps_left=kps_left, 
                                    kps_right=kps_right, 
                                    joints_left=joints_left, 
                                    joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

"""
train
"""
if not args.evaluate:
    """
    train
    """
    cameras_train, poses_train, poses_train_2d = fetch(
                                                 subjects_train, 
                                                 action_filter, 
                                              subset=args.subset)

    lr = args.learning_rate
    if semi_supervised:
        cameras_semi, _, poses_semi_2d = fetch(subjects_semi, 
                                               action_filter, 
                                        parse_3d_poses=False)
        
        if not args.disable_optimizations and not args.dense and args.stride == 1:
            # Use optimized model for single-frame predictions
            model_traj_train = TemporalModelOptimized1f(
                               poses_valid_2d[0].shape[-2], 
                               poses_valid_2d[0].shape[-1], 
                               1,
                               filter_widths=filter_widths, 
                               causal=args.causal, 
                               dropout=args.dropout, 
                               channels=args.channels)
        else:
            # When incompatible settings are detected (stride > 1,             # dense filters, or disabled optimization) fall back 
            # to normal model

            model_traj_train = TemporalModel(
                               poses_valid_2d[0].shape[-2], 
                               poses_valid_2d[0].shape[-1], 
                               1,
                               filter_widths=filter_widths, 
                               causal=args.causal, 
                               dropout=args.dropout, 
                               channels=args.channels,
                               dense=args.dense)
       
        # Model 
        model_traj = TemporalModel(poses_valid_2d[0].shape[-2],  
                                   poses_valid_2d[0].shape[-1], 
                                   1,
                                   filter_widths=filter_widths, 
                                   causal=args.causal, 
                                   dropout=args.dropout, 
                                   channels=args.channels,
                                   dense=args.dense)
        # CUDA
        if torch.cuda.is_available():
            model_traj = model_traj.cuda()
            model_traj_train = model_traj_train.cuda()

        # Optimizer
        optimizer = optim.Adam(list(model_pos_train.parameters()) +
                               list(model_traj_train.parameters()),
                               lr=lr, 
                               amsgrad=True)
                               #weight_decay=1e-5)
       
        # Loss 
        losses_2d_train_unlabeled = []
        losses_2d_train_labeled_eval = []
        losses_2d_train_unlabeled_eval = []
        losses_2d_valid = []

        losses_traj_train = []
        losses_traj_train_eval = []
        losses_traj_valid = []
    else:
        optimizer = optim.Adam(model_pos_train.parameters(), 
                               lr=lr, amsgrad=True)
                               #weight_decay=1e-5)
        
    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    losses_vel= []
    losses_accel = []
    losses_mjt = []
    losses_mrpj = []
    losses_mpjpe = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    """
    training data chunks
    """
    
    train_generator = ChunkedGenerator(args.batch_size//args.stride, 
                                       cameras_train,  
                                       poses_train, 
                                       poses_train_2d, args.stride,
                                       pad=pad, 
                                       causal_shift=causal_shift, 
                                       shuffle=True, 
                                       augment=args.data_augmentation,
                                       kps_left=kps_left, 
                                       kps_right=kps_right, 
                                       joints_left=joints_left, 
                                       joints_right=joints_right)

    train_generator_eval = UnchunkedGenerator(cameras_train, 
                                              poses_train, 
                                              poses_train_2d,
                                              pad=pad, 
                                              causal_shift=causal_shift, 
                                              augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
    if semi_supervised:
        semi_generator = ChunkedGenerator(args.batch_size//args.stride, 
                                          cameras_semi, 
                                          None, 
                                          poses_semi_2d, 
                                          args.stride,
                                          pad=pad, 
                                 causal_shift=causal_shift, 
                                          shuffle=True,
                                          random_seed=4321, 
                            augment=args.data_augmentation,
                                          kps_left=kps_left, 
                                        kps_right=kps_right, 
                                    joints_left=joints_left, 
                                  joints_right=joints_right,
                                               endless=True)
        semi_generator_eval = UnchunkedGenerator(cameras_semi, 
                                                 None, 
                                                 poses_semi_2d,
                                                 pad=pad, 
                                     causal_shift=causal_shift, 
                                                 augment=False)
        print('INFO: Semi-supervision on {} frames'.format(semi_generator_eval.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']

        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        
        lr = checkpoint['lr']

        if semi_supervised:
            model_traj_train.load_state_dict(checkpoint['model_traj'])
            model_traj.load_state_dict(checkpoint['model_traj'])
            semi_generator.set_random_state(checkpoint['random_state_semi'])
            
    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')
    
    # Pos model only

    print("")
    print("TRAINING:")
    if(semi_supervised):
      print("SEMI-SUPERVISED reg warmup + semi-supervised back projection")
    else: 
      print("REGULAR supervised")
    print("")
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos_train.train()

        print("EPOCH", epoch+1, "/", args.epochs)

        """
        semi supervised
        """
        if semi_supervised:
            # TRAINING: Semi-supervised scenario
            model_traj_train.train()
            for (_, batch_3d, batch_2d), (cam_semi, _, batch_2d_semi) in \
                 zip(train_generator.next_epoch(), \
                 semi_generator.next_epoch()):

                """
                Fall back to supervised training for the first 
                epoch (to avoid instability)
                """
                skip = epoch < args.warmup
                
                cam_semi = torch.from_numpy(cam_semi.astype('float32'))
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                if torch.cuda.is_available():
                    cam_semi = cam_semi.cuda()
                    inputs_3d = inputs_3d.cuda()
                    
                inputs_traj = inputs_3d[:, :, :1].clone()
                inputs_3d[:, :, 0] = 0
              
                """ 
                Split point between labeled and unlabeled samples
                in the batch
                """

                split_idx = inputs_3d.shape[0]

                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_2d_semi = torch.from_numpy(batch_2d_semi.astype('float32'))

                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_semi = inputs_2d_semi.cuda()

                """
                torch.cat()
                Concatenates the given sequence of seq tensors in the 
                given dimension. So if A and B are of shape (3, 4), 
                torch.cat([A, B], dim=0) will be of shape (6, 4) and 
                torch.stack([A, B], dim=0) will be of shape (2, 3, 4).
                """

                inputs_2d_cat =  torch.cat((inputs_2d, 
                                            inputs_2d_semi), 
                                            dim=0) if not skip else inputs_2d
                optimizer.zero_grad()

                # Compute 3D poses, forward model
                predicted_3d_pos_cat = model_pos_train(inputs_2d_cat)

                # Compute losses
                loss_3d_pos = mpjpe(predicted_3d_pos_cat[:split_idx], inputs_3d)
                epoch_loss_3d_train += inputs_3d.shape[0] *\
                                       inputs_3d.shape[1] * \
                                       loss_3d_pos.item()
                N += inputs_3d.shape[0]*inputs_3d.shape[1]
                loss_total = loss_3d_pos * 1.1
                #print("loss_total", loss_total, 
                      #"loss_3d_pos", loss_3d_pos)

                # Compute global trajectory, forward model
                predicted_traj_cat = model_traj_train(inputs_2d_cat)

                # Weight inversely proportional to depth
                w = 1 / inputs_traj[:, :, :, 2] 

                # Compute loss
                loss_traj = weighted_mpjpe(predicted_traj_cat[:split_idx], inputs_traj, w)
                epoch_loss_traj_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_traj.item()
                assert inputs_traj.shape[0]*inputs_traj.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]
                #print("loss_total", loss_total, 
                       #"loss_traj", loss_traj)
                loss_total += loss_traj

                # Semi-supervised add loss terms here
                if(VEL_LOSS):
                  print("SEMI VEL")
                  #p = predicted_3d_pos_cat.detach().numpy()
                  #i = inputs_3d.detach().numpy()
                  #p = predicted_3d_pos_cat
                  #i = inputs_3d.detach()
                  p = predicted_3d_pos_cat.data.cpu().numpy()
                  i = inputs_3d.data.cpu().numpy()
                  #print("VEL_LOSS 1")
                  if(p.shape == i.shape):
                    vel_loss = mean_velocity_error(p, i, loss=True)
                    #losses_vel.append(vel_loss)
                    loss_total += vel_loss

                # Semi-supervised
                if(ACCEL_LOSS):
                  print("SEMI ACCEL")
                  #p = predicted_3d_pos_cat.detach().numpy()
                  #i = inputs_3d.detach().numpy()
                  #p = predicted_3d_pos_cat
                  #i = inputs_3d
                  p = predicted_3d_pos_cat.data.cpu().numpy()
                  i = inputs_3d.data.cpu().numpy()
                  if(p.shape == i.shape):
                    accel_loss = mean_acceleration_error(p, i, loss=True)
                    if torch.cuda.is_available():
                      accel_loss = accel_loss.cuda()
                    losses_accel.append(accel_loss)
                    loss_total += 10*accel_loss

                # Semi-supervised
                if(MRPJ_LOSS):
                  #p = predicted_3d_pos_cat.detach().numpy()
                  #i = inputs_3d.detach().numpy()
                  #p = predicted_3d_pos_cat
                  #i = inputs_3d.detach()
                  p = predicted_3d_pos_cat.data.cpu().numpy()
                  i = inputs_3d.data.cpu().numpy()
                  mrpj_loss = mrpj(p, i)
                  #losses_mrpj.append(mrpj_loss)
                  #print("LOSS", loss_total, "MRPJ_LOSS", mrpj_loss)
                  loss_total += mrpj_loss
  
                # Semi-supervised
                if(MJT_LOSS): # implemented
                  #p = predicted_3d_pos_cat.detach().numpy()
                  #i = inputs_3d.detach().numpy()
                  #p = predicted_3d_pos_cat
                  #i = inputs_3d
                  p = predicted_3d_pos_cat.data.cpu().numpy()
                  i = inputs_3d.data.cpu().numpy()
                  mjt_loss = mjt(p, i)
                  #losses_mjt.append(mjt_loss)
                  loss_total += mjt_loss

                # Semi-supervised
                if(PMPJPE_LOSS):
                  #p = predicted_3d_pos_cat[0].detach().numpy()
                  #i = inputs_3d[0].detach().numpy()
                  #p = predicted_3d_pos_cat[0]
                  #i = inputs_3d[0]
                  p = predicted_3d_pos_cat[0].data.cpu().numpy()
                  i = inputs_3d[0].data.cpu().numpy()
                  pmpjpe_loss, p, i = p_mpjpe(p, i, loss=True)
                  losses_mpjpe.append(mpjpe_loss)
                  print("PMPJPE_LOSS", loss_total, mpjpe_loss)
                  loss_total += pmpjpe_loss

                # Semi-supervised
                if not skip:
                    # Semi-supervised 2d back projection loss " + 
                    predicted_semi = predicted_3d_pos_cat[split_idx:]
                    if pad > 0:
                        target_semi = inputs_2d_semi[:, 
                                                     pad:-pad, 
                                           :, :2].contiguous()
                    else:
                        target_semi = inputs_2d_semi[:, :, :, :2].contiguous()

                    # 3d backproject to 2d
                    projection_func = project_to_2d_linear if args.linear_projection else project_to_2d
                    reconstruction_semi = projection_func(predicted_semi + predicted_traj_cat[split_idx:], cam_semi)
                    loss_reconstruction = mpjpe(reconstruction_semi, target_semi) # On 2D poses
                    epoch_loss_2d_train_unlabeled += predicted_semi.shape[0]*predicted_semi.shape[1] * loss_reconstruction.item()
                    if not args.no_proj:
                        loss_total += loss_reconstruction

                    # Bone length kinematic contraints
                    if args.bone_length_term:
                        #print("semi train: bone length term")
                        dists = predicted_3d_pos_cat[:, :, 1:] - \
                                predicted_3d_pos_cat[:, :, 
                                dataset.skeleton().parents()[1:]]
                        bone_lengths = torch.mean(torch.norm(dists,
                                                             dim=3), 
                                                  dim=1)
                        # Bone Loss
                        penalty = torch.mean(torch.abs( torch.mean(bone_lengths[:split_idx], dim=0) \
                                  - torch.mean(bone_lengths[split_idx:], dim=0)))
                        #print("loss_total", loss_total,
                              #"bone penalty", loss_traj)
                        loss_total += penalty
                    
                    N_semi += predicted_semi.shape[0]*predicted_semi.shape[1]
                else:
                    # regular supervised case
                    #print("warmup: skip semi-supervised back projection")
                    N_semi += 1       # To avoid division by zero


                print("LOSS", loss_total)

                loss_total.backward() # backpropogate
                optimizer.step()      # apply gradients

            # append 
            losses_traj_train.append(epoch_loss_traj_train / N)
            losses_2d_train_unlabeled.append(epoch_loss_2d_train_unlabeled / N_semi)

        else:
            """
            TRAINING: Regular supervised scenario
            in semi-supervised used as warmup
            """
            print("=== TRAINING: regular supervised scenario ===")
            for _, batch_3d, batch_2d in train_generator.next_epoch():
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                    inputs_2d = inputs_2d.cuda()
                inputs_3d[:, :, 0] = 0

                optimizer.zero_grad()

                # Compute 3D poses, forward model
                predicted_3d_pos = model_pos_train(inputs_2d)
                if torch.cuda.is_available():
                  predicted_3d_pos = predicted_3d_pos.cuda()
 
                ###### SUPERVISED LOSS STARTS #####################

                """
                # supervised protocol 1
                Mean per-joint position error (i.e. mean Euclidean distance),
                often referred to as "Protocol #1" in many papers.
                """

                base = mpjpe(predicted_3d_pos, inputs_3d)
                loss_3d_pos = base
                #epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item() # orig
                N += inputs_3d.shape[0]*inputs_3d.shape[1] # orig 

                loss_total = loss_3d_pos

                #if(DEBUG):
                  #print("LOSS TOTAL", loss_total.requires_grad)

                ###### SUPERVISED additional loss terms ########
                # supervised
                if(VEL_LOSS):
                  p = predicted_3d_pos
                  i = inputs_3d
                  if(p.shape == i.shape):
                    vel_loss = mean_velocity_error_torch(p, i) # torch
                    if(DEBUG):
                      print("LOSS TOTAL PRE", loss_total, "VEL_LOSS", vel_loss)
                    loss_total += vel_loss * 1
                    if(DEBUG):
                      print("LOSS TOTAL POST", loss_total, "VEL_LOSS", vel_loss)

                # supervised
                if(ACCEL_LOSS):
                  p = predicted_3d_pos
                  i = inputs_3d
                  if(p.shape == i.shape):
                    accel_loss = mean_acceleration_error_torch(p, i)
                    if(DEBUG):
                      print("LOSS TOTAL PRE", loss_total, "ACCEL_LOSS", accel_loss)
                    loss_total += accel_loss * 5
                    if(DEBUG):
                      print("LOSS TOTAL POST", loss_total, "ACCEL_LOSS", accel_loss)

                # supervised
                if(MRPJ_LOSS):
                  p = predicted_3d_pos
                  i = inputs_3d
                  mrpj_loss = mrpj_torch(p, i)
                  #mrpj_loss = mrpj_loss.cuda()
                  if(DEBUG):
                    print("LOSS", loss_total, "MRPJ_LOSS", mrpj_loss)
                  loss_total += mrpj_loss * 4
                  if(DEBUG):
                    print("LOSS TOTAL_POST", loss_total, "MRPJ_LOSS", mrpj_loss)

                """
                # supervised protocol 2
                Pose error: P_MPJPE after rigid alignment (scale, rotation, and translation),
                often referred to as "Protocol #2" in many papers.
                """
                if(PMPJPE_LOSS):

                  i = inputs_3d.cpu().detach().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                  p = predicted_3d_pos.cpu().detach().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                  #mpjpe_loss = p_mpjpe(p, i) # add back

                  p_mpjpe_loss_torch, p2, t2 = p_mpjpe_torch(p, i, loss=True) # add back

                  #ratio = loss_3d_pos.data / p_mpjpe_loss_torch.data
                  #p_mpjpe_loss_torch *= ratio 
                  if(not SCALARS_SET):
                    k_mpjpe = base.item() / p_mpjpe_loss_torch
                    SCALARS_SET = True
                  #else:
                    #k_t_loss = 2
 
                  if(DEBUG):
                    print("LOSS TOTAL_PRE", loss_total, "MPJPE_LOSS", p_mpjpe_loss_torch, p_mpjpe_loss_torch.requires_grad)

                  #loss_total += p_mpjpe_loss_torch 
                  loss_total += k_mpjpe

                  if(DEBUG):
                    print("LOSS TOTAL_POST", loss_total, "P_MPJPE_LOSS", p_mpjpe_loss_torch)

                # supervised
                if(MJT_LOSS): 
                  p = predicted_3d_pos
                  mjt_loss = mjt_torch(p, p2, i, t2, base.item())
                  if(DEBUG):
                    print("LOSS TOTAL PRE", loss_total, "MJT_LOSS", mjt_loss)
                  loss_total += mjt_loss
                  if(DEBUG):
                    print("LOSS TOTAL_POST", loss_total, "MJT_LOSS", mjt_loss)

                epoch_loss_3d_train += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_total.item() # added

                loss_total.backward() # backpropogate
                optimizer.step() # apply gradients

        losses_3d_train.append(epoch_loss_3d_train / N)

        """
        TRAINING PASS COMPLETE
        """

        """
        End-of-epoch evaluation/characterization

        model.eval():
        will notify all your layers that you are in eval mode, 
        that way, batchnorm or dropout layers will work in eval 
        mode instead of training mode.

        torch.no_grad(): 
        impacts the autograd engine and deactivate it. 
        reduces memory usage and speeds up computations but 
        won't be able to backprop (which you don't want in an 
        eval script)
        """
       
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict())
            model_pos.eval()
            if semi_supervised:
                model_traj.load_state_dict(model_traj_train.state_dict())
                model_traj.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            N = 0
            
            if not args.no_eval:
                # TRAINING: end of training, EVALUATE on TEST set
                print("epoch training evaluation")
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # Predict 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)

                    # Loss
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                    if semi_supervised:
                        cam = torch.from_numpy(cam.astype('float32'))
                        if torch.cuda.is_available():
                            cam = cam.cuda()

                        # Predict 3D trajectory
                        predicted_traj = model_traj(inputs_2d)
                        loss_traj = mpjpe(predicted_traj, inputs_traj)
                        epoch_loss_traj_valid += inputs_traj.shape[0]*inputs_traj.shape[1] * loss_traj.item()
                        assert inputs_traj.shape[0]*inputs_traj.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]

                        if pad > 0:
                            target = inputs_2d[:, pad:-pad, :, :2].contiguous()
                        else:
                            target = inputs_2d[:, :, :, :2].contiguous()
                        reconstruction = project_to_2d(predicted_3d_pos + predicted_traj, cam)
                        loss_reconstruction = mpjpe(reconstruction, target) # On 2D poses
                        epoch_loss_2d_valid += reconstruction.shape[0]*reconstruction.shape[1] * loss_reconstruction.item()
                        assert reconstruction.shape[0]*reconstruction.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)
                if semi_supervised:
                    losses_traj_valid.append(epoch_loss_traj_valid / N)
                    losses_2d_valid.append(epoch_loss_2d_valid / N)

                """
                TRAIN Evaluate on REG TRAINING set this time in 
                evaluation mode
                Logging Purposes only...
                """
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        """
                        This can only happen when downsampling 
                        the dataset
                        """
                        continue
                        
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # TRAIN, EVALUATE REG Predict 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                 
                    # TRAIN, EVALUATE REG Loss
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0]*inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0]*inputs_3d.shape[1]

                    if semi_supervised:
                        cam = torch.from_numpy(cam.astype('float32'))
                        if torch.cuda.is_available():
                            cam = cam.cuda()

                        # Predict 3D trajectory poses
                        predicted_traj = model_traj(inputs_2d)

                        # Loss
                        loss_traj = mpjpe(predicted_traj, inputs_traj)
                        epoch_loss_traj_train_eval += inputs_traj.shape[0]*inputs_traj.shape[1] * loss_traj.item()
                        assert inputs_traj.shape[0]*inputs_traj.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]

                        if pad > 0:
                            target = inputs_2d[:, pad:-pad, :, :2].contiguous()
                        else:
                            target = inputs_2d[:, :, :, :2].contiguous()

                        # Project 3D to 2D 
                        reconstruction = project_to_2d(predicted_3d_pos + predicted_traj, cam)

                        # LOSS 2D loss
                        loss_reconstruction = mpjpe(reconstruction, target)
                        epoch_loss_2d_train_labeled_eval += reconstruction.shape[0]*reconstruction.shape[1] * loss_reconstruction.item()
                        assert reconstruction.shape[0]*reconstruction.shape[1] == inputs_3d.shape[0]*inputs_3d.shape[1]
                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

                if semi_supervised:
                    losses_traj_train_eval.append(epoch_loss_traj_train_eval / N)
                    losses_2d_train_labeled_eval.append(epoch_loss_2d_train_labeled_eval / N)

                # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                epoch_loss_2d_train_unlabeled_eval = 0
                N_semi = 0
                if semi_supervised:
                    for cam, _, batch_2d in semi_generator_eval.next_epoch():
                        cam = torch.from_numpy(cam.astype('float32'))
                        inputs_2d_semi = torch.from_numpy(batch_2d.astype('float32'))
                        if torch.cuda.is_available():
                            cam = cam.cuda()
                            inputs_2d_semi = inputs_2d_semi.cuda()

                        # Predict 3D poses
                        predicted_3d_pos_semi = model_pos(inputs_2d_semi)
                        predicted_traj_semi = model_traj(inputs_2d_semi)
                        if pad > 0:
                            target_semi = inputs_2d_semi[:, pad:-pad, :, :2].contiguous()
                        else:
                            target_semi = inputs_2d_semi[:, :, :, :2].contiguous()

                        # Project 3D to 2D 
                        reconstruction_semi = project_to_2d(predicted_3d_pos_semi + predicted_traj_semi, cam)
                        loss_reconstruction_semi = mpjpe(reconstruction_semi, target_semi)

                        epoch_loss_2d_train_unlabeled_eval += reconstruction_semi.shape[0]*reconstruction_semi.shape[1] \
                                                              * loss_reconstruction_semi.item()
                        N_semi += reconstruction_semi.shape[0]*reconstruction_semi.shape[1]

                    # logging
                    losses_2d_train_unlabeled_eval.append(epoch_loss_2d_train_unlabeled_eval / N_semi)

                """
                TRAINING eval log calculation COMPLETE 
                """
        elapsed = (time() - start_time)/60

        """
        TRAIN STEP END 
        """

        """
        LOG STEP STATS 
        """
        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000))
        else:
            if semi_supervised:
                print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f traj_eval %f 3d_valid %f '
                      'traj_valid %f 2d_train_sup %f 2d_train_unsup %f 2d_valid %f ' %
                      #'vel %f accel %f mjt %f mrpj %f mrjpe %f' % 
                      #'mjt %f mrpj %f mrjpe %f' % 
                        (
                        epoch + 1,
                        elapsed,
                        lr,
                        losses_3d_train[-1] * 1000,
                        losses_3d_train_eval[-1] * 1000,
                        losses_traj_train_eval[-1] * 1000,
                        losses_3d_valid[-1] * 1000,
                        losses_traj_valid[-1] * 1000,
                        losses_2d_train_labeled_eval[-1],
                        losses_2d_train_unlabeled_eval[-1],
                        losses_2d_valid[-1], 
                        #losses_vel[-1],
                        #losses_accel[-1],
                        #losses_mjt[-1],
                        #losses_mrpj[-1],
                        #losses_mpjpe[-1]
                        ))
            else:
                print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                        epoch + 1,
                        elapsed,
                        lr,
                        losses_3d_train[-1] * 1000,
                        losses_3d_train_eval[-1] * 1000,
                        losses_3d_valid[-1]  *1000))
        print("")
        
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1
        
        # Decay BatchNorm momentum
        momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        model_pos_train.set_bn_momentum(momentum)
        if semi_supervised:
            model_traj_train.set_bn_momentum(momentum)
            
        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)
            
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)
           
        """
        plot
        """ 
        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            """
            if 'matplotlib' not in sys.modules:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
            """
            
            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            if semi_supervised:
                plt.figure()
                plt.plot(epoch_x, losses_traj_train[3:], '--', color='C0')
                plt.plot(epoch_x, losses_traj_train_eval[3:], color='C0')
                plt.plot(epoch_x, losses_traj_valid[3:], color='C1')
                plt.legend(['traj. train', 'traj. train (eval)', 'traj. valid (eval)'])
                plt.ylabel('Mean distance (m)')
                plt.xlabel('Epoch')
                plt.xlim((3, epoch))
                plt.savefig(os.path.join(args.checkpoint, 'loss_traj.png'))

                plt.figure()
                plt.plot(epoch_x, losses_2d_train_labeled_eval[3:], color='C0')
                plt.plot(epoch_x, losses_2d_train_unlabeled[3:], '--', color='C1')
                plt.plot(epoch_x, losses_2d_train_unlabeled_eval[3:], color='C1')
                plt.plot(epoch_x, losses_2d_valid[3:], color='C2')
                plt.legend(['2d train labeled (eval)', '2d train unlabeled', '2d train unlabeled (eval)', '2d valid (eval)'])
                plt.ylabel('MPJPE (2D)')
                plt.xlabel('Epoch')
                plt.xlim((3, epoch))
                plt.savefig(os.path.join(args.checkpoint, 'loss_2d.png'))
            plt.close('all')
    print("==== TRAINING COMPLETE on", epoch, "epochs ====\n")

"""
evaluate
"""

def evaluate(test_generator, action=None, return_predictions=False):
    print()
    print("=== EVALUATE ", 
      action, 
      "size")
    epoch_loss_3d_pos = 0 # Protocol 1
    epoch_loss_3d_pos_procrustes = 0 # Protocol 2
    epoch_loss_3d_pos_scale = 0 # Protocol 3
    epoch_loss_3d_vel = 0
    epoch_loss_3d_accel = 0
    epoch_loss_3d_bone_len = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model / Inference
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                #print("evaluate: test time augmentation")
                """
                Undo flipping and take average with non-flipped 
                version
                ([2, 1621, 17, 3])
                """
                #print("flip shape ", predicted_3d_pos.shape)
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right]= predicted_3d_pos[1, :, joints_right + joints_left]

                #print("jl+jr ", joints_left + joints_right)
                #print("jl+jr ", joints_left + joints_right)

                predicted_3d_pos = torch.mean(predicted_3d_pos, 
                                              dim=0, 
                                              keepdim=True)

            if return_predictions:
                print ("return_predictions", predicted_3d_pos)
                return predicted_3d_pos.squeeze(0).cpu().numpy()
                
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0    

            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            # Loss
       
            # Protocol #3 scaled only
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            # Protocol 1 euclidean distance hmmm
            error = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()

            # Protocal 2 scaled, rotated, fixed
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            """
            print("p_mpjpe", 
                  "predicted", 
                   type(predicted_3d_pos), predicted_3d_pos.shape, 
                   "inputs", 
                   type(inputs), inputs.shape)
            print(inputs)
            """

            #if(MJT_LOSS): # always true?
            if(False): # always true?
              p_mpjpe_loss, p, i = p_mpjpe_torch(predicted_3d_pos, inputs)   
              epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe_loss.cpu().numpy()
            else:
              p_mpjpe_loss = p_mpjpe(predicted_3d_pos, inputs)
              epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe_loss

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

            # Compute acceleration error
            epoch_loss_3d_accel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_acceleration_error(predicted_3d_pos, inputs)
 
    # stats
            
    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000  # Protocal 1
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000 # Protocal 2
    e3 = (epoch_loss_3d_pos_scale / N)*1000 # Protocol 3
    ev = (epoch_loss_3d_vel / N)*1000 # Velocity
    ac  = (epoch_loss_3d_accel / N)*1000 # Acceleration
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('Acceleration Error (MPJAPE):', ac, 'mm')
    print('----------')

    return e1, e2, e3, ev, ac

"""
render
"""

if args.render:
    print('Rendering...')
    
    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    else:
        ground_truth = None
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')
        
    gen = UnchunkedGenerator(None, 
                             None, 
                             [input_keypoints],
                             pad=pad, 
                             causal_shift=causal_shift, 
                             augment=args.test_time_augmentation,
                             kps_left=kps_left, 
                             kps_right=kps_right, 
                             joints_left=joints_left, 
                             joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)
    
    if ground_truth is not None:
        # Reapply trajectory
        trajectory = ground_truth[:, :1]
        ground_truth[:, 1:] += trajectory
        prediction += trajectory
    
    # Invert camera transformation
    cam = dataset.cameras()[args.viz_subject][args.viz_camera]
    if ground_truth is not None:
        prediction = camera_to_world(prediction, 
                                     R=cam['orientation'],  
                                     t=cam['translation'])
        ground_truth = camera_to_world(ground_truth, 
                                       R=cam['orientation'], 
                                       t=cam['translation'])
    else:
        """
        If the ground truth is not available, 
        take the camera extrinsic params from a random subject.
        They are almost the same, and anyway, 
        we only need this for visualization purposes.
        """
        for subject in dataset.cameras():
            if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                break
        prediction = camera_to_world(prediction, R=rot, t=0)

        """
        We don't have the trajectory, 
        but at least we can rebase the height
        """
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    
    anim_output = {'Reconstruction': prediction}
    if ground_truth is not None and not args.viz_no_ground_truth:
        anim_output['Ground truth'] = ground_truth
    
    input_keypoints = image_coordinates(input_keypoints[..., :2], 
                                        w=cam['res_w'],  
                                        h=cam['res_h'])
    
    from common.visualization import render_animation
    render_animation(input_keypoints, 
                     anim_output, # poses
                     dataset.skeleton(), 
                     dataset.fps(), 
                     args.viz_bitrate, cam['azimuth'], 
                     args.viz_output,
                     limit=args.viz_limit, 
                     downsample=args.viz_downsample, 
                     size=args.viz_size,
                     input_video_path=args.viz_video, 
                     viewport=(cam['res_w'], 
                     cam['res_h']),
                     input_video_skip=args.viz_skip)
    
else:
    print('==== Evaluating ====')
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))

    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)): # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        
        return out_poses_3d, out_poses_2d

    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
        errors_accel = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(None, 
                                     poses_act, 
                                     poses_2d_act,
                                     pad=pad, 
                                     causal_shift=causal_shift, 
                                     augment=args.test_time_augmentation,
                                     kps_left=kps_left, 
                                     kps_right=kps_right, 
                                     joints_left=joints_left, 
                                     joints_right=joints_right)
            e1, e2, e3, ev, ac = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)
            errors_accel.append(ac)

        # difficult targets
        print('Direction', 'Protocol #1')
        print('Direction', 'Protocol #2')

        print('Pose', 'Protocol #2')

        print('Purchase', 'Protocol #1')
        print('Purchase', 'Protocol #2')

        print('Sitting', 'Protocol #1')
        print('Sitting', 'Protocol #2')

        print('Wait', 'Protocol #2')
        print('WalkD', 'Protocol #2')

        # overall

        print('Protocol #1   (MPJPE) action-wise average:', 
               round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', 
               round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', 
               round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', 
               round(np.mean(errors_vel), 2), 'mm')
        print('Acceleration  (MPJVE) action-wise average:', 
               round(np.mean(errors_accel), 2), 'mm')

    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')

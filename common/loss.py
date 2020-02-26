# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
todo
- DoF component I can provide you on top of the 
  range of the motion, and I would like to have 
- Dynamics constraint such maximum accele/ation human for x muscle contraction  - 
- vector intersects
- human acceleration reversal limits
"""

FRAME_RATE = 50

import torch
import torch.nn as nn
import numpy as np
import common.m as m

def limits(predicted, target):
  sym_losses = []
  rom_losses = []
  #print("POS p", predicted.shape)
  for frame in predicted[:,0,:,:]:
    print("FRAME", frame)
    sym_loss, thetas, rom_loss = m.in_frame(frame)
    sym_losses.append(sym_loss)
    rom_losses.append(sym_loss)

  sym_loss = torch.mean(torch.tensor(np.norm(sym_losses)))
  rom_loss = torch.mean(torch.tensor(np.norm(rom_losses)))
  return sym_loss + rom_loss

def limits_torch(p1, p2, t1, t2, scale=None):
  #print("LIMITS_TORCH", "p1", p1.shape, "p2", p2.shape, "t1", t1.shape, "t2", t2.shape)
  #sym_loss, limits, theta_loss = m.in_frame_torch(predicted[:,0,:,:], target[:,0,:,:]) 
  losses = m.in_frame_torch(torch.tensor(p1[:,0,:,:]).cuda(), p2[0,:,:,:], torch.tensor(t1).cuda(), t2[0,:, :,:], scale)
  return losses

def mrpj_torch_example(predicted, target):
  """
  length ratios
  """
  #for p in predicted[:,0,:,:]:
  t_m = m.calculate_mean_ratios(target)
  loss = m.in_frame_ratio_torch(predicted[:,0,:,:], t_m)
  #loss = torch.mean(frames)
  return loss

def angular_vel(predicted, target):
  frames = []
  for frame in predicted[:,0,:,:]:
    thetas, limits = m.in_frame_torch(predicted[:,0,:,:], target[:,0,:,:])

  # angular velocity
  angle_vel = np.diff(frames, axis=0)/FRAME_RATE

  vels = []
  for frame in angle_vel:
    loss = m.in_frame_vel(frame)
    vels.append(loss)
  loss = torch.mean(torch.tensor(vels))
  return loss

def mrpj(predicted, target):
  """
  length ratios
  """

  frames = []
  t_m = m.calculate_mean_ratios(target)
  for p in predicted[:,0,:,:]:
    #print("RAT", "p", p, "t", t)
    frames.append(m.in_frame_ratio(p, t_m))
  loss = torch.mean(torch.norm(torch.tensor(frames)))
  return loss

def mrpj_torch(p, t):
  #print("MRPJ_TORCH", p.shape, t.shape)

  """
  length ratios
  """
  t_m = m.calculate_mean_ratios(t) 
  loss = m.in_frame_ratio_torch(p[:,0,:,:], t_m) #p1
  #loss = m.in_frame_ratio_torch(p[0,:,:,:], t_m) #p2
  return loss

def mjt(predicted, target):
  predicted_aligned, target = transform(predicted, target)
  predicted_aligned = predicted_aligned[np.newaxis]
  target = target[np.newaxis]
  return limits(predicted_aligned, target)

def mjt_torch(p1, p2, t1, t2, scale=None):
  return limits_torch(p1, p2, t1, t2, scale)

def mpjpe(predicted, target):
  """
  Mean per-joint position error (i.e. mean Euclidean distance),
  often referred to as "Protocol #1" in many papers.
  """
  assert predicted.shape == target.shape
  return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
  """
  Weighted mean per-joint position error (i.e. mean Euclidean distance)
  """
  assert predicted.shape == target.shape
  assert w.shape[0] == predicted.shape[0]
  return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def transform(predicted, target):
  assert predicted.shape == target.shape

  muX = np.mean(target, axis=1, keepdims=True)
  muY = np.mean(predicted, axis=1, keepdims=True)

  X0 = target - muX
  Y0 = predicted - muY

  normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
  normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

  X0 /= normX
  Y0 /= normY

  H = np.matmul(X0.transpose(0, 2, 1), Y0)

  # still converting
  U, s, Vt = np.linalg.svd(H)
  V = Vt.transpose(0, 2, 1)
  R = np.matmul(V, U.transpose(0, 2, 1))

  # Avoid improper rotations (reflections),
  # i.e. rotations with det(R) = -1
  sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
  V[:, :, -1] *= sign_detR
  s[:, -1] *= sign_detR.flatten()
  R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

  tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

  a = tr * normX / normY # Scale
  t = muX - a*np.matmul(muY, R) # Translation

  # Perform rigid transformation on the input
  predicted_aligned = a*np.matmul(predicted, R) + t

  a = tr * normX / normY # Scale
  t = muX - a*np.matmul(muY, R) # Translation

  # Perform rigid transformation on the input
  predicted_aligned = a*np.matmul(predicted, R) + t

  return predicted_aligned, target

def p_limits(predicted, target, loss=False):
  predicted_aligned, target = transform(predicted, target)
  predicted_aligned = predicted_aligned[np.newaxis]
  target = target[np.newaxis]

  print("INVALID tranformation limits")
  return limits(predicted_aligned, target)

def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t

    # Return P_MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))

def p_mpjpe_torch(predicted, target, loss=False):
  """
  Pose error: MPJPE after rigid alignment 
  (scale, rotation, and translation), often referred to as 
  "Protocol #2" in many papers.
  predicted_aligned, target = transform(predicted, target)
  """

  predicted_aligned, target = transform(predicted, target)
  predicted_aligned = predicted_aligned[np.newaxis]
  target = target[np.newaxis]

  predicted = torch.tensor(predicted).cuda()
  target = torch.tensor(target).cuda()

  # Return P_MPJPE
  if(loss):
    return torch.mean(torch.abs(torch.tensor(predicted_aligned).cuda() - torch.tensor(target))).cuda(), torch.tensor(predicted_aligned).cuda(), torch.tensor(target).cuda()
  else:
    return torch.mean(torch.abs(torch.tensor(predicted_aligned).cuda() - torch.tensor(target))).cuda(),None,None
    
def scale(predicted, target):
  assert predicted.shape == target.shape

  norm_predicted = torch.mean(torch.sum(predicted**2, 
                                        dim=3, 
                                        keepdim=True), 
                              dim=2, 
                              keepdim=True)
  norm_target = torch.mean(torch.sum(target*predicted, 
                                     dim=3, 
                                     keepdim=True), 
                           dim=2, 
                           keepdim=True)
  s = norm_target / norm_predicted
  return s

def n_mpjpe(predicted, target):
  """
  Normalized MPJPE (scale only), adapted from:
  https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
  """
  s = scale(predicted, target)
  #print("Scale limits")
  #limits(s*predicted, target)
  return mpjpe(s * predicted, target)

def mean_velocity_error(predicted, target, loss=False):
  assert predicted.shape == target.shape

  """
  np.diff equivalent https://discuss.pytorch.org/t/equivalent-function-like-numpy-diff-in-pytorch/35327/2
  """
  velocity_predicted = np.diff(predicted, axis=0)
  velocity_target = np.diff(target, axis=0)
  return np.mean(np.linalg.norm(velocity_predicted -
                                velocity_target,
                                axis=len(target.shape)-1))

def mean_velocity_error_torch(predicted, target):
  assert predicted.shape == target.shape

  """
  np.diff equivalent https://discuss.pytorch.org/t/equivalent-function-like-numpy-diff-in-pytorch/35327/2
  """

  """
  print("MVET", predicted.shape)
  print("MVET PREDICTED VEL", velocity_predicted.shape)
  print("MVET PREDICTED VEL", predicted[0,0], predicted[1, 0])
  print("first", predicted[:-1,0,0])
  print("second", predicted[1:,0,0]) 
  """

  """
  velocity_predicted = predicted[1:] - predicted[:-1]
  velocity_target = target[1:] - target[:-1]
  return torch.mean(torch.abs(velocity_predicted - velocity_target))
  """

  #velocity_predicted= torch.dist(predicted[:-1], predicted[1:])
  #velocity_target = torch.dist(target[:-1], target[1:])

  velocity_predicted = distances(predicted)
  velocity_target = distances(target)

  loss = nn.MSELoss()
  return loss(velocity_predicted, velocity_target)

def distances(b):
  a = []
  #frame = torch.tensor(0).cuda()
  for i in range(0,17):
    d = distance(b[:,0,i][:-1], b[:,0,i][1:])
    a.append(d)
  frame = torch.stack(a).cuda()
  #print("FRAME", frame.shape)
  return frame

def distance(a, b):
  o = torch.tensor([0,0,0]).float().cuda()
  o = torch.zeros(a.shape).cuda()
  #print(a, b)
  #d = ((b[:,] - o) - (a[:,] - o), o).cuda()
  d = (b[:,] - a[:,]).cuda()

  d = distance_broadcastable(d).float()

  #print("D", d.shape, d)

  return d

def distance_broadcastable(t):
  sq = t*t
  sum = torch.sum(sq,dim=1)
  d = torch.sqrt(sum)
  #print("BROADCASTABLE SUM", sum.shape, sum)
  #print("BROADCASTABLE D", d.shape, d)
  return d

def mean_acceleration_error(predicted, target, loss=False):
  assert predicted.shape == target.shape

  acceleration_predicted = np.diff(np.diff(predicted, axis=0), axis=0)
  acceleration_target = np.diff(np.diff(target, axis=0), axis=0)
  return np.mean(np.linalg.norm(acceleration_predicted -
                                acceleration_target,
                                axis=len(target.shape)-1))

def mean_acceleration_error_torch(predicted, target):
  assert predicted.shape == target.shape

  """
  acceleration_predicted = predicted[1:] - predicted[:-1]
  acceleration_predicted = acceleration_predicted[1:] - acceleration_predicted[:-1]
  acceleration_target = target[1:] - target[:-1]
  acceleration_target = acceleration_target[1:] - acceleration_target[:-1]
  #return torch.mean(torch.abs(acceleration_predicted -
                                #acceleration_target))
  """
  #print("SHAPE", predicted.shape)
  vel_predicted= distances(predicted)
  vel_target = distances(target)
  accel_predicted = torch.abs(vel_predicted[-1:] - vel_predicted[:1])
  accel_target = torch.abs(vel_target[-1:] - vel_target[:1])

  loss = nn.MSELoss()
  return loss(accel_predicted, accel_target)


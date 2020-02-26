"""
libs
"""
#import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from torch.nn.modules.distance import PairwiseDistance as pwd
import numpy as np
from common.skeleton import Skeleton
import vg
import torch
import torch.nn as nn

"""
defines
"""
DEBUG = False

SCALARS_SET = False
k_sym_loss =  0
k_t_loss = 0
k_rom = 0
k_vel_loss = 0
k_accel_loss = 0
k_k_limit = 0

k_kin = 1.0
k_dyn = 1.0

INVALID = -1
PELVIS = 0

SPINE = 7
CHEST = 8
NECK = 9
HEAD = 10

L_HIP = 4
L_KNEE = 5
L_FOOT = 6
L_SHOULDER = 11
L_ELBOW = 12
L_HAND = 13

R_HIP = 1
R_KNEE = 2
R_FOOT = 3
R_SHOULDER = 14
R_ELBOW = 15
R_HAND = 16
ROOT = 17

PARENTS = [INVALID, PELVIS, R_HIP, R_KNEE,
           PELVIS, L_HIP, L_KNEE, PELVIS,
           SPINE, CHEST, NECK, CHEST, L_SHOULDER,
           L_ELBOW, CHEST, R_SHOULDER, R_ELBOW]

# legs top to bottom (3), arms top to bottom (3)
LEFT = [L_HIP, L_KNEE, L_FOOT, L_SHOULDER, L_ELBOW, L_HAND]
RIGHT = [R_HIP, R_KNEE, R_FOOT, R_SHOULDER, R_ELBOW, R_HAND]

s = Skeleton(PARENTS, LEFT, RIGHT)

# CENTER
s._meta[PELVIS] = {"location":"center",
                   "joint":"pelvis",
                   "joints":(ROOT, PELVIS, SPINE),
                   "min":0,
                   "max":135}

s._meta[SPINE] = {"location":"center",
                  "joint":"spine",
                  "joints":(PELVIS, SPINE, CHEST),
                  "min":90,
                  "max":180}

s._meta[CHEST] = {"location":"center",
                  "joint":"chest"}

s._meta[NECK] = {"location":"center", 
                 "joint":"neck",
                  "joints":(CHEST, NECK, HEAD),
                  "min":90, 
                  "max":180}

s._meta[HEAD] = {"location":"center", "joint":"head"}

# LEFT
s._meta[L_HIP] = {"location":"left", 
                  "joint":"hip",
                  "joints":(PELVIS, L_HIP, L_KNEE),
                  "min":45, 
                  "max":180}
s._meta[L_KNEE] = {"location":"left", 
                   "joint":"knee", 
                   "joints":(L_HIP, L_KNEE, L_FOOT),
                   "min":45, 
                   "max":180}
s._meta[L_FOOT] = {"location":"left", 
                   "joint":"foot"}
s._meta[L_SHOULDER] = {"location":"left", 
                       "joint":"shoulder",
                       "joints":(SPINE, CHEST, L_SHOULDER),
                       "min":0, 
                       "max":135}
s._meta[L_ELBOW] = {"location":"left", 
                    "joint":"elbow", 
                    "joints":(L_SHOULDER, L_ELBOW, L_HAND),
                    "min":0, 
                    "max":180}
s._meta[L_HAND] = {"location":"left", 
                   "joint":"hand"}

# RIGHT
s._meta[R_HIP] = {"location":"right", 
                  "joint":"hip",
                  "joints":(PELVIS, R_HIP, R_KNEE),
                  "min":45, 
                  "max":180}
s._meta[R_KNEE] = {"location":"right", 
                   "joint":"knee", 
                   "joints":(R_HIP, R_KNEE, R_FOOT),
                   "min":45, 
                   "max":180}
s._meta[R_FOOT] = {"location":"right", 
                   "joint":"foot"}
s._meta[R_SHOULDER] = {"location":"right", 
                       "joint":"shoulder",
                       "joints":(SPINE, CHEST, R_SHOULDER),
                       "min":0, 
                       "max":135}
s._meta[R_ELBOW] = {"location":"right", 
                    "joint":"elbow", 
                    "joints":(R_SHOULDER, R_ELBOW, R_HAND),
                    "min":0, 
                    "max":180}
s._meta[R_HAND]  = {"location":"right", 
                    "joint":"hand"}
s._meta[ROOT]  = {"location":"center", 
                  "joint":"root"}

meta = s.meta()
invalids = []
max_invalid_dot_l_knee = 0
max_invalid_dot_r_knee = 0
max_invalid_dot_l_elbow = 0
max_invalid_dot_r_elbow = 0
valid = 0

BATCH = 10000
MAX_SYM_LOSS  = 0.0325
MAX_RATIO_LOSS  = 0.25
SYM_LOSS = True
RATIO_LOSS = True
THETA_LOSS = True
THETA_DOT_LOSS = True

# velocities
FLEXION = 0
EXTENSION = 1

VEL_LIMIT = 10
ACCEL_LIMIT = 20

vels = {}

"""
Data from Maximum Velocities in Flexion-hukin-2015-0139-1.pdf
"""

# shoulder
f = {'ISO': 15.0, 'ISOv': 2.9,
     'CM': 16.6, 'CMv': 2.9, 
     'UR': 17.6, 'URv': 2.5}
e = {'ISO': 18.6, 'ISOv': 3.6,
     'CM': 16.1, 'CMv': 2.0, 
     'UR': 18.7, 'URv': 2.8}
vels["SHOULDER"] = {"f": f, "e": e}

# elbow
f = {'ISO': 18.6, 'ISOv': 3.0,
     'CM': 17.4, 'CMv': 3.6, 
     'UR': 19.9, 'URv': 1.5}
e = {'ISO': 25.6, 'ISOv': 5.8,
     'CM': 25.1, 'CMv': 5.7, 
     'UR': 27.9, 'URv': 3.7}
vels["ELBOW"] = {"f": f, "e": e}

# hip 
f = {'ISO': 12.0, 'ISOv': 1.2,
     'CM': 11.6, 'CMv': 1.3, 
     'UR': 12.0, 'URv': 1.6}
e = {'ISO': 12.4, 'ISOv': 12.8,
     'CM': 13.3, 'CMv': 1.4, 
     'UR': 14.1, 'URv': 2.6}
vels["HIP"] = {"f": f, "e": e}

# knee
f = {'ISO': 16.6, 'ISOv': 5.2,
     'CM': 18.1, 'CMv': 2.5, 
     'UR': 18.6, 'URv': 3.5}
e = {'ISO': 22.4, 'ISOv': 3.6,
     'CM': 24.3, 'CMv': 3.4, 
     'UR': 28.4, 'URv': 3.6}
vels["KNEE"] = {"f": f, "e": e}

"""
functions
"""

def plot(p, meta, data):
  """
  lc = mc.LineCollection(data)
  fig, ax = plt.subplots()
  plt.title("Kinematic Tree")
  points = ax.scatter(p[:,0], 
                      p[:,1], 
                      color='green', 
                      edgecolors='yellow', zorder=10)

  for i, r in enumerate(p):
    theta = "" if(not "theta" in meta[i]) else \
               str(meta[i]["theta"])
      
    ax.annotate(meta[i]["location"] + " " + 
                meta[i]["joint"],
               (r[0], r[1]))
    ax.annotate(theta,
               (r[0], r[1]-10))
  ax.add_collection(lc)
  ax.autoscale()
  ax.margins(0.1)
  plt.grid(True)
  plt.show()
  #print(s.get_parents(16))
  """
  lc = Line3DCollection(data)
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  #fig, ax = plt.subplots()
  plt.title("Kinematic Tree")
  points = ax.scatter(p[:,0],
                      p[:,1],
                      p[:,2],
                      color='green',
                      edgecolors='yellow', zorder=10)

  """
  # annotate
  for i, r in enumerate(data):
    r = r[1].tolist()
    theta = "" if(not "theta" in meta[i]) else \
               str(meta[i]["theta"])

    ax.text(r[0], r[1], r[2], meta[i]["location"]+ " " + meta[i]["joint"])
    ax.text(r[0], r[1]-.1, r[2], theta)
  """
  ax.add_collection(lc)
  ax.autoscale()
  ax.margins(0.1)
  ax.set_xlabel('X axis')
  ax.set_ylabel('Y axis')
  ax.set_zlabel('Z axis')
  plt.grid(True)
  plt.show()

def unsigned_angle(a, b): # unused
  dot = (np.dot(a, b))
  rad = np.arccos(np.dot(a, b)/
                 (np.linalg.norm(a) * np.linalg.norm(b)))
  #print("Dot", dot, "Rad", rad)
  return np.array(rad).flatten()

def scrub_data(p):
  #p = np.array(p)
  p *= -1 # flip vertically
  return p

def joint_angles(p):
  data = [] 
  for j, j_parent in enumerate(PARENTS):
    if (j_parent == INVALID):
      continue
    data.append([(p[j,0], p[j,1]), (p[j_parent,0], p[j_parent,1])])
  return p, meta, data

def pose(p):
  p, meta, data = joint_angles(p)
  plot(p, meta, data)

def vect_angle(a, b):
  """
  vg:
  sign = np.array(np.sign(np.cross(v1, v2).dot(look)))
  # 0 means collinear: 0 or 180. Let's call that clockwise.
  sign[sign == 0] = 1
  return sign * angle(v1, v2, look, units=units)
  """
  #print (a, b)
  #return np.radians(vg.signed_angle(a, b, look=vg.basis.z))
  return angle_torch(a, b)

def angle_torch(v1, v2, look=None, assume_normalized=False, units="deg"):
    #print("angle_torch")
    """
    Compute the unsigned angle between two vectors. For stacked inputs, the
    angle is computed pairwise.

    When `look` is provided, the angle is computed in that viewing plane
    (`look` is the normal). Otherwise the angle is computed in 3-space.

    Args:
        v1 (np.arraylike): A `3x1` vector or a `kx3` stack of vectors.
        v2 (np.arraylike): A vector or stack of vectors with the same shape as
            `v1`.
        look (np.arraylike): A `3x1` vector specifying the normal of a viewing
            plane, or `None` to compute the angle in 3-space.
        assume_normalized (bool): When `True`, assume the input vectors
            are unit length. This improves performance, however when the inputs
            are not normalized, setting this will cause an incorrect results.
        units (str): `'deg'` to return degrees or `'rad'` to return radians.

    Return:
        object: For `3x1` inputs, a `float` with the angle. For `kx1` inputs,
            a `kx1` array.
    """
    if units not in ["deg", "rad"]:
        raise ValueError("Unrecognized units {}; expected deg or rad".format(units))

    if look is not None:
        # This is a simple approach. Since this is working in two dimensions,
        # a smarter approach could reduce the amount of computation needed.
        v1, v2 = [reject(v, from_v=look) for v in (v1, v2)]

    dot_products = torch.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))

    if assume_normalized:
        cosines = dot_products
    else:
        cosines = dot_products / magnitude_torch(v1) / magnitude_torch(v2)

    # Clip, because the dot product can slip past 1 or -1 due to rounding and
    # we can't compute arccos(-1.00001).
    angles = torch.acos(torch.clamp(cosines, -1.0, 1.0))
    if units == "deg":
        angles = degrees_torch(angles)
    #print("angles", angles.shape, angles)

    return angles[0] if v1.ndimension == 1 and v2.ndimension == 1 else angles

def degrees_torch(radians):
  return radians/np.pi  * 180

def magnitude_torch(vector):
    """
    Compute the magnitude of `vector`. For stacked inputs, compute the magnitude
    of each one.

    Args:
        vector (np.arraylike): A `3x1` vector or a `kx3` stack of vectors.

    Returns:
        object: For `3x1` inputs, a `float` with the magnitude. For `kx1`
            inputs, a `kx1` array.
    """
    #print("mag torch vector", vector.shape)
    #if vector[:].ndimension == 1:
    return torch.norm(vector[:], dim=1)
    #elif vector.ndimension == 2:
        #return torch.norm(vector, dim=1)

def in_frame_vel(m):
  if(not THETA_DOT_LOSS):
    return
  global valid
  p = np.array(m)

  loss = 0 

  loss += incr_validity(np.abs(p[3]), 
                vels['HIP']['e']['UR'])      # L_HIP
  loss += incr_validity(np.abs(p[4]), 
                vels['SHOULDER']['e']['UR']) # L_SHOULDER
  loss += incr_validity(np.abs(p[5]), 
                vels['KNEE']['e']['UR'])     # L_KNEE
  loss += incr_validity(np.abs(p[6]), 
                vels['ELBOW']['e']['UR'])    # L_ELBOW
  loss += incr_validity(np.abs(p[7]), 
                vels['HIP']['e']['UR'])      # R_HIP
  loss += incr_validity(np.abs(p[8]), 
                vels['SHOULDER']['e']['UR']) # R_SHOULDER
  loss += incr_validity(np.abs(p[9]), 
                vels['KNEE']['e']['UR'])     # R_KNEE
  loss += incr_validity(np.abs(p[10]), 
                vels['ELBOW']['e']['UR'])    # R_ELBOW

  print("in_frame_vels loss", loss)

  return float(loss)

def incr_validity(vel, vels):
  global valid
  if (vel > vels):
    invalids.append(m)
    print("invalid:", vel - vels)
    return vel - vels
  else:
    valid +=1 
  return 0

def in_frame_ratio(m, mean):
  if(not RATIO_LOSS):
    return
  global valid
  r = calculate_ratios(m)

  """
  r = np.array(r)
  m = np.array(mean)
  d = np.mean(np.norm(r-m))
  """
  m = torch.tensor(mean)
  d = torch.mean(torch.norm(r-m))

  loss = 0
  if(d > MAX_RATIO_LOSS):
    invalids.append(m)
    loss = d
  else:
    valid +=1 
  return float(loss)

def in_frame_ratio_torch(m, mean):
  if(not RATIO_LOSS):
    return
  global valid
  r = calculate_ratios_torch(m)

  m = torch.tensor(mean)
  d = torch.mean(torch.norm(r-m))

  loss = 0
  if(d > MAX_RATIO_LOSS):
    invalids.append(m)
    loss = d
  else:
    valid +=1
  return loss

def in_frame(m):
  sym_loss = calculate_symmetry(m)
  #rom_loss
  thetas = calculate_thetas(m)
  return thetas

#def in_frame_torch(m1, m2, r1, r2, base):
def in_frame_torch(p, i, base):
  """
  print("m", m[0], "r", r[0])
  """
  global SCALARS_SET
  global k_sym_loss
  global k_t_loss
  global k_rom
  global k_vel_loss
  global k_accel_loss
  global k_k_limit
  global k_kin
  global k_dyn

  #if(DEBUG):
    #print("IN_FRAME_TORCH", m1.shape, m2.shape, r1.shape, r2.shape)

  sym_loss = calculate_symmetry_torch(p)
  t1, r_t1, tt1, r_tt1, t_loss, rom, r_rom1, k_limit = theta_mean(p, i) 

  loss = nn.MSELoss()

  vel_loss = loss(t1, r_t1)
  accel_loss = loss(tt1, r_tt1) 

  """
  losses = t_loss * 8 + rom + vel_loss*4 + accel_loss*4 + k_limit/2 # arc=1
  losses = t_loss * 32 + rom/2 + (vel_loss + accel_loss) * 8 # t*32,  rom/ 2 k * 8 is good!
  """
  #losses = (t_loss * 8 + rom * 2) * 2
  #losses += ((vel_loss+accel_loss)*8  + k_limit*16)*2

  k = (p.shape[0] * (p.shape[1]-1)) 
  
  sym_loss = sym_loss/k
  t_loss = t_loss/k
  rom = rom/k
  vel_loss = vel_loss/k
  accel_loss = accel_loss/k
  k_limit = k_limit/k
  rom = rom/k

  """
  if(not SCALARS_SET):
    #print("SET SCALARS")
    k_sym_loss =  base / sym_loss
    k_t_loss = base / t_loss
    k_rom = base / rom
    k_vel_loss = base / vel_loss
    k_accel_loss = base / accel_loss
    k_k_limit = base / k_limit
    SCALARS_SET = True
  else:
    k_t_loss = 2 # 33333e35d1k2d2bm.log

  sym_loss = sym_loss * k_sym_loss
  t_loss = t_loss * k_t_loss
  rom = rom * k_rom
  vel_loss = vel_loss * k_vel_loss
  accel_loss = accel_loss * k_accel_loss
  k_limit = k_limit * k_k_limit
  """

  if(DEBUG):
    print("S", sym_loss, "T", t_loss, "R", rom, "V", vel_loss, "A", accel_loss, "K", k_limit)

  kin = (sym_loss + t_loss + rom) * k_kin
  dyn = (vel_loss + accel_loss) * k_dyn
  losses = (kin+dyn)

  losses = losses.cuda()

  #print("LOSSES", losses)
  return losses 

def theta_mean(m, r):
  rom, theta = calculate_thetas_torch(m) 

  """
  print("PREDICTED")
  print("THETA", theta)
  print("M", m[0])
  print("M_THETA", theta)
  """

  r_rom, r_theta = calculate_thetas_torch(r) 

  """
  print("GROUND TRUTH")
  print("R", r[0])
  print("R_THETA",r_theta)
  """

  # theta loss
  loss = nn.MSELoss()
  #theta_loss = torch.mean(torch.norm(theta - r_theta).cuda()).cuda()
  theta_loss = loss(theta, r_theta)
  #print("THETA_LOSS", theta_loss)

  # vel loss
  t =  torch.abs(theta[:,1:] - theta[:,:-1])
  r_t = torch.abs(r_theta[:,1:] - r_theta[:,:-1])
  vel_limit = kin_oob(t, VEL_LIMIT)
  #m_t = torch.mean(torch.norm(t - r_t))
  m_t = loss(t, r_t)
  #print("M_VEL", t, "R_VEL", r_t, "V_LOSS", m_t)
  # accel loss
  tt =  torch.abs(t[:,1:] - t[:,:-1])
  r_tt = torch.abs(r_t[:,1:] - r_t[:,:-1])
  accel_limit = kin_oob(tt, ACCEL_LIMIT)
  #m_tt = torch.mean(torch.norm(tt - r_tt))
  m_tt = loss(tt, r_tt)
  #print("M_ACCEL", tt, "R_ACCEL", r_tt, "V_LOSS", m_tt)
  
  k_limit  = vel_limit + accel_limit
 
  return t, r_t, tt, r_tt, theta_loss, rom, r_rom, k_limit

def calculate_symmetry_torch(p):
  if(not SYM_LOSS):
    return
  global valid
  
  #l_armv = torch.mean(torch.norm(torch.norm(torch.norm(p[:,L_HAND], dim=1) - torch.norm(p[:,L_ELBOW], dim=1)) - torch.norm(torch.norm(p[:,R_HAND],dim=1) - torch.norm(p[:,R_ELBOW], dim=1)))).cuda()
  #u_armv = torch.mean(torch.norm(torch.norm(torch.norm(p[:,L_ELBOW], dim=1) - torch.norm(p[:,L_SHOULDER], dim=1)) - torch.norm(torch.norm(p[:,R_ELBOW],dim=1) - torch.norm(p[:,R_SHOULDER], dim=1)))).cuda()
  #l_legv = torch.mean(torch.norm(torch.norm(torch.norm(p[:,L_FOOT], dim=1) - torch.norm(p[:,L_KNEE], dim=1)) - torch.norm(torch.norm(p[:,R_FOOT],dim=1) - torch.norm(p[:,R_KNEE], dim=1)))).cuda()
  #u_legv = torch.mean(torch.norm(torch.norm(torch.norm(p[:,L_KNEE], dim=1) - torch.norm(p[:,L_HIP], dim=1)) - torch.norm(torch.norm(p[:,R_KNEE],dim=1) - torch.norm(p[:,R_HIP], dim=1)))).cuda()

  loss = nn.MSELoss()
  l_armv = loss(torch.norm(torch.norm(p[:,L_HAND], dim=1) - torch.norm(p[:,L_ELBOW], dim=1)), torch.norm(torch.norm(p[:,R_HAND],dim=1) - torch.norm(p[:,R_ELBOW], dim=1)))
  u_armv = loss(torch.norm(torch.norm(p[:,L_ELBOW], dim=1) - torch.norm(p[:,L_SHOULDER], dim=1)), torch.norm(torch.norm(p[:,R_ELBOW],dim=1) - torch.norm(p[:,R_SHOULDER], dim=1)))
  l_legv = loss(torch.norm(torch.norm(p[:,L_FOOT], dim=1) - torch.norm(p[:,L_KNEE], dim=1)), torch.norm(torch.norm(p[:,R_FOOT],dim=1) - torch.norm(p[:,R_KNEE], dim=1)))
  u_legv = loss(torch.norm(torch.norm(p[:,L_KNEE], dim=1) - torch.norm(p[:,L_HIP], dim=1)),torch.norm(torch.norm(p[:,R_KNEE],dim=1) - torch.norm(p[:,R_HIP], dim=1)))

  """
  print("L_ARMV", l_armv)
  print("U_ARMV", u_armv)
  print("L_LEGV", l_legv)
  print("U_LEGV", u_legv)
  """
  sum = torch.mean(torch.sum(torch.tensor([l_armv, u_armv, l_legv, u_legv]))).cuda()
  #print("SYMLOSS", sum)
  return sum 

def calculate_symmetry(m):
  if(not SYM_LOSS):
    return
  global valid
  #print("SYMMETRY", print(m), m)
  p = np.array(m)
  l_arm = np.linalg.norm(p[L_HAND] - p[L_ELBOW]) - \
          np.linalg.norm(p[R_HAND] - p[R_ELBOW])
  u_arm = np.linalg.norm(p[L_ELBOW] - p[L_SHOULDER]) - \
          np.linalg.norm(p[R_ELBOW] - p[R_SHOULDER])
  l_leg = np.linalg.norm(p[L_FOOT] - p[L_KNEE]) - \
          np.linalg.norm(p[R_FOOT] - p[R_KNEE])
  u_leg = np.linalg.norm(p[L_KNEE] - p[L_HIP]) - \
          np.linalg.norm(p[R_KNEE] - p[R_HIP])
  #print("LOSS sym:", l_arm, u_arm, l_leg, u_leg)
  
  if(np.abs(l_arm) > MAX_SYM_LOSS or 
     np.abs(u_arm) > MAX_SYM_LOSS or 
     np.abs(l_leg) > MAX_SYM_LOSS or 
     np.abs(u_leg) > MAX_SYM_LOSS):
    #print("INVALIDS sym", l_arm, u_arm, l_leg, u_leg)
    invalids.append(p)
  else:
    valid += 1

  return (l_arm, u_arm, l_leg, u_leg)

def calculate_mean_ratios(target):
  #print("CALCULATE_MEAN_RATIONS", target.shape)
  """
   trunk
  """
  h = mean_trunk_ratio(target, L_HIP, PELVIS, R_HIP, PELVIS)
  #print("hip", h)

  s = mean_trunk_ratio(target, L_SHOULDER, CHEST, R_SHOULDER, CHEST)
  #print("shoulder", s)

  """
  limbs
  """
  l_a = mean_limb_ratio(target, L_SHOULDER, L_ELBOW, L_HAND)
  #print("left arm", l_a)

  r_a = mean_limb_ratio(target, R_SHOULDER, R_ELBOW, R_HAND)
  #print("left arm", r_a)

  l_l = mean_limb_ratio(target, L_HIP, L_KNEE, L_FOOT)
  #print("left leg", l_l)

  r_l = mean_limb_ratio(target, R_HIP, R_KNEE, R_FOOT)
  #print("right leg", r_l)

  return((h, s, l_a, r_a, l_l, r_l))

def mean_limb_ratio(t, a, b, c):
  a = mean_length(t, a, b)
  b = mean_length(t, b, c)
  return a/b

def mean_trunk_ratio(t, a, b, c, d):
  a = mean_length(t, a, b)
  b = mean_length(t, c, d)
  return a/b

def mean_length(target, a, b):
  """
  mean = np.mean(np.linalg.norm(target[:,:,a,:]-
                 target[:,:, b,:],
                 axis=0))
  """
  mean = torch.mean(torch.dist(target[:,:,a,:],
                 target[:,:, b,:]))
  return mean

def length(p, a, b):
  #return np.linalg.norm(p[a] - p[b])
  return torch.dist(p[:,a], p[:,b])

def calculate_ratios_torch(p):
  """
  trunk
  """
  l_shoulder = length(p, CHEST, L_SHOULDER)
  r_shoulder = length(p, CHEST, R_SHOULDER)
  s = l_shoulder / r_shoulder

  l_hip = length(p, PELVIS, L_HIP)
  r_hip = length(p, PELVIS, R_HIP)
  h = l_hip / r_hip

  """
  limbs
  """
  l_ua = length(p, L_SHOULDER, L_ELBOW)
  l_la = length(p, L_ELBOW, L_HAND)
  l_a = l_ua / l_la

  r_ua = length(p, R_SHOULDER, R_ELBOW)
  r_la = length(p, R_ELBOW, R_HAND)
  r_a = r_ua / r_la

  l_ul = length(p, L_HIP, L_KNEE)
  l_ll = length(p, L_KNEE, L_HAND)
  l_l = l_ul / l_ll

  r_ul = length(p, R_HIP, R_KNEE)
  r_ll = length(p, R_KNEE, R_HAND)
  r_l = r_ul / r_ll
  return(torch.tensor((h, s, l_a, r_a, l_l, r_l)))


def calculate_ratios(m):
  p = np.array(m)
  """
  trunk
  """
  l_shoulder = length(p, CHEST, L_SHOULDER)
  r_shoulder = length(p, CHEST, R_SHOULDER)
  s = l_shoulder / r_shoulder

  l_hip = length(p, PELVIS, L_HIP)
  r_hip = length(p, PELVIS, R_HIP)
  h = l_hip / r_hip

  """
  limbs
  """
  l_ua = length(p, L_SHOULDER, L_ELBOW)
  l_la = length(p, L_ELBOW, L_HAND)
  l_a = l_ua / l_la

  r_ua = length(p, R_SHOULDER, R_ELBOW)
  r_la = length(p, R_ELBOW, R_HAND)
  r_a = r_ua / r_la

  l_ul = length(p, L_HIP, L_KNEE)
  l_ll = length(p, L_KNEE, L_HAND)
  l_l = l_ul / l_ll

  r_ul = length(p, R_HIP, R_KNEE)
  r_ll = length(p, R_KNEE, R_HAND)
  r_l = r_ul / r_ll
  return(torch.tensor((h, s, l_a, r_a, l_l, r_l)))

def calculate_thetas_torch(p):
  if(not THETA_LOSS):
    return

  #print("CALCULATE_THETAS_TORCH", type(p), p.shape)

  """
  # root - re-add for PELVIS
  #meow = torch.stack((p[:,PELVIS][0]+1, p[:,PELVIS][1], p[:,PELVIS][2]))
  print(meow, p)
  p = scrub_data(p)
  print(p.shape)
  print(type(p), p.shape, p.tolist())
  """

  # center
  """
  data = calculate_joint_angle_torch(p, PELVIS)
  thetas.append(calculate_joint_angle_torch(p, PELVIS))
  """
  neck_rom, neck, necks = calculate_joint_angle_torch(p, NECK)
  spine_rom, spine, spines = calculate_joint_angle_torch(p, SPINE)

  # left limbs
  l_hip_rom, l_hip, l_hips = calculate_joint_angle_torch(p, L_HIP)
  l_shoulder_rom, l_shoulder, l_shoulders = calculate_joint_angle_torch(p, L_SHOULDER)
  l_knee_rom, l_knee, l_knees = calculate_joint_angle_torch(p, L_KNEE)
  l_elbow_rom, l_elbow, l_elbows = calculate_joint_angle_torch(p, L_ELBOW)

  # right limbs
  r_hip_rom, r_hip, r_hips = calculate_joint_angle_torch(p, R_HIP)
  r_shoulder_rom, r_shoulder, r_shoulders = calculate_joint_angle_torch(p, R_SHOULDER)
  r_knee_rom, r_knee, r_knees = calculate_joint_angle_torch(p, R_KNEE)
  r_elbow_rom, r_elbow, r_elbows = calculate_joint_angle_torch(p, R_ELBOW)

  """
  print("R_ELBOWS", r_elbows)
  print("ROMS", neck_rom, spine_rom, 
                l_hip_rom, l_shoulder_rom, l_knee_rom, l_elbow_rom, 
                r_hip_rom, r_shoulder_rom, r_knee_rom, r_elbow_rom)
  """

  """
  r_elbow_lim = r_elbow_limit_torch(p)
  l_elbow_lim = l_elbow_limit_torch(p)
  r_knee_lim = r_knee_limit_torch(p)
  l_knee_lim = l_knee_limit_torch(p)
  limits = torch.mean(torch.tensor((r_elbow_lim, l_elbow_lim, r_knee_lim, l_knee_lim)))
  """
  rom = torch.mean(torch.tensor((neck_rom, spine_rom, l_hip_rom, l_shoulder_rom, l_knee_rom, l_elbow_rom, r_hip_rom, r_shoulder_rom, r_knee_rom, r_elbow_rom)))

  thetas = torch.stack((necks, spines, l_hips, l_shoulders, l_knees, l_elbows, r_hips, r_shoulders, r_knees, r_elbows)).cuda()
  return rom, thetas

def calculate_thetas(m):
  if(not THETA_LOSS):
    return
  #global p
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow

  p = np.array(m)
  
  # root
  root = (p[PELVIS][0]+1, p[PELVIS][1], p[PELVIS][2])
  p = np.vstack((p, root))
  p = scrub_data(p)
  #print(p.shape)
  #print(type(p), p.shape, p.tolist())

  np.thetas = np.array([])

  # center
  thetas.append(calculate_joint_angle(p, PELVIS))
  thetas.append(calculate_joint_angle(p, NECK))
  thetas.append(calculate_joint_angle(p, SPINE))

  # left limbs
  thetas.append(calculate_joint_angle(p, L_HIP))
  thetas.append(calculate_joint_angle(p, L_SHOULDER))
  thetas.append(calculate_joint_angle(p, L_KNEE))
  thetas.append(calculate_joint_angle(p, L_ELBOW))

  # right limbs
  thetas.append(calculate_joint_angle(p, R_HIP))
  thetas.append(calculate_joint_angle(p, R_SHOULDER))
  thetas.append(calculate_joint_angle(p, R_KNEE))
  thetas.append(calculate_joint_angle(p, R_ELBOW))

  loss = 0
  loss += r_elbow_limit(p)
  loss += l_elbow_limit(p)
  loss += r_knee_limit(p)
  loss += l_knee_limit(p)

  if((valid+len(invalids)) % BATCH == 0 and valid > 0):
    """
    print("INVALID", len(invalids), 
      "/", (valid+len(invalids)), 
      100*len(invalids)/float(valid), "%")

    print("MAX lk, rk, le, re", max_invalid_dot_l_knee, 
          max_invalid_dot_r_knee, 
          max_invalid_dot_l_elbow, 
          max_invalid_dot_r_elbow)
    """

  #print("INVALID thetas", thetas)
  return thetas, loss
   
def r_knee_limit(p):
  global valid
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow
  c = np.array(p[PELVIS])
  s = np.array(p[R_HIP])
  e = np.array(p[R_KNEE])
  w = np.array(p[R_FOOT])

  v_sc  = s-c
  v_es  = e-s
  v_we  = w-e

  norm = np.array(np.cross(v_es, v_sc)).flatten()
  v_we = np.array(v_we).flatten()

  dot = np.dot(norm, v_we)

  if(dot >= 0):
    #print("INVALID R KNEE JOINT", "dot", dot, "points:",
           #c, s, e, w)

    invalids.append(p)
    if(dot > max_invalid_dot_l_knee):
      max_invalid__dot_r_knee = dot
    return np.norm(dot)
  valid += 1
  return 0

def r_knee_limit_torch(p):
  """
  global valid
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow
  """
  c = p[:, PELVIS]
  s = p[:, R_HIP]
  e = p[:, R_KNEE]
  w = p[:, R_FOOT]

  v_sc  = s-c
  v_es  = e-s
  v_we  = w-e

  #norm = np.array(np.cross(v_es, v_sc)).flatten()
  #v_we = np.array(v_we).flatten()
  norm = torch.cross(v_es, v_sc).view(-1)
  v_we = v_we.view(-1)

  dot = torch.dot(norm, v_we)

  if(dot >= 0):
    #print("INVALID R KNEE JOINT", "dot", dot, "points:",
           #c, s, e, w)

    """
    invalids.append(p)
    if(dot > max_invalid_dot_l_knee):
      max_invalid__dot_r_knee = dot
    """
    return torch.abs(dot)
  #valid += 1
  return torch.tensor(0).cuda()

def l_knee_limit(p):
  global valid
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow
  c = np.array(p[PELVIS])
  s = np.array(p[L_HIP])
  e = np.array(p[L_KNEE])
  w = np.array(p[L_FOOT])

  v_sc  = s-c
  v_es  = e-s
  v_we  = w-e

  norm = np.array(np.cross(v_es, v_sc)).flatten()
  v_we = np.array(v_we).flatten()

  dot = np.dot(norm, v_we)

  if(dot <= 0):
    #print("INVALID L KNEE JOINT", "dot", dot, "points:",
           #c, s, e, w)
    invalids.append(p)
    if(dot < max_invalid_dot_l_knee):
      max_invalid_dot_l_knee = dot
    return np.abs(dot)
  valid += 1
  return 0

def l_knee_limit_torch(p):
  """
  global valid
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow
  """

  c = p[:, PELVIS]
  s = p[:, L_HIP]
  e = p[:, L_KNEE]
  w = p[:, L_FOOT]

  v_sc  = s-c
  v_es  = e-s
  v_we  = w-e

  #norm = np.array(np.cross(v_es, v_sc)).flatten()
  #v_we = np.array(v_we).flatten()
  norm = torch.cross(v_es, v_sc).view(-1)
  v_we = v_we.view(-1)

  dot = torch.dot(norm, v_we)

  if(dot <= 0):
    #print("INVALID L KNEE JOINT", "dot", dot, "points:",
           #c, s, e, w)
    """
    invalids.append(p)
    if(dot < max_invalid_dot_l_knee):
      max_invalid_dot_l_knee = dot
    """
    return torch.abs(dot)
  #valid += 1
  return torch.tensor(0).cuda()

def r_elbow_limit_torch(p):
  """
  global valid
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow
  """
  c = p[:, CHEST]
  s = p[:, R_SHOULDER]
  e = p[:, R_ELBOW]
  w = p[:, R_HAND]

  v_sc  = s-c
  v_es  = e-s
  v_we  = w-e

  #norm = np.array(torch.cross(v_es, v_sc)).flatten()
  #v_we = np.array(v_we).flatten()
  norm = torch.cross(v_es, v_sc).view(-1)
  v_we = v_we.view(-1)

  dot = torch.dot(norm, v_we)

  if(dot <= 0):
    #print("INVALID R ELBOW JOINT", "dot", dot, "points:",
           #c.tolist(), s.tolist(), e.tolist(), w.tolist())
    """
    invalids.append(p)
    if(dot < max_invalid_dot_r_elbow):
      max_invalid_dot_l_knee = dot
    """
    return torch.abs(dot)

  #valid += 1
  return torch.tensor(float(0)).cuda()

def r_elbow_limit(p):
  global valid
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow
  c = np.array(p[CHEST])
  s = np.array(p[R_SHOULDER])
  e = np.array(p[R_ELBOW])
  w = np.array(p[R_HAND])

  v_sc  = s-c
  v_es  = e-s
  v_we  = w-e

  norm = np.array(np.cross(v_es, v_sc)).flatten()
  v_we = np.array(v_we).flatten()

  dot = np.dot(norm, v_we)
  if(dot <= 0):
    #print("INVALID R ELBOW JOINT", "dot", dot, "points:",
           #c.tolist(), s.tolist(), e.tolist(), w.tolist())
    invalids.append(p)
    if(dot < max_invalid_dot_r_elbow):
      max_invalid_dot_l_knee = dot
    return np.abs(dot)

  valid += 1
  return torch.tensor(0).cuda()

def r_elbow_limit_torch(p):
  """
  global valid
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow
  """
  c = p[:, CHEST]
  s = p[:, R_SHOULDER]
  e = p[:, R_ELBOW]
  w = p[:, R_HAND]

  v_sc  = s-c
  v_es  = e-s
  v_we  = w-e

  #norm = np.array(np.cross(v_es, v_sc)).flatten()
  #v_we = np.array(v_we).flatten()
  norm = torch.cross(v_es, v_sc).view(-1)
  v_we = v_we.view(-1)

  dot = torch.dot(norm, v_we)
  if(dot <= 0):
    #print("INVALID R ELBOW JOINT", "dot", dot, "points:",
           #c.tolist(), s.tolist(), e.tolist(), w.tolist())
    """
    invalids.append(p)
    if(dot < max_invalid_dot_r_elbow):
      max_invalid_dot_l_knee = dot
    """
    return torch.abs(dot)

  #valid += 1
  return torch.tensor(float(0)).cuda()

def l_elbow_limit(p):
  global valid
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow
  c = np.array(p[CHEST])
  s = np.array(p[L_SHOULDER])
  e = np.array(p[L_ELBOW])
  w = np.array(p[L_HAND])

  v_sc  = s-c
  v_es  = e-s
  v_we  = w-e

  norm = np.array(np.cross(v_es, v_sc)).flatten()
  v_we = np.array(v_we).flatten()

  dot = np.dot(norm, v_we)

  if(dot >= 0):
    #print("INVALID L ELBOW JOINT", "dot", dot, "points:",
           #c.tolist(), s.tolist(), e.tolist(), w.tolist())
    invalids.append(p)
    if(dot > max_invalid_dot_l_elbow):
      max_invalid_dot_l_knee
      return np.abs(dot)
  valid += 1
  return valid

def l_elbow_limit_torch(p):

  """
  global valid
  global max_invalid_dot_l_knee
  global max_invalid_dot_r_knee
  global max_invalid_dot_l_elbow
  global max_invalid_dot_r_elbow
  """
  c = p[:, CHEST]
  s = p[:, L_SHOULDER]
  e = p[:, L_ELBOW]
  w = p[:, L_HAND]

  v_sc  = s-c
  v_es  = e-s
  v_we  = w-e

  #norm = np.array(np.cross(v_es, v_sc)).flatten()
  #v_we = np.array(v_we).flatten()
  norm = torch.cross(v_es, v_sc).view(-1)
  v_we = v_we.view(-1)

  dot = torch.dot(norm, v_we)

  if(dot >= 0):
    #print("INVALID L ELBOW JOINT", "dot", dot, "points:",
           #c.tolist(), s.tolist(), e.tolist(), w.tolist())

    """
    invalids.append(p)
    if(dot > max_invalid_dot_l_elbow):
      max_invalid_dot_l_knee
    """
    return torch.abs(dot)
  #valid += 1
  return torch.tensor(float(0)).cuda()

def calculate_joint_angle_torch(p, j):
  theta = joint_angle_torch(p,
                      meta[j]["joints"][0],
                      meta[j]["joints"][1],
                      meta[j]["joints"][2],
                      meta[j]["joint"])

  #print("THETA", theta.shape, theta)

  """
  if(theta < meta[j]["min"] or theta > meta[j]["max"]):
    print(meta[j]["location"], meta[j]["joint"],
           "out of range", theta)
  """

  #meta[j]['theta'] = theta[0]

  """
  print("CALCULATE_JOINT_ANGLE_TORCH", 
        meta[j]["location"],
        meta[j]["joint"],
        theta,
        "min, max",
        meta[j]["min"],
        meta[j]["max"])
  """
  rom = torch.tensor(float(0)).cuda()
  rom += oob(theta, meta[j])

  #print("ROM", rom)

  mean = (torch.mean(torch.norm(theta - meta[j]["min"]))+
         torch.mean(torch.norm(theta - meta[j]["max"])))/2
  #print("MEAN LOSS", mean)

  return rom, mean, theta


def oob(theta, meta):
  mint = torch.tensor(0).float().cuda()
  mint = mint.new_full(theta.size(), float(meta["min"]))
  min_oob = torch.le(theta, meta["min"]).cuda()
  min_loss = torch.mean(torch.abs(min_oob.float() * (mint - theta)))

  #print("META", meta)
  #print("THETA", theta)
  #print("MIN", min_oob)
  #print("MIN LOSS", min_loss)

  maxt = torch.tensor(0).float().cuda()
  maxt = maxt.new_full(theta.size(), float(meta["max"]))
  max_oob = torch.gt(theta, meta["max"]).cuda()
  max_loss = torch.mean(torch.abs(max_oob.float() * (maxt - theta)))

  #print("MAX", max_oob)
  #print("MAX LOSS", max_loss)

  return torch.sum(min_loss + max_loss)

def kin_oob(kin, limit):
  maxt = torch.tensor(0).float().cuda()
  maxt = maxt.new_full(kin.size(), limit)
  max_oob = torch.gt(kin, limit).cuda()
  max_loss = torch.mean(torch.abs(max_oob.float() * (maxt - kin)))

  #print("KIN", max_oob)
  #print("KIN LOSS", max_loss)
  return max_loss

def calculate_joint_angle(p, j):
  theta = joint_angle(p,
                      meta[j]["joints"][0],
                      meta[j]["joints"][1],
                      meta[j]["joints"][2],
                      meta[j]["joint"])
  """
  if(theta < meta[j]["min"] or theta > meta[j]["max"]):
    print(meta[j]["location"], meta[j]["joint"], 
           "out of range", theta)
  """
  meta[j]['theta'] = theta[0]

  """
  print(meta[j]["location"], 
        meta[j]["joint"], 
        theta, 
        "min, max", 
        meta[j]["min"], 
        meta[j]["max"])
  """
  return theta[0]

def collisions():
  intersect(p[PELVIS], p[SPINE], p[R_SHOULDER], p[R_ELBOW])

def joint_angle_torch(p, j1, j2, j3, description):
  a = p[:,j1]
  b = p[:,j2]
  c = p[:,j3]
  #print("joint_angle_torch a", a)
  theta = vector_angle_torch(a, b, c)
  #print("joint_angle_torch", theta)
  #meta[j2]['theta'] = theta[0]
  #print(description, theta[0])
  return theta.cuda()

def joint_angle(p, j1, j2, j3, description):
  a = p[j1]
  b = p[j2]
  c = p[j3]
  theta = vector_angle(a, b, c)
  meta[j2]['theta'] = theta[0]
  #print(description, theta[0])
  return theta

def vector_angle(a, b, c):
  """
  #a = np.matrix(a)
  #b = np.matrix(b)
  #c = np.matrix(c)
  #print(a, b, c)
  deg = (vect_angle(b-a, c-a))
  print(np.radians(deg))
  #R = rotation_matrix(b-a, c-a)
  #print(euler(R))
  return deg
  """

  a = np.matrix(a)
  b = np.matrix(b)
  c = np.matrix(c)
  o = np.matrix([0,0,0])
  v1 = (b - o) - (a - o)
  v2 = (c - o) - (b - o)
  #uangle = unsigned_angle(v1.tolist()[0], v2.tolist()[0])
  a = vect_angle(v1, v2)
  prj = vg.scalar_projection(np.array(v2.tolist()[0]),np.array(v1.tolist()[0]))
  rj = vg.reject(np.array(v2.tolist()[0]),np.array(v1.tolist()[0]))
  #print("v1", v1, "v2", v2, "prj", prj, "rj", rj, "a", a)
  a = 180 - a
  return a

def vector_angle_torch(a, b, c):
  #a = np.matrix(a)
  #b = np.matrix(b)
  #c = np.matrix(c)
  #o = np.matrix([0,0,0])
  o = torch.zeros(b.shape).cuda()
  v1 = (b - o) - (a - o)
  v2 = (c - o) - (b - o)
  #print("vector_angle_torch", o, v1, v2)
  uangle = unsigned_angle(v1.tolist()[0], v2.tolist()[0])
  a = vect_angle(v1, v2)
  #print("vect_angle_torch", a)
  #prj = vg.scalar_projection(np.array(v2.tolist()[0]),np.array(v1.tolist()[0]))
  #rj = vg.reject(np.array(v2.tolist()[0]),np.array(v1.tolist()[0]))
  #print("v1", v1, "v2", v2, "prj", prj, "rj", rj, "a", a)
  a = 180 - a
  return a

def rotation_matrix(A,B):
  # a and b are in the form of numpy array
  ax = A[0]
  ay = A[1]
  az = A[2]

  bx = B[0]
  by = B[1]
  bz = B[2]

  au = A/(np.sqrt(ax*ax + ay*ay + az*az))
  bu = B/(np.sqrt(bx*bx + by*by + bz*bz))

  R = np.array([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]],
                [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]],
                [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]]])
  return R

def euler(R):
  """
  https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
  """

  theta_x = np.arctan2(R[2,1], R[2,2])
  theta_y = np.arctan2(-R[2,0],
                       np.sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2]))
  theta_z = np.arctan2(R[1,0], R[0,0])
  return((theta_x, theta_y, theta_z))


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def intersect(a1, a2, b1, b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = dot( dap, db)
    num = dot( dap, dp )
    return (num / denom.astype(float))*db + b1

"""
main
"""
def main(p):
  in_frame(p)
  pose(p)
"""
run
"""
#main()

# ==== DATASET PATHS
M40_PATH = "/home/ubuntu/Documents/nerf数据集/modelnet40_ply_hdf5_2048"
SON_PATH = "/home/ubuntu/Documents/nerf数据集/h5_files/main_split_nobg"
LM_PATH = "/home/ubuntu/Documents/nerf数据集/lm_base/lm"

BOP_PATH = "/home/ubuntu/Documents/nerf数据集/bop_toolkit"

# ==== REAGENT PARAMETERS
# iterations and replay buffer
BATCH_SIZE = 32
ITER_TRAIN, ITER_EVAL = 32,32
NUM_TRAJ = 1

# agent parameters
EXPERT_MODE = 'steady'
DISENTANGLED = True
STEPSIZES = [0.0033, 0.01, 0.03, 0.09, 0.27]  # trippling

# model parameters
BENCHMARK = False  # if True, compute target embedding only once - set False for training
IN_CHANNELS = 3
FEAT_DIM = 2048
STATE_DIM = FEAT_DIM
HEAD_DIM = 256
ACTION_DIM = 6
NUM_ACTIONS = 6  # 6 actions [+-x, +-y, +-z]
NUM_STEPSIZES = 5  # 5 step sizes
NUM_NOPS = 3  # for all actions

# RL parameters
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
CLIP_VALUE = False
C_VALUE, C_ENTROPY = 0.3, 1e-3

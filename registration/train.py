import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)
import torch.nn.functional as F
import os
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import argparse
from torch.utils.tensorboard import SummaryWriter   
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)).replace("/registration", ""))
from environment import environment as env
from environment import transformations as tra
from environment.buffer import Buffer
from registration.model import Agent
import registration.model as util_model
import utility.metrics as metrics
from utility.logger import Logger
from dataset.dataset import DatasetModelnet40, DatasetLinemod
import config as cfg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def CD_loss(pose_test, data, gamma = 0.8):
    n_pose = len(pose_test)
    cd_loss = 0.0
    gt_transforms = data['transform_gt'].to(DEVICE)  
    #?device
    
    igt_transforms = torch.eye(4, device=pose_test.device).repeat(gt_transforms.shape[0], 1, 1)
    igt_transforms[:, :3, :3] = gt_transforms[:, :3, :3].transpose(2, 1)
    igt_transforms[:, :3, 3] = -(igt_transforms[:, :3, :3] @ gt_transforms[:, :3, 3].view(-1, 3, 1)).view(-1, 3)
    points_src = data['points_src'][..., :3].to(DEVICE)
    points_ref = data['points_ref'][..., :3].to(DEVICE)
    if 'points_raw' in data:
        points_raw = data['points_raw'][..., :3].to(DEVICE)  
    else:
        points_raw = points_ref    

    for i in range(n_pose):

        
        i_weight = 1#gamma**(n_pose - i - 1)
        pred_test = pose_test[i].clone()

        src_transformed_test = (pred_test[:, :3, :3] @ points_src.transpose(2, 1)).transpose(2, 1)\
                + pred_test[:, :3, 3][:, None, :]

        ref_clean = points_raw
        residual_transforms = pred_test @ igt_transforms  #igt_transforms是

        src_clean_test = (residual_transforms[:, :3, :3] @ points_raw.transpose(2, 1)).transpose(2, 1)\
                    + residual_transforms[:, :3, 3][:, None, :]

        dist_src_1 = torch.min(tra.square_distance(src_transformed_test, ref_clean), dim=-1)[0]
        dist_ref_1 = torch.min(tra.square_distance(points_ref, src_clean_test), dim=-1)[0]

        batch_chamfer = torch.mean(dist_src_1, dim=1)  + torch.mean(dist_ref_1, dim=1)   
        i_loss = torch.mean(batch_chamfer)

        cd_loss = cd_loss+ i_weight*i_loss  
    return cd_loss
def train(agent, logger, dataset, noise_type, epochs, lr, lr_step, alpha, model_path, reward_mode=""):
    optimizer = torch.optim.AdamW(agent.parameters(), lr=0.001, weight_decay=0.0001, eps=1e-8)  
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.001, 16000+10,  
        pct_start=0.1, cycle_momentum=False, anneal_strategy='linear')
    Dataset = DatasetModelnet40 if dataset == "m40" else DatasetLinemod
    train_dataset = Dataset("train", noise_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_dataset = Dataset("val", noise_type)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_dataset = Dataset("test" if dataset == "m40" else "eval", noise_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    RANDOM_STATE = np.random.get_state()  # otherwise loader produces deterministic samples after iter 1

    best_chamfer = np.infty
    scaler = GradScaler(enabled=True)

    training = True
    total_steps = 0
    i = 0

    while training:
        print(f"Epoch {i}")
        i += 1

        # -- train
        agent.train()
        np.random.set_state(RANDOM_STATE)

        progress = tqdm(BackgroundGenerator(train_loader), total=len(train_loader))
        for data in progress:
            optimizer.zero_grad()
 
            source, target, pose_source, pose_target = env.init(data)

            if cfg.DISENTANGLED:
                pose_target = tra.to_disentangled(pose_target, source)

            _,_,clone_loss,_= agent(source, target,pose_source, pose_target, deter = False,iter = cfg.ITER_TRAIN,back = False)

            # -- update
            loss = clone_loss#+chamfer_dist_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()  

            
            total_steps += 1
            if total_steps > epochs:
                training = False
                break
            '''
            loss.backward()
            optimizer.step()

            episode += 1
            '''
        
        #scheduler.step()
        RANDOM_STATE = np.random.get_state()  # evaluation sets seeds again -- keep random state of the training stage

        # -- test
        if val_loader is not None:
            chamfer_val = evaluate(agent, logger, val_loader, prefix='val')

        if test_loader is not None:
            chamfer_test = evaluate(agent, logger, test_loader)


        if chamfer_test <= best_chamfer:
            print(f"new best: {chamfer_test}")
            best_chamfer = chamfer_test
            infos = {
                'epoch': i,
                'optimizer_state_dict': optimizer.state_dict()
            }
            util_model.save(agent, f"{model_path}.zip", infos)
        logger.dump(step=i)


def evaluate(agent, logger, loader, prefix='test'):
    agent.eval()
    progress = tqdm(BackgroundGenerator(loader), total=len(loader))
    predictions = []
    with torch.no_grad():
        for data in progress:
            source, target, pose_source, pose_target = env.init(data)
            if cfg.DISENTANGLED:
                pose_target = tra.to_disentangled(pose_target, source)

            pred, pose_source,val_loss,a = agent(source, target,pose_source, pose_target,deter = True, iter = cfg.ITER_EVAL, back = False )

            predictions.append(pose_source)

    predictions = torch.cat(predictions)
    _, summary_metrics = metrics.compute_stats(predictions, data_loader=loader)
    # log test metrics
    print(f"MAE R: {summary_metrics['r_mae']:0.2f}")
    print(f"MAE t: {summary_metrics['t_mae']:0.3f}")
    print(f"ISO R: {summary_metrics['r_iso']:0.2f}")
    print(f"ISO t: {summary_metrics['t_iso']:0.3f}")
    print(f"ADI AUC: {(summary_metrics['adi_auc10'] * 100):0.1f}%")
    print(f"CD: {summary_metrics['chamfer_dist'] * 1000:0.2f}")
    # log test metrics
    if False:
        logger.record(f"{prefix}/add", summary_metrics['add'])
        logger.record(f"{prefix}/adi", summary_metrics['adi'])
        return summary_metrics['add']
    else:
        logger.record(f"{prefix}/mae-r", summary_metrics['r_mae'])
        logger.record(f"{prefix}/mae-t", summary_metrics['t_mae'])
        logger.record(f"{prefix}/iso-r", summary_metrics['r_iso'])
        logger.record(f"{prefix}/iso-t", summary_metrics['t_iso'])
        logger.record(f"{prefix}/chamfer", summary_metrics['chamfer_dist'])
        logger.record(f"{prefix}/adi-auc", summary_metrics['adi_auc10'] * 100)
        return summary_metrics['r_mae']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReAgent - training on ModelNet40 and LINEMOD')
    parser.add_argument('--mode', type=str, default='il', choices=['pretrain', 'il', 'ilrl'],
                        help='pretraining (pretrain), IL-only (il), IL+RL with a step-wise reward (ilrls).')
    parser.add_argument('--dataset', type=str, default='m40', choices=['m40', 'lm'],
                        help='Dataset used for training. All experiments on ModelNet40 and ScanObjectNN use the same '
                             'weights - train both with "m40". Experiments on LINEMOD ("lm") use no pretraining.')
    args = parser.parse_args()

    # PATHS
    dataset = args.dataset
    mode = args.mode
    code_path = os.path.dirname(os.path.abspath(__file__)).replace("/registration", "")
    if not os.path.exists(os.path.join(code_path, "logs")):
        os.mkdir(os.path.join(code_path, "logs"))
    if not os.path.exists(os.path.join(code_path, "weights")):
        os.mkdir(os.path.join(code_path, "weights"))
    model_path = os.path.join(code_path, f"weights/{dataset}_{mode}")
    logger = Logger(log_dir=os.path.join(code_path, f"logs/{dataset}/"), log_name=f"{mode}",
                    reset_num_timesteps=True)

    # TRAINING
    agent = Agent().to(DEVICE)

    if args.mode == "pretrain" and dataset == "m40":
        print(f"Training: dataset '{dataset}'  - mode '{args.mode}'")
        train(agent, logger, dataset, noise_type="clean", epochs=50, lr=1e-3, lr_step=10, alpha=0,
              model_path=model_path)
    else:
        if args.mode == "il":
            alpha = 0.0
            reward_mode = ""
        elif args.mode == "ilrl":
            alpha = 2.0 if dataset == "m40" else 0.1  # reduced influence on lm
            reward_mode = "step"
        else:
            raise ValueError("No pretraining on LINEMOD. Use 'il' or 'ilrl' instead.")
        print(f"Training: dataset '{dataset}' - mode '{args.mode}'{f' - alpha={alpha}' if args.mode != 'il' else ''}")

        if dataset == "m40":
            print("  loading pretrained weights...")
            if os.path.exists(os.path.join(code_path, f"weights/m40_pretrain.zip")):
                util_model.load(agent, os.path.join(code_path, f"weights/m40_pretrain.zip"))
            else:
                raise FileNotFoundError(f"No pretrained weights found at "
                                        f"{os.path.join(code_path, f'weights/m40_pretrain.zip')}. Run with "
                                        f"'pretrain' first or download the provided weights.")

        noise_type = "jitter" if dataset == "m40" else "segmentation"
        epochs = 50 if dataset == "m40" else 100
        lr = 1e-4 if dataset == "m40" else 1e-3
        lr_step = 10 if dataset == "m40" else 20

        train(agent, logger, dataset, noise_type, epochs=epochs, lr=lr, lr_step=lr_step,
              alpha=alpha, reward_mode=reward_mode, model_path=model_path)


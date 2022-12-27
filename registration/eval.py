import torch, time
import os
import argparse
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)).replace("/registration", ""))
import config as cfg
cfg.BENCHMARK = True
from environment import environment as env
from environment import transformations as tra
from registration.model import Agent
import registration.model as util_model
import utility.metrics as metrics
from utility.visualization import CloudVisualizer, OutlineVisualizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(agent, test_loader, dataset_name, bop_results_path="", visualize=False):
    if visualize:
        if dataset_name != "lm":
            vis = CloudVisualizer(0.01)
        else:
            vis = OutlineVisualizer("lm", list(range(1, 16)), 0.01)
            obj_idx = 0

    agent.eval()
    progress = tqdm(BackgroundGenerator(test_loader), total=len(test_loader))
    predictions = []
    if bop_results_path != "" and os.path.exists(bop_results_path):
        open(bop_results_path, 'w').write("")  # BOP toolkit expects file to contain results of a single evaluation run

    rcd_times, n_cnt = 0, 0
    with torch.no_grad():
        i = 0
        index = -3
        for data in progress:
            source, target, pose_source, pose_target = env.init(data)
            cc =  tra.apply_trafo(source, pose_target, False)
            if cfg.DISENTANGLED:
                pose_target = tra.to_disentangled(pose_target, source)
            current_source = source
            if visualize:
                if dataset_name == "lm":
                    rgb = test_loader.dataset.get_rgb(int(data['scene'][obj_idx]), int(data['frame'][obj_idx]))
                    vis.reset(i,data, rgb,   split="test")
                else:
                    i += 1
                    vis.reset(i,source[index, :, :3].cpu().numpy(), target[index, :, :3].cpu().numpy()
                               ,current_source[index, :, :3].cpu().numpy())

            t1 = time.time()
            _, pose_source,_, current_source = agent(source, target,pose_source, pose_target,deter = True, iter = cfg.ITER_EVAL, back = False )
            rcd_times += time.time() - t1
            n_cnt += len(source)
            print("time:",rcd_times / n_cnt, n_cnt)
            if visualize:
                if dataset_name == "lm":
                    vis.update(i,tra.to_global(pose_source.clone(), source).cpu())
                else:
                    vis.update(i,current_source[index, :, :3].cpu().numpy())
                    

            if dataset_name == "lm":
                # undo normalization
                predictions_unnorm = pose_source.clone().cpu()
                predictions_unnorm[:, :3, 3] *= data['normalization'][:, 0, 0][:, None]
                # apply refinement to initial estimate to get the full model-to-camera estimation
                #   note: prediction is from initial pose to model space
                init_i2c = data['est']
                prediction_m2i = torch.eye(4, device="cpu").repeat(pose_source.shape[0], 1, 1)
                prediction_m2i[:, :3, :3] = predictions_unnorm[:, :3, :3].transpose(2, 1)
                prediction_m2i[:, :3, 3] = -(prediction_m2i[:, :3, :3] @ predictions_unnorm[:, :3, 3].view(-1, 3, 1))\
                    .squeeze()
                estimates_m2c = init_i2c @ prediction_m2i
                # save in BOP format
                estimates_bop = ""
                for i_est, estimate in enumerate(estimates_m2c):
                    scene_id, im_id, obj_id = data['scene'][i_est], data['frame'][i_est],\
                                              data['gt']['obj_id'][i_est]
                    conf, duration = 1.0, 0.0  # no confidence estimated, duration would need to be for single frame
                    estimates_bop += f"{scene_id},{im_id},{obj_id},{conf:0.3f}," \
                                     f"{' '.join([f'{float(v):0.6f}' for v in estimate[:3, :3].reshape(-1)])}," \
                                     f"{' '.join([f'{float(v):0.6f}' for v in estimate[:3, 3].reshape(-1)])}," \
                                     f"{duration:0.3f}\n"
                with open(bop_results_path, 'a') as file:
                    file.write(estimates_bop)
            else:
                predictions.append(pose_source)

    # COMPUTE STATISTICS
    if dataset_name == "lm":
        print(f"Stored predictions in BOP format to {bop_results_path}.")
    else:
        predictions = torch.cat(predictions)
        eval_metrics, summary_metrics = metrics.compute_stats(predictions, data_loader=test_loader)

        # log test metrics
        print(f"MAE R: {summary_metrics['r_mae']:0.2f}")
        print(f"MAE t: {summary_metrics['t_mae']:0.3f}")
        print(f"ISO R: {summary_metrics['r_iso']:0.2f}")
        print(f"ISO t: {summary_metrics['t_iso']:0.3f}")
        print(f"ADI AUC: {(summary_metrics['adi_auc10'] * 100):0.1f}%")
        print(f"CD: {summary_metrics['chamfer_dist'] * 1000:0.2f}")
        return summary_metrics['chamfer_dist']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReAgent - evaluation on ModelNet40, ScanObjectNN and LINEMOD')
    parser.add_argument('--mode', type=str, default='ilrl', choices=['il', 'ilrl','pretrain'],
                        help='IL-only (il), IL+RL with a step-wise reward (ilrl)')
    parser.add_argument('--dataset', type=str, default='m40-model', choices=['m40-model', 'm40-cat', 'son', 'lm'],
                        help='Dataset used for evaluation. For ModelNet40, either the held-out models ("m40-model") '
                             'or the held-out categories ("m40-cat") can be selected.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size used for evaluation. Timings are computed with batch size 1. Visualization '
                             'shows the first sample in each batch.')
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help='If selected, shows a visualization of the registration process.')
    args = parser.parse_args()

    code_path = os.path.dirname(os.path.abspath(__file__)).replace("/registration", "")
    if args.dataset.startswith("m40"):
        from dataset.dataset import DatasetModelnet40
        test_dataset = DatasetModelnet40("test" if "cat" in args.dataset else "val", "jitter")
        pretrain = os.path.join(code_path, f"weights/m40_{args.mode}.zip")  # same weights for M40 and SON
        bop_results_path = ""
    elif args.dataset == "son":
        from dataset.dataset import DatasetScanObjectNN
        test_dataset = DatasetScanObjectNN("test", "sensor")
        pretrain = os.path.join(code_path, f"weights/m40_{args.mode}.zip")  # same weights for M40 and SON
        bop_results_path = ""
    else:
        from dataset.dataset import DatasetLinemod
        test_dataset = DatasetLinemod("test", "segmentation")
        pretrain = os.path.join(code_path, f"weights/lm_{args.mode}.zip")
        if not os.path.exists(os.path.join(code_path, f"results")):
            os.mkdir(os.path.join(code_path, f"results"))
        bop_results_path = os.path.join(code_path, f"results/reagent-{args.mode}_{args.dataset}-test.csv")
    # note: test is deterministic (sets seed to dataset sample's index)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("  loading weights...")
    agent = Agent().to(DEVICE)
    if os.path.exists(pretrain):
        util_model.load(agent, pretrain)
    else:
        raise FileNotFoundError(f"No weights found at {pretrain}. Download pretrained weights or run training first.")

    evaluate(agent, test_loader, args.dataset, bop_results_path, args.visualize)

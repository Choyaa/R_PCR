
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from environment import environment as env
from config import *
from environment import transformations as tra
from environment.buffer import Buffer
from scipy.spatial.transform import Rotation
import mayavi.mlab as mlab
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Agent(nn.Module):

    def __init__(self):
        super().__init__()

        self.state_emb = StateEmbed()
        self.h_initial_r = StateEmbed_h()
        self.h_initial_t = StateEmbed_h()
        self.actor_critic = ActorCriticHead()
        self.gru_r = Gru()
        self.gru_t = Gru()

        self.state_r = Basic_encoder()
        self.state_t = Basic_encoder()
    def forward(self, src, tgt, pose_source, pose_target,  deter = False, iter = ITER_EVAL, back = False):
        predictions = []
        clone_loss = 0.0

        gamma = 0.8
        pose_global = []
        h_r = self.h_initial_r(src, tgt)
        h_t = self.h_initial_r(src, tgt)
        h_r = torch.tanh(h_r)
        h_t = torch.tanh(h_t)

        new_source = src 

        euler_r = tra.matrix_to_euler_angles(pose_source[:, :3, :3], 'XYZ')
        euler_t = pose_source[:, :3, 3]
        euler_r = torch.unsqueeze(euler_r, -1)
        euler_t = torch.unsqueeze(euler_t, -1)
        for step in range(iter):
            i_weight = gamma**(iter - step - 1)
            
            expert_action = env.expert(pose_source, pose_target, mode=EXPERT_MODE)
            #state embedding
            state= self.state_emb(new_source.detach().clone(), tgt)
            #motion encoder
            gru_inp_r = self.state_r(state, euler_r)
            gru_inp_t = self.state_t(state, euler_t)
            h_r = self.gru_r(h_r, gru_inp_r)
            h_t = self.gru_t(h_t, gru_inp_t)

            h_r1 = h_r.view(h_r.shape[0], -1)
            h_t1 = h_t.view(h_t.shape[0], -1)
            action = self.actor_critic(h_r1,h_t1)

            # reshape a to B x axis x [step, sign]
            action = (action[0].view(-1, 3, 2 * NUM_STEPSIZES + 1),
                action[1].view(-1, 3, 2 * NUM_STEPSIZES + 1))

            action_res = action_from_logits(action, deterministic=deter)

            new_source, pose_source = env.step(src, action_res, pose_source, DISENTANGLED)  
            #############
            euler_r = tra.matrix_to_euler_angles(pose_source[:, :3, :3], 'XYZ')
            euler_t = pose_source[:, :3, 3]
            euler_r = torch.unsqueeze(euler_r, -1).detach().clone()
            euler_t = torch.unsqueeze(euler_t, -1).detach().clone()

            ###############
            # to global
            pose_global = to_global(pose_source, src) 
            predictions.append(pose_global)
            #######

            loss_translation = F.cross_entropy(action[0].view(-1, 11, 1, 1, 1),
                                                expert_action[:, 0].reshape(-1, 1, 1, 1))
            loss_rotation = F.cross_entropy(action[1].view(-1, 11, 1, 1, 1),
                                            expert_action[:, 1].reshape(-1, 1, 1, 1))
            iter_loss1 = (loss_translation + loss_rotation ) / 2

            clone_loss = clone_loss+ i_weight* iter_loss1
        clone_loss = clone_loss.mean()           
        predictions= torch.stack(predictions)

        return   predictions, pose_global, clone_loss,new_source
def mlab_pointcloud(source ,target):
    s = source.cpu().numpy()
    t = target.cpu().numpy()
    x1 = s[0,:,0]
    y1 = s[0,:,1]
    z1 = s[0,:,2]
    x2 = t[0,:,0]
    y2 = t[0,:,1]
    z2 = t[0,:,2]
    s1 = mlab.points3d(x1, y1, z1, color = (0.22,0.42,0.77))
    t1 = mlab.points3d(x2, y2, z2,color =(0.97,0.55,0.078))
    mlab.show()
def to_global(posess, pcd):
    """
    Remove rotation-induced translation from translation vector - see eq. 11 in paper.
    """
    poses = posess.clone()
    poses[:, :3, 3] = poses[:, :3, 3] + pcd[..., :3].mean(dim=1) \
                      - (poses[:, :3, :3] @ pcd[..., :3].mean(dim=1)[:, :, None]).view(-1, 3)
    return poses
class Initial_h(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(IN_CHANNELS, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)

    def forward(self, src, tgt):
        B, N, D = src.shape

        # O=(src,tgt) -> S=[Phi(src), Phi(tgt)]
        h_src = self.embed(src.transpose(2, 1))
        if BENCHMARK and len(tgt.shape) != 3:
            emb_tgt = tgt  # re-use target embedding from first step
        else:
            h_tgt = self.embed(tgt.transpose(2, 1))

        return h_src, h_tgt

    def embed(self, x):
        B, D, N = x.shape

        # embedding: BxDxN -> BxFxN
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)

        return x3


class Gru(nn.Module):

    def __init__(self):
        super().__init__()
        self.gru_dim = 512
        self.convz = nn.Conv1d(self.gru_dim*2, self.gru_dim, 1, groups = 2)
        self.convr = nn.Conv1d(self.gru_dim*2, self.gru_dim, 1, groups = 2)
        self.convq = nn.Conv1d(self.gru_dim*2, self.gru_dim, 1, groups = 2)  #设置groups，groups=4，参数减少4倍


    def forward(self, h, s):
        hs = torch.cat([h, s], dim = 1)
        z = torch.sigmoid(self.convz(hs))
        r = torch.sigmoid(self.convr(hs))
        q = torch.tanh(self.convq(torch.cat([r*h, s], dim = 1)))
        h = (1-z) * h + z * q

        return h

class Basic_encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.convs = nn.Conv1d(1, 1, 16,stride = 1)#self.convs = nn.Conv1d(STATE_DIM, HEAD_DIM*2, 16)
        ##input (N, 1,2048)，output(N, 1, 512)（N，cin， cout）
        self.convrt1 = nn.Conv1d(3, 256, 1) #conv1d（cin， cout， k）， Lout取决于padding和delratiion
        self.conv = nn.Conv1d(256+2033, 512, 1)


    def forward(self, s, rt):  
        s = F.relu(self.convs(s))  #dimension reduction
        s = s.transpose(1,2)

        rt1 = F.relu(self.convrt1(rt))  

        s_rt = torch.cat([s, rt1], dim = 1)  #batch*(512+256)
        out = F.relu(self.conv(s_rt))  

        return out  #batch*512




class StateEmbed(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(IN_CHANNELS, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

    def forward(self, src, tgt):
        B, N, D = src.shape

        # O=(src,tgt) -> S=[Phi(src), Phi(tgt)]
        emb_src = self.embed(src.transpose(2, 1))
        if BENCHMARK and len(tgt.shape) != 3:
            emb_tgt = tgt  # re-use target embedding from first step
        else:
            emb_tgt = self.embed(tgt.transpose(2, 1))
        # state = torch.cat((emb_src, emb_tgt), dim=1)
        # state = state
        state = torch.zeros(emb_src.shape[0], emb_src.shape[1]*2,1).cuda()
        state[:, 0: emb_src.shape[1]*2+1:2] = emb_src
        state[:, 1: emb_src.shape[1]*2+1:2] = emb_tgt  #test1 交叉复制
        state = state.transpose(1,2)

        return state

    def embed(self, x):
        B, D, N = x.shape

        # embedding: BxDxN -> BxFxN
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)

        # pooling: BxFxN -> BxFx1
        x_pooled = torch.max(x3, 2, keepdim=True)[0]
        return x_pooled

class StateEmbed_h(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(IN_CHANNELS, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

    def forward(self, src, tgt):
        B, N, D = src.shape

        # O=(src,tgt) -> S=[Phi(src), Phi(tgt)]
        emb_src = self.embed(src.transpose(2, 1))
        if BENCHMARK and len(tgt.shape) != 3:
            emb_tgt = tgt  # re-use target embedding from first step
        else:
            emb_tgt = self.embed(tgt.transpose(2, 1))
        state = torch.cat((emb_src, emb_tgt), dim=1)
        state = state

        return state

    def embed(self, x):
        B, D, N = x.shape

        # embedding: BxDxN -> BxFxN
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)

        # pooling: BxFxN -> BxFx1
        x_pooled = torch.max(x3, 2, keepdim=True)[0]
        return x_pooled



class ActorCriticHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.dim = 512
        self.activation = nn.ReLU()

        self.emb_r = nn.Sequential(
            nn.Linear(self.dim, HEAD_DIM*2),
            self.activation,
            nn.Linear(HEAD_DIM*2, HEAD_DIM),
            self.activation
        )
        self.action_r = nn.Linear(HEAD_DIM, NUM_ACTIONS * NUM_STEPSIZES + NUM_NOPS)

        self.emb_t = nn.Sequential(
            nn.Linear(self.dim, HEAD_DIM*2),
            self.activation,
            nn.Linear(HEAD_DIM*2, HEAD_DIM),
            self.activation
        )
        self.action_t = nn.Linear(HEAD_DIM, NUM_ACTIONS * NUM_STEPSIZES + NUM_NOPS)


    def forward(self, state_r, state_t):
        # S -> S'
        emb_t = self.emb_t(state_t)
        emb_r = self.emb_r(state_r)
        # S' -> pi
        action_logits_t = self.action_t(emb_t)
        action_logits_r = self.action_r(emb_r)



        return [action_logits_t, action_logits_r]


# -- action helpers
def action_from_logits(logits, deterministic=True):
    #logits = logitss.clone()
    distributions = _get_distributions(*logits)
    actions = _get_actions(*(distributions + (deterministic,)))

    return torch.stack(actions).transpose(1, 0)


def action_stats(logits, action):
    distributions = _get_distributions(*logits)
    logprobs, entropies = _get_logprob_entropy(*(distributions + (action[:, 0], action[:, 1])))

    return torch.stack(logprobs).transpose(1, 0), torch.stack(entropies).transpose(1, 0)


def _get_distributions(action_logits_t, action_logits_r):
    #action_logits_t = action_logits_tt.clone()
    #action_logits_r = action_logits_rr.clone()
    distribution_t = Categorical(logits=action_logits_t)
    distribution_r = Categorical(logits=action_logits_r)

    return distribution_t, distribution_r


def _get_actions(distribution_t, distribution_r, deterministic=True):
    if deterministic:
        action_t = torch.argmax(distribution_t.probs, dim=-1)
        action_r = torch.argmax(distribution_r.probs, dim=-1)
    else:
        action_t = distribution_t.sample()
        action_r = distribution_r.sample()
    return action_t, action_r


def _get_logprob_entropy(distribution_t, distribution_r, action_t, action_r):
    logprob_t = distribution_t.log_prob(action_t)
    logprob_r = distribution_r.log_prob(action_r)

    entropy_t = distribution_t.entropy()
    entropy_r = distribution_r.entropy()

    return [logprob_t, logprob_r], [entropy_t, entropy_r]


# --- model helpers
def load(model, path):
    infos = torch.load(path)
    model.load_state_dict(infos['model_state_dict'])
    return infos


def save(model, path, infos={}):
    infos['model_state_dict'] = model.state_dict()
    torch.save(infos, path)


def plot_grad_flow(model):
    """
    via https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                print(f"no grad for {n}")
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, -1, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=-1, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=torch.max(torch.stack(max_grads)).cpu())
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

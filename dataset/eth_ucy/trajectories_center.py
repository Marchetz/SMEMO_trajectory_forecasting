import logging
import os
import math
import random

import numpy as np

import torch
from torch.utils.data import Dataset
#import cv2
import pdb

seed = 1000
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#TODO: togliere quelli più lontani


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            #line = line.strip().split(delim)
            line = line.strip().split('\t')

            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def rotate_pc(pc, alpha):
    M = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])
    return M @ pc

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, is_train, test_name, data_dir, data_dir_val=None, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.test_name = test_name
        self.is_train = is_train
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        means = []
        stds = []


        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        if data_dir_val is not None:
            val_files = os.listdir(data_dir_val)
            val_files = [os.path.join(data_dir_val, _path) for _path in val_files]
            all_files += val_files


        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        for path in all_files:
            data = read_file(path, delim)

            # after
            # data[:, 2] = data[:, 2] - data[:, 2].mean()
            # data[:, 3] = data[:, 3] - data[:, 3].mean()

            # if (path == 'dataset/eth_ucy/datasets/' + test_name + '/train/biwi_hotel_train.txt')\
            #         or (path == 'dataset/eth_ucy/datasets/' + test_name + '/test/biwi_hotel.txt'):
            #     data[:, [2, 3]] = data[:, [3, 2]]
            #     data[:, 2] = data[:, 2] + 10.0
            #     data[:, 3] = data[:, 3] + 4.0
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            means.append(data[:, 2:4].mean(0))
            stds.append(data[:, 2:4].std(0))

            #before
            # data[2] = data[2] - data[2].mean()
            # data[3] = data[3] - data[3].mean()

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq

                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq

                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:

                    for i in range(num_peds_considered):
                        temp = np.arange(num_peds_considered).tolist()
                        order = [i]
                        temp.remove(i)
                        order.extend(temp)
                        seq_list.append(curr_seq[:num_peds_considered][order])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered][order])
                        non_linear_ped += np.array(_non_linear_ped)[order].tolist()
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered][order])


        self.mean = torch.Tensor(means).mean(0)
        self.std = torch.Tensor(stds).mean(0)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.pasts = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.futures = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.pasts_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.futures_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def shift(self, xy, center):
        # theta = random.random() * 2.0 * math.pi
        xy = xy - center[np.newaxis, np.newaxis, :]
        return xy

    def theta_rotation(self, xy, theta):
        # theta = random.random() * 2.0 * math.pi
        ct = math.cos(theta)
        st = math.sin(theta)

        r = np.array([[ct, st], [-st, ct]])
        return np.einsum('ptc,ci->pti', xy, r)

    def center_scene(self, xy, obs_length=9, ped_id=0, goals=None):
        ## Center
        xy = xy.permute(1,0,2)
        # center = xy[obs_length - 1, ped_id]  ## Last Observation
        # xy = shift(xy, center)

        ## Rotate
        last_obs = xy[obs_length-1, ped_id]
        second_last_obs = xy[obs_length-2, ped_id]
        diff = np.array([last_obs[0] - second_last_obs[0], last_obs[1] - second_last_obs[1]])
        thet = np.arctan2(diff[1], diff[0])
        rotation = -thet + np.pi/2
        xy = self.theta_rotation(xy, rotation)

        return torch.Tensor(xy).permute(1,0,2), rotation


    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        past = self.pasts[start:end].permute(0, 2, 1)
        future = self.futures[start:end].permute(0, 2, 1)
        track = torch.cat((past, future), 1)

        #permutation agents
        if self.is_train:
            ids = torch.cat((torch.Tensor([0]).int(), torch.arange(past.shape[0])[1:][torch.randperm(past.shape[0]-1)]),0)
            track = track[ids]

        present = track[0, 7]
        track = track - present
        idx_near = [True]
        idx_near.extend((torch.norm(track[:1] - track[1:], dim=2) < 6.0).any(1).tolist())
        track = track[idx_near]

        #norm: rotate scene so primary pedestrian moves northwards at end of observation
        #track, _ = self.center_scene(track, 8)

        # #rotation aug new
        track_rot = []
        angles = np.arange(0, 360, 15)
        #angle = random.choice(angles)
        angle = random.uniform(0, 360)
        alpha = angle * np.pi / 180

        if self.is_train:
            for i in range(track.shape[0]):
                track_rot.append(rotate_pc(track[i].numpy().transpose(1, 0), alpha).transpose(1, 0))
            track_rot = torch.Tensor(track_rot)
        else:
            track_rot = track

        past = track_rot[:, :8]
        future = track_rot[:, 8:]
        present = track_rot[:, 7:]

        length = past.shape[0]
        out = [index, past, future, present, length]
        return out

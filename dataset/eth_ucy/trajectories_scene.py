import logging
import os
import math
import random

import numpy as np

import torch
from torch.utils.data import Dataset
import cv2
import pdb

seed = 1000
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

logger = logging.getLogger(__name__)


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

#come ruota questo??
def rotate_pc(pc, alpha):
    M = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])
    return M @ pc

def world2image(traj_w, H_inv):
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)
    # to pixel coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2])
    return traj_uvz[:, :2].astype(int)

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

        self.is_train = is_train
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        self.scenes_total = {}
        self.Hom = {}
        self.size_scene = {}
        self.scenes_name = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
        self.dict_scenes = {'biwi_eth': 'eth', 'biwi_hotel': 'hotel', 'students001': 'univ', 'students003': 'univ',
                            'crowds_zara01': 'zara1', 'crowds_zara02': 'zara2',
                            'crowds_zara03': 'zara1', 'uni_examples': 'univ'}
        self.scost_total = {'eth': torch.Tensor([100.0, 70.0]),
                       'hotel': torch.Tensor([40.0, 70.0]),
                       'univ': torch.Tensor([0.0, 0.0]),
                       'zara1': torch.Tensor([0.0, 0.0]),
                       'zara2': torch.Tensor([0.0, 50.0])}
        self.angle_total = {'eth': -80,
                       'hotel': -80,
                       'univ': 0.0,
                       'zara1': 0.0,
                       'zara2': 0.0}


        self.pad_image = 200
        self.scost_pad = 200
        for name in self.scenes_name:
            scene = cv2.imread('dataset/eth_ucy/datasets_frames/' + name + '/annotated.png', 0)
            c_h = int(scene.shape[0] / 2)
            c_w = int(scene.shape[1] / 2)
            self.size_scene[name] = [c_w, c_h]
            scene = scene > 180
            scene = torch.Tensor(scene).int()
            temp_pad = np.pad(scene, ((self.pad_image, self.pad_image), (self.pad_image, self.pad_image)), 'edge')
            self.scenes_total[name] = temp_pad

            folder_data = 'dataset/eth_ucy/datasets_frames/' + name + '/'
            homography_file = folder_data + 'H.txt'
            H = np.loadtxt(os.path.join(homography_file))
            H_inv = np.linalg.inv(H)
            self.Hom[name] = H_inv
            c_h = int(scene.shape[0] / 2)
            c_w = int(scene.shape[1] / 2)

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        if data_dir_val is not None:
            val_files = os.listdir(data_dir_val)
            val_files = [os.path.join(data_dir_val, _path) for _path in val_files]
            all_files += val_files


        num_peds_in_seq = []
        seq_list = []
        seq_list_scene = []
        seq_list_rel = []
        seq_list_time = []
        loss_mask_list = []
        non_linear_ped = []

        for path in all_files:
            data = read_file(path, delim)
            if is_train:
                scene_name = path.split('/')[-1][:-10]
                scene_tag = self.dict_scenes[scene_name]
            else:
                scene_name = path.split('/')[-1][:-4]
                scene_tag = self.dict_scenes[scene_name]

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq_time = np.zeros((len(peds_in_curr_seq), self.seq_len))

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
                    timing = curr_ped_seq[:,0]
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq

                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_time[_idx, :] = timing
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq

                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_scene.append(scene_tag)
                    seq_list_time.append(curr_seq_time[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])


        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_list_time = np.concatenate(seq_list_time, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.pasts = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.scenes = seq_list_scene
        self.futures = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.pasts_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.futures_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.timing = torch.from_numpy(seq_list_time)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq


    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        past = self.pasts[start:end].permute(0, 2, 1)
        future = self.futures[start:end].permute(0, 2, 1)
        time = self.timing[start:end]

        scene_name = self.scenes[index]
        scene_map = self.scenes_total[scene_name]
        H_inv = self.Hom[scene_name]
        c_w, c_h = self.size_scene[scene_name]

        if self.is_train:
            ids = torch.randperm(past.shape[0])
            past = past[ids]
            future = future[ids]
            time = time[ids]

        if self.is_train:
            if random.getrandbits(1):
                temp = torch.cat((past, future), 1).flip(1)
                past = temp[:, :self.obs_len]
                future = temp[:, self.obs_len:]

        tracks = torch.cat((past, future), 1)

        scost = self.scost_total[scene_name]
        angle = self.angle_total[scene_name]
        matRot_track = cv2.getRotationMatrix2D((c_w, c_h), angle, 1)
        tracks_new = []
        clip = 100
        scene_img = torch.Tensor()
        for t in tracks:
            t = torch.Tensor(world2image(t, H_inv))
            t = torch.Tensor(cv2.transform(t.reshape(-1, 1, 2).numpy(), matRot_track).squeeze())
            t = t + scost + self.scost_pad
            pres = t[7].int()
            img_new = scene_map[pres[1] - clip:pres[1] + clip, pres[0] - clip:pres[0] + clip]
            scene_img = torch.cat((scene_img, torch.Tensor(img_new).unsqueeze(0)), 0)

        scene_img_onehot = np.eye(2, dtype=np.float32)[scene_img.int()]

        present = past[0, 7]
        # #rotation aug new
        past_rot = []
        future_rot = []
        angle = random.uniform(0, 360)
        alpha = angle * np.pi / 180
        #TODO
        if self.is_train:
            for i in range(past.shape[0]):

                past_rot.append(rotate_pc(past[i].numpy().transpose(1, 0), alpha).transpose(1, 0))
                future_rot.append(rotate_pc(future[i].numpy().transpose(1, 0), alpha).transpose(1, 0))
            past = torch.Tensor(past_rot)
            future = torch.Tensor(future_rot)

        length = past.shape[0]
        out = [index, past, future, present, length, time, scene_img_onehot]
        return out

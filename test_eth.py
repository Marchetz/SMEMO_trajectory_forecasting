#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

#modello: model_rel_gru

import os
import io
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import random
import sys
import datetime
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")
from dataset.eth_ucy.trajectories import TrajectoryDataset
RANDOM_SEED = 1000


np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=3)
np.set_printoptions(threshold=sys.maxsize)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

class tester:
    def __init__(self, args):

        # Configuration
        self.args = args
        self.model = torch.load('pretrained/ETH/model_' + args.test)

        train_path = 'dataset/eth_ucy/datasets/' + args.test + '/train'
        test_path = 'dataset/eth_ucy/datasets/' + args.test + '/test'

        print('loading dataset...')
        self.data_train = TrajectoryDataset(True, args.test, train_path, data_dir_val=None, obs_len=args.len_past, pred_len=args.len_future,
                                            skip=args.skip, delim=args.delim)
        self.data_test = TrajectoryDataset(False, args.test, test_path, obs_len=args.len_past, pred_len=args.len_future,
                                           skip=args.skip, delim=args.delim)
        self.loader_train = DataLoader(self.data_train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.loader_num_workers, collate_fn=self.collate)
        self.loader_test = DataLoader(self.data_test, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.loader_num_workers, collate_fn=self.collate)
        print('dataset loaded')


    def collate(self, batch):
        (index, past, future, present, length, time) = zip(*batch)
        pasts = torch.cat(past)
        futures = torch.cat(future)
        present = torch.cat(present)
        time = torch.cat(time)
        _len = length

        # rel coords
        track = torch.cat((pasts, futures),1)
        track_rel = track[:, 1:] - track[:, :-1]
        track = track[:, 1:]
        pasts_rel = track_rel[:, :7]
        futures_rel = track_rel[:, 7:]
        pasts = track[:, :7]
        futures = track[:, 7:]

        cum_start_idx = [0] + np.cumsum(length).tolist()
        seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        return index, pasts, futures, pasts_rel, futures_rel, present, _len, seq_start_end, cum_start_idx, time

    # def draw(self, past, pred, future, index, iteration):
    #
    #     for p in past:
    #         plt.plot(p[:, 0], p[:, 1], color='b')
    #     for f in future:
    #         plt.plot(f[:, 0], f[:, 1], color='g')
    #     for pr in pred:
    #         for single_pred in pr:
    #             plt.plot(single_pred[:, 0], single_pred[:, 1], color='r', alpha=.4)
    #
    #     plt.axis('equal')
    #
    #     # Save figure in Tensorboard
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='jpeg')
    #     buf.seek(0)
    #     image = Image.open(buf)
    #     image = ToTensor()(image).unsqueeze(0)
    #     self.writer.add_image('Image_test/ex_' + str(index), image.squeeze(0), iteration)
    #     plt.close()

    def test(self):

        # TEST ON SINGLE DATASET
        it_test = iter(self.loader_test)
        self.model.eval()
        with torch.no_grad():
            num_agents = 0
            ADE = FDE =  0
            for step, (index, past, future, past_rel, future_rel, present, length, start_end, batch_split, time) in enumerate(
                    tqdm.tqdm(it_test)):

                past = past.float().cuda()
                past_rel = past_rel.float().cuda()
                future = future.unsqueeze(1).repeat(1, self.args.num_heads, 1, 1).cuda()
                pred, pred_rel, _ = self.model(past, past_rel, length)

                # quantitative
                distances = torch.norm(pred - future, dim=3)
                num_agents += distances.shape[0]
                FDE += distances[:, :, -1].min(1)[0].sum()
                ADE += distances.mean(2).min(1)[0].sum()

        print('test dataset: ' + self.args.test)
        print('ADE: ', round(ADE.item() / num_agents, 4))
        print('FDE: ', round(FDE.item() / num_agents, 4))



def init_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")

    # NTM
    parser.add_argument('--controller_size', type=int, default=100)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--num_input', type=int, default=2)
    parser.add_argument('--controller_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--memory_n', type=int, default=128)
    parser.add_argument('--memory_m', type=int, default=20)

    # Dataset options
    parser.add_argument('--test', default='zara2', type=str)
    parser.add_argument('--delim', default=' ')
    parser.add_argument('--loader_num_workers', default=0, type=int)
    parser.add_argument('--len_past', default=8, type=int)
    parser.add_argument('--len_future', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)

    return parser.parse_args()


def main():
    # Initialize arguments
    args = init_arguments()

    # Initialize random
    init_seed(args.seed)

    # train
    t = tester(args)
    t.test()


if __name__ == '__main__':
    main()

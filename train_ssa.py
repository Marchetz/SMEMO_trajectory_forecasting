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

# models and dataset modules
from models.model_rel_gru import model_rel_gru
from dataset.ssa.dataset_ssa import sequenceDataset

# Default values for program arguments
#get rand int between 0 and 999
RANDOM_SEED = random.randint(0,999)
print('random seed: {}'.format(RANDOM_SEED))

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
np.set_printoptions(threshold=sys.maxsize)

def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)

class trainer:
    def __init__(self, args):
        self.args = args
        self.name_test = str(datetime.datetime.now())[:19]
        folder_test = 'training/ssa/' + self.name_test + '_' + self.args.info
        folder_run = 'runs/ssa/' + self.name_test + '_' + args.info
        self.writer = SummaryWriter(folder_run)
        if not os.path.exists(folder_run):
            os.makedirs(folder_run)
        if not os.path.exists(folder_test):
            os.makedirs(folder_test)
        self.folder_test = folder_test + '/'

        # Initialize the model
        self.model = model_rel_gru(args)

        # Optimization
        self.criterionLoss = nn.MSELoss().cuda()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

        print('loading dataset... ')
        self.data_train = sequenceDataset('dataset/ssa/data_train.npy', args.len_past, args.num_input)
        self.data_val = sequenceDataset('dataset/ssa/data_val.npy', args.len_past, args.num_input)
        self.data_test = sequenceDataset('dataset/ssa/test.npy', args.len_past, args.num_input)
        self.loader_train = DataLoader(self.data_train, collate_fn=self.collate, batch_size=args.batch_size, num_workers=0, shuffle=True)
        self.loader_val = DataLoader(self.data_val, collate_fn=self.collate, batch_size=args.batch_size, num_workers=0, shuffle=False)
        self.loader_test = DataLoader(self.data_test, collate_fn=self.collate, batch_size=args.batch_size, num_workers=0, shuffle=False)
        print('dataset loaded!')
        self.configuration_training()

    def collate(self, batch):

        (pasts_list, futures_list, num_agents) = zip(*batch)
        pasts = torch.cat(pasts_list)
        futures = torch.cat(futures_list)

        track = torch.cat((pasts,futures),1)
        track_rel = torch.zeros(track.shape)
        track_rel[:, 1:] = track[:, 1:] - track[:, :-1]
        pasts_rel = track_rel[:, :20]
        futures_rel = track_rel[:, 20:]

        _len = num_agents
        return pasts, futures, pasts_rel, futures_rel, _len

    def configuration_training(self):
        opt_name = str(self.opt).split(' ')[0]
        self.writer.add_text('Dataset', 'dataset train: {}'.format(len(self.data_train)), 0)
        self.writer.add_text('Dataset', 'dataset val: {}'.format(len(self.data_val)), 0)
        self.writer.add_text('Dataset', 'dataset test: {}'.format(len(self.data_test)), 0)
        self.writer.add_text('Dataset', 'past length: {}'.format(self.args.len_past), 0)
        self.writer.add_text('Dataset', 'future length: {}'.format(self.args.len_future), 0)
        self.writer.add_text('Training', 'batch_size: {}'.format(self.args.batch_size), 0)
        self.writer.add_text('Training', 'opt: {}'.format(opt_name), 0)
        self.writer.add_text('Training', 'learning rate: {}'.format(self.args.learning_rate), 0)
        self.writer.add_text('Training', 'multiplier loss: {}'.format(self.args.loss_multiplier), 0)
        self.writer.add_text('Model', 'controller size: {}'.format(self.args.controller_size), 0)
        self.writer.add_text('Model', 'sequence width: {}'.format(self.args.num_input), 0)
        self.writer.add_text('Model', 'controller layers: {}'.format(self.args.controller_layers), 0)
        self.writer.add_text('Model', 'num heads: {}'.format(self.args.num_heads), 0)
        self.writer.add_text('Model', 'memory_n: {}'.format(self.args.memory_n), 0)
        self.writer.add_text('Model', 'memory_m: {}'.format(self.args.memory_m), 0)


    def draw(self, past, pred, future, index, iteration):

        for p in past:
            plt.plot(p[:, 0], p[:, 1], color='c')
        for f in future:
            plt.plot(f[:, 0], f[:, 1], color='g')
        for pr in pred:
            plt.plot(pr[0, :, 0], pr[0, :, 1], color='r', alpha=.4)

        # plt.axis('equal')
        plt.xlim(-1.80, 1.80)
        plt.ylim(-1.80, 1.80)

        # Save figure in Tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        self.writer.add_image('Image_val/ex_' + str(index), image.squeeze(0), iteration)
        plt.close()

    def train(self):

        self.model = self.model.cuda()
        self.model.train()

        iteration = 0
        best_fde = 9999
        for epoch in range(self.args.num_epoch):
            it = iter(self.loader_train)
            for (past, future, past_rel, future_rel, length) in tqdm.tqdm(it):
                self.opt.zero_grad()
                past = past.float().cuda()
                past_rel = past_rel.float().cuda()
                future = future.float().cuda()
                pred, pred_rel, _ = self.model(past, past_rel, length)
                loss = self.criterionLoss(pred, future.unsqueeze(1)) * self.args.loss_multiplier
                loss.backward()
                clip_grads(self.model)
                self.opt.step()
                self.writer.add_scalar('loss_total/loss_total', loss, iteration)

                if (iteration+1) % 500 == 0:
                    print('val: ' + str(self.args.info) + '_' + 'best iteration: ' + str(iteration))
                    it_val = iter(self.loader_val)
                    self.model.eval()
                    with torch.no_grad():
                        count = 0
                        ADE = FDE_1s = FDE_2s = FDE_3s = FDE_4s = 0
                        for step, (past, future, past_rel, future_rel, length) in enumerate(tqdm.tqdm(it_val)):
                            past = past.float().cuda()
                            past_rel = past_rel.float().cuda()
                            future = future.float().cuda()
                            pred, pred_rel, _ = self.model(past, past_rel, length)

                            errors = torch.norm(pred.squeeze(1) - future, dim=2)
                            count += errors.shape[0]
                            ADE += torch.sum(torch.mean(errors, dim=1))
                            FDE_1s += torch.sum(errors[:, 9])
                            FDE_2s += torch.sum(errors[:, 19])
                            FDE_3s += torch.sum(errors[:, 29])
                            FDE_4s += torch.sum(errors[:, -1])

                            self.draw(past[0:length[0]].cpu(), pred[0:length[0]].cpu(), future[0:length[0]].cpu(), step, iteration)

                    self.writer.add_scalar('accuracy/ADE', ADE / count, iteration)
                    self.writer.add_scalar('accuracy/FDE_1s', FDE_1s / count, iteration)
                    self.writer.add_scalar('accuracy/FDE_2s', FDE_2s / count, iteration)
                    self.writer.add_scalar('accuracy/FDE_3s', FDE_3s / count, iteration)
                    self.writer.add_scalar('accuracy/FDE_4s', FDE_4s / count, iteration)
                    torch.save(self.model, self.folder_test + 'model_it_' + str(iteration) + '_' + self.name_test)

                    if (FDE_4s / count) < best_fde:
                        best_fde = FDE_4s / count
                        torch.save(self.model, 'pretrained/ssa/' + 'model_' + self.args.info)

                    self.model.train()
                iteration = iteration + 1


def init_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", type=str, default='name_exp')
    parser.add_argument("--learning_rate", type=int, default=0.001)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--loss_multiplier', type=int, default=1)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")

    # SMEMO
    parser.add_argument('--controller_size', type=int, default=100)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--num_input', type=int, default=2)
    parser.add_argument('--controller_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--memory_n', type=int, default=128)
    parser.add_argument('--memory_m', type=int, default=20)
    parser.add_argument('--len_past', type=int, default=20)
    parser.add_argument('--len_future', type=int, default=40)

    return parser.parse_args()


def main():
    # Initialize arguments
    args = init_arguments()

    # Initialize random
    init_seed(args.seed)

    # train
    t = trainer(args)
    t.train()


if __name__ == '__main__':
    main()

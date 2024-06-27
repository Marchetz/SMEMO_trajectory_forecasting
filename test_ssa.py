import argparse
import random
import sys
import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")
from dataset.ssa.dataset_ssa import sequenceDataset
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
        self.model = torch.load(args.model_path)

        print('loading dataset...')
        self.data_train = sequenceDataset('dataset/ssa/data_train.npy', args.len_past, args.num_input)
        self.data_test = sequenceDataset('dataset/ssa/test.npy', args.len_past, args.num_input)
        self.loader_train = DataLoader(self.data_train, collate_fn=self.collate, batch_size=args.batch_size, num_workers=0, shuffle=True)
        self.loader_test = DataLoader(self.data_test, collate_fn=self.collate, batch_size=args.batch_size, num_workers=0, shuffle=False)
        self.loader_test_1 = DataLoader(self.data_test, collate_fn=self.collate, batch_size=1, num_workers=0, shuffle=False)

        print('dataset loaded')

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

        it_test = iter(self.loader_test)
        self.model.eval()
        fde_total = []
        ade_total = []
        with torch.no_grad():
            count = 0
            ADE = FDE = 0
            pred_total = []
            past_total = []
            for step, (past, future, past_rel, future_rel, length) in enumerate(tqdm.tqdm(it_test)):
                cum_start_idx = [0] + np.cumsum(length).tolist()
                seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
                past = past.float().cuda()
                past_rel = past_rel.float().cuda()
                future = future.unsqueeze(1).repeat(1, self.args.num_heads, 1, 1).cuda()
                pred, pred_rel, _ = self.model(past, past_rel, length)

                # quantitative
                errors = torch.norm(pred - future, dim=3).squeeze(1)
                count += errors.shape[0]
                fde_total.extend(errors[:, -1].cpu().tolist())
                ade_total.extend(torch.mean(errors, dim=1).cpu().tolist())
                ADE += torch.sum(torch.mean(errors, dim=1))
                FDE += torch.sum(errors[:, -1])

                pred = pred.squeeze(1)
                for i in range(len(seq_start_end)):
                    start = seq_start_end[i][0]
                    end = seq_start_end[i][1]
                    past_total.append(past[start:end])
                    pred_total.append(pred[start:end])

        # #order gt
        it_test = iter(self.loader_test_1)
        order_gt_total = []
        for step, (past, future, past_rel, future_rel, length) in enumerate(tqdm.tqdm(it_test)):
            #     if past.shape[0] == 5:
            index_center = []
            index_foo = []
            sign_init = past[:, 0].sign()
            for i in range(len(future)):
                if (future[i].sign() == sign_init[i]).all():
                    index_foo.append(torch.Tensor([i, torch.norm(future[i][-1] - torch.zeros(2), dim=0)]))
                    index_center.append(torch.Tensor([80]).squeeze())
                else:
                    # index_center.append(torch.where(future[i].sign() != sign_init[i])[0][0] + 20)
                    index_center.append(torch.where((future[i].sign() != sign_init[i]).sum(1) == 2)[0][0] + 20)
            if len(index_foo) > 0:
                index_foo = torch.stack(index_foo)
                temp = index_foo[index_foo[:, 1].sort()[1]]
                temp[:, 1] = torch.arange(len(index_foo)) + 80
                index_center = torch.stack(index_center)
                index_center[temp[:, 0].long()] = temp[:, 1]
                order_gt = index_center.sort()[1]
            else:
                order_gt = torch.stack(index_center).sort()[1]
            order_gt_total.append(order_gt)

        col_total = 0
        order_smemo_total = []
        for k in tqdm.tqdm(range(len(pred_total))):
            #     if pred_total[k].shape[0] == 5:
            pr = pred_total[k].cpu()
            pas = past_total[k].cpu()
            index_center = []
            index_foo = []
            sign_init = pas[:, 0].sign()
            max_dist = pas[:, 0].abs().argmax(1)
            for i in range(len(pr)):
                if (pr[i].sign()[:, max_dist[i]] == sign_init[i, max_dist[i]]).all():
                    index_foo.append(torch.Tensor([i, torch.norm(pr[i][-1] - torch.zeros(2), dim=0)]))
                    index_center.append(torch.Tensor([80]).squeeze())
                else:
                    # pr[i].sign()[:, max_dist[i]] != sign_init[i,  max_dist[i]]
                    index_center.append(
                        torch.where(pr[i].sign()[:, max_dist[i]] != sign_init[i, max_dist[i]])[0][0] + 20)
                    # index_center.append(torch.where((pr[i].sign() != sign_init[i]).sum(1) == 2)[0][0] + 20)

            if len(index_foo) > 0:
                index_foo = torch.stack(index_foo)
                temp = index_foo[index_foo[:, 1].sort()[1]]
                temp[:, 1] = torch.arange(len(index_foo)) + 80
                index_center = torch.stack(index_center)
                index_center[temp[:, 0].long()] = temp[:, 1]
                order_pred = index_center.sort()[1]
            else:
                order_pred = torch.stack(index_center).sort()[1]
            order_smemo_total.append(order_pred)

        # smemo
        values = []
        for i in range(len(order_gt_total)):
            tau, p_value = stats.kendalltau(order_gt_total[i], order_smemo_total[i])
            values.append(tau)

        # Calculate mean
        mean_fde = np.mean(fde_total)
        mean_ade = np.mean(ade_total)
        mean_kde = np.mean(values)

        print('dataset SSA')
        print('ADE: ', round(ADE.item() / count,5))
        print('FDE: ', round(FDE.item() / count,5))
        print('KENDALL: ', np.mean(values))

        print('-------------------------------------------------------')


def init_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", type=str, default='')
    parser.add_argument("--model_path", type=str, default='pretrained/SSA/model_official')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")

    # SMEMO
    parser.add_argument('--controller_size', type=int, default=100)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--num_input', type=int, default=2)
    parser.add_argument('--controller_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--memory_n', type=int, default=128)
    parser.add_argument('--memory_m', type=int, default=20)

    # Dataset options
    parser.add_argument('--loader_num_workers', default=0, type=int)
    parser.add_argument('--len_past', default=20, type=int)
    parser.add_argument('--len_future', default=40, type=int)

    return parser.parse_args()


def main():
    # Initialize arguments
    args = init_arguments()

    # Initialize random
    # init_seed(args.seed)

    # train
    t = tester(args)
    t.test()


if __name__ == '__main__':
    main()

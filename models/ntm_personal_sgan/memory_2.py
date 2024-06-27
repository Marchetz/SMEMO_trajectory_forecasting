"""An NTM's memory implementation."""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import pdb

class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(N, M))

        # Initialize memory bias
        #zeros
        #nn.init.constant_(self.mem_bias, 1e-6)

        #random
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def similarity_mask_inf(self, k, β, memory_total, mask):
        k = k.unsqueeze(1)
        sim = β * (F.cosine_similarity(memory_total + 1e-16, k + 1e-16, dim=-1) + 1)
        sim_2 = sim.clone()
        temp = mask == 0.0
        index_all_zeros = torch.where(temp.all(dim=1))[0]
        sim_2[mask == 0] = -float("inf")
        if len(index_all_zeros) > 0:
            sim_2[index_all_zeros] = 0.0
        w = F.softmax(sim_2, dim=1)
        if len(index_all_zeros) > 0:
            mask_softmax = torch.ones(w.shape).cuda()
            mask_softmax[index_all_zeros] = 0.0
            w = w * mask_softmax

        return w

    def similarity(self, k, β, memory_total):

        k = k.unsqueeze(1)
        w = F.softmax(β * F.cosine_similarity(memory_total + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w


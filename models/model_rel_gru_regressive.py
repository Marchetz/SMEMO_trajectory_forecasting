import pdb
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_max, scatter_add
from .ntm_personal_sgan.memory import NTMMemory
import torch.nn.functional as F
from torch.nn import Parameter


class model_rel_gru_regressive(torch.nn.Module):
    def __init__(self, config):
        """ Initialize the SMEMO  model

        Attributes
        ----------
        config : smemo parameters
        """
        super(model_rel_gru_regressive, self).__init__()

        self.name_model = 'model_rel_gru_regressive'

        # Save args
        self.len_future = config.len_future
        self.num_input = config.num_input
        self.num_output = config.num_input
        self.controller_size = config.controller_size
        self.controller_layers = config.controller_layers
        self.num_heads = config.num_heads
        self.N = config.memory_n
        self.M = config.memory_m

        # encoder_past
        self.embed_past = nn.Sequential(
            nn.Linear(self.num_input, 8),
            nn.ReLU(),
            nn.Linear(8, 16)
        )
        self.embed_past_rel = nn.Sequential(
            nn.Linear(self.num_input, 8),
            nn.ReLU(),
            nn.Linear(8, 16)
        )
        self.encoder_past = nn.GRU(16 + self.M, self.controller_size, 1, batch_first=True)
        self.encoder_past_rel = nn.GRU(16, self.controller_size, 1, batch_first=True)

        # decoder
        self.decoder = nn.GRU(self.controller_size + self.M, self.controller_size + self.M, 1, batch_first=True)
        self.FC_output = nn.Linear(self.controller_size + self.M, 2)

        # multipred
        self.read_lengths = [self.M, 1]
        self.write_lengths = [self.M, 1, self.M, self.M]
        self.fc_reads = nn.ModuleList([])
        self.fc_writes = nn.ModuleList([])
        dim_fc = self.controller_size

        for i in range(self.num_heads):
            self.fc_reads += [nn.Linear(dim_fc, sum(self.read_lengths))]
        self.fc_read_past = nn.Linear(dim_fc, sum(self.read_lengths))
        for i in range(self.num_heads):
            self.fc_writes += [nn.Linear(dim_fc, sum(self.write_lengths))]
        self.fc_write_past = nn.Linear(dim_fc, sum(self.write_lengths))

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.controller_layers, 1, self.controller_size) * 0.05)
        self.lstm_h_bias_rel = Parameter(torch.randn(self.controller_layers, 1, self.controller_size) * 0.05)
        self.lstm_h_bias_fut = Parameter(torch.randn(self.controller_layers, 1, self.controller_size + self.M) * 0.05)

        # Create the NTM components
        self.memory = NTMMemory(self.N, self.M)

        # weight initialization: xavier
        self.init_r = self.reset_parameters()

    def reset_parameters(self):

        # weight initialization: xavier
        torch.nn.init.xavier_normal_(self.fc_write_past.weight)
        torch.nn.init.zeros_(self.fc_write_past.bias)
        for i in range(self.num_heads):
            torch.nn.init.xavier_normal_(self.fc_reads[i].weight)
            torch.nn.init.zeros_(self.fc_reads[i].bias)

        torch.nn.init.xavier_normal_(self.embed_past[0].weight)
        torch.nn.init.zeros_(self.embed_past[0].bias)
        torch.nn.init.xavier_normal_(self.embed_past[0].weight)
        torch.nn.init.zeros_(self.embed_past[0].bias)
        torch.nn.init.xavier_normal_(self.embed_past_rel[0].weight)
        torch.nn.init.zeros_(self.embed_past_rel[0].bias)
        torch.nn.init.xavier_normal_(self.embed_past_rel[0].weight)
        torch.nn.init.zeros_(self.embed_past_rel[0].bias)

        nn.init.xavier_normal_(self.encoder_past.weight_ih_l0)
        nn.init.xavier_normal_(self.encoder_past.weight_hh_l0)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)
        nn.init.xavier_normal_(self.encoder_past_rel.weight_ih_l0)
        nn.init.xavier_normal_(self.encoder_past_rel.weight_hh_l0)
        nn.init.zeros_(self.encoder_past_rel.bias_ih_l0)
        nn.init.zeros_(self.encoder_past_rel.bias_hh_l0)

        nn.init.xavier_normal_(self.decoder.weight_ih_l0)
        nn.init.xavier_normal_(self.decoder.weight_hh_l0)
        nn.init.zeros_(self.decoder.bias_ih_l0)
        nn.init.zeros_(self.decoder.bias_hh_l0)

        init_r = torch.randn(1, self.M) * 0.01
        self.register_buffer("read{}_bias".format(0), init_r.data)
        return init_r

    def _split_cols(self, mat, lengths):
        """Split a 2D matrix to variable length columns."""
        assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
        l = np.cumsum([0] + lengths)
        results = []
        for s, e in zip(l[:-1], l[1:]):
            results += [mat[:, s:e]]
        return results

    def init_sequence(self, length, device):
        """Initializing the state."""
        self.batch_size = len(length)
        self.memory.reset(len(length))
        self.previous_state = self.create_new_state(sum(length), device)

    def create_new_state(self, size, device):
        init_r = self.init_r.clone().repeat(size, 1).to(device)
        lstm_h = self.lstm_h_bias.clone().repeat(1, size, 1).to(device)
        lstm_h_rel = self.lstm_h_bias_rel.clone().repeat(1, size, 1).to(device)
        lstm_h_fut = self.lstm_h_bias_fut.clone().repeat(1, size, 1).to(device)
        return init_r, (lstm_h), (lstm_h_rel), (lstm_h_fut)

    def get_init_status(self, length):
        num_a = sum(length)
        prev_reads, prev_controller_state, prev_controller_state_rel, prev_controller_state_fut = self.previous_state
        read = prev_reads.unsqueeze(1)

        return [self.memory.memory, prev_controller_state, prev_controller_state_rel,
                torch.zeros(num_a, self.N), torch.zeros(num_a, self.N), read, torch.zeros(num_a, self.M), torch.zeros(num_a, self.M),
                torch.zeros(num_a, self.N, self.M), torch.zeros(num_a, self.N, self.M),
                torch.zeros(num_a, self.M), torch.zeros(num_a, 1), torch.zeros(num_a, self.M), torch.zeros(num_a, 1)]


    def forward(self, past, past_rel, length):
        device = torch.device('cuda')
        self.init_sequence(length, device)

        pred = torch.Tensor().to(device)
        present = past[:, -1].unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        length = torch.Tensor(length).long().to(device)
        temp = torch.arange(len(length)).cuda().to(device)

        length_repeat = length.repeat(self.num_heads)
        temp_repeat = torch.arange(len(length_repeat)).cuda().to(device)

        # Past
        for i in range(past.shape[1]):
            if i != past.shape[1] - 1:
                current = past[:, i, :].unsqueeze(1)
                current_rel = past_rel[:, i, :].unsqueeze(1)
                out, self.previous_state = self.forward_single(current, current_rel, length, temp)

            if i == past.shape[1] - 1:
                current = past[:, i, :].unsqueeze(1)
                current_rel = past_rel[:, i, :].unsqueeze(1)
                out, self.previous_state = self.forward_future(current, current_rel, length, temp_repeat, length_repeat, first=True)

                point_future = present + out
                pred = torch.cat((pred, point_future), dim=2)
                present = present + out
                present_rel = out

        # Future
        for i in range(self.len_future-1):
            current = present.view(-1,1,2)
            current_rel = present_rel.reshape(-1,1,2)
            out, self.previous_state  = self.forward_future(current, current_rel, length, temp_repeat, length_repeat, first=False)
            point_future = present + out
            pred = torch.cat((pred, point_future), dim=2)
            present = present + out
            present_rel = out
        return pred

    def forward_future(self, past, past_rel, length, temp, length_repeat, first=False):
        """
        Forward pass that encodes past and future and decodes the future.
        :param past: past trajectory
        :param future: future trajectory
        :return: decoded future
        """
        # INIT
        prev_reads, prev_controller_state, prev_controller_state_rel, prev_controller_state_fut = self.previous_state
        if first:
            prev_controller_state_fut = prev_controller_state_fut.repeat(1, self.num_heads,1)
        past = self.embed_past(past)
        past_rel = self.embed_past_rel(past_rel)

        # gru_rel
        output_past_rel, state_past_rel = self.encoder_past_rel(past_rel, prev_controller_state_rel)

        input = torch.cat((past, prev_reads.unsqueeze(1)), dim=2)
        output_past, state_past = self.encoder_past(input, prev_controller_state)

        if first:
            self.new_memory = self.memory.memory.repeat(self.num_heads, 1, 1)

        #reading
        info_label = output_past
        o_total = torch.Tensor().cuda()
        for i in range(self.num_heads):
            if first:
                o = self.fc_reads[i](info_label)
            else:
                o = self.fc_reads[i](info_label[i * sum(length):i * sum(length) + sum(length)])
            o_total = torch.cat((o_total, o), 0)
        k, β = self._split_cols(o_total.squeeze(1), self.read_lengths)
        β = F.softplus(β)
        mem = torch.repeat_interleave(self.new_memory, length_repeat, dim=0)
        w_read = self.memory.similarity(k, β, mem)
        reading_total = torch.matmul(w_read.unsqueeze(1), mem).squeeze(1)

        if first:
            output_past_total = output_past_rel.repeat(self.num_heads, 1, 1)
        else:
            output_past_total = output_past_rel
        input_decoder = torch.cat((output_past_total, reading_total.unsqueeze(1)), dim=2)
        output_future, state_future = self.decoder(input_decoder, prev_controller_state_fut)
        out_total = self.FC_output(output_future)

        # WRITING
        o_total = torch.Tensor().cuda()
        for i in range(self.num_heads):
            if first:
                o = self.fc_writes[i](info_label)
            else:
                o = self.fc_writes[i](info_label[i * sum(length):i * sum(length) + sum(length)])
            o_total = torch.cat((o_total, o), 0)

        k, β, e, a = self._split_cols(o_total.squeeze(1), self.write_lengths)
        e = torch.sigmoid(e)
        β = F.softplus(β)
        mem = torch.repeat_interleave(self.new_memory, length_repeat, dim=0)
        w_write = self.memory.similarity(k, β, mem)

        erase = torch.matmul(w_write.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w_write.unsqueeze(-1), a.unsqueeze(1))

        temp_long = torch.repeat_interleave(temp, length_repeat, dim=0)
        labels = temp_long.view(temp_long.size(0), 1, 1).expand(-1, erase.size(1), erase.size(2))
        unique_labels = temp.view(temp.size(0), 1, 1).expand(-1, erase.size(1), erase.size(2))

        out_erase = torch.zeros_like(unique_labels, dtype=torch.float)
        out_erase, argmax = scatter_max(erase, labels, dim=0, out=out_erase)
        out_add = torch.zeros_like(unique_labels, dtype=torch.float)
        out_add, argmax = scatter_max(add, labels, dim=0, out=out_add)

        self.new_memory = (1 - out_erase) * self.new_memory + out_add

        if first:
            state_past = state_past.repeat(1, self.num_heads,1)
            state_past_rel = state_past_rel.repeat(1, self.num_heads,1)
        state = (reading_total, state_past, state_past_rel, state_future)
        out_total = out_total.view(self.num_heads, sum(length), 1, 2).permute(1, 0, 2, 3)
        return out_total, state

    def forward_single(self, past, past_rel, length=None, temp=None):
        """
        Forward pass that encodes past and future and decodes the future.
        :param past: past trajectory
        :param future: future trajectory
        :return: decoded future
        """
        # INIT
        prev_reads, prev_controller_state, prev_controller_state_rel, prev_controller_state_fut = self.previous_state
        past = self.embed_past(past)
        past_rel = self.embed_past_rel(past_rel)

        # gru_rel
        output_past_rel, state_past_rel = self.encoder_past_rel(past_rel, prev_controller_state_rel)

        # gru_abs
        input = torch.cat((past, prev_reads.unsqueeze(1)), dim=2)
        output_past, state_past = self.encoder_past(input, prev_controller_state)

        # reading
        memory_total = torch.repeat_interleave(self.memory.memory, length, dim=0)
        info_label = output_past
        o_total = self.fc_read_past(info_label)
        k, β = self._split_cols(o_total.squeeze(1), self.read_lengths)
        β = F.softplus(β)
        w_read = self.memory.similarity(k, β, memory_total)
        reading_total = torch.matmul(w_read.unsqueeze(1), memory_total).squeeze(1)
        output_past_total = output_past_rel
        input_decoder = torch.cat((output_past_total, reading_total.unsqueeze(1)), dim=2)
        output_future, state_future = self.decoder(input_decoder, prev_controller_state_fut)
        out_total = self.FC_output(output_future)

        # WRITING
        o = self.fc_write_past(info_label)
        k, β, e, a = self._split_cols(o.squeeze(1), self.write_lengths)
        e = torch.sigmoid(e)
        β = F.softplus(β)
        w_write = self.memory.similarity(k, β, memory_total)

        erase = torch.matmul(w_write.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w_write.unsqueeze(-1), a.unsqueeze(1))

        temp_long = torch.repeat_interleave(temp, length, dim=0)
        labels = temp_long.view(temp_long.size(0), 1, 1).expand(-1, erase.size(1), erase.size(2))
        unique_labels = temp.view(temp.size(0), 1, 1).expand(-1, erase.size(1), erase.size(2))

        out_erase = torch.zeros_like(unique_labels, dtype=torch.float)
        out_erase, argmax = scatter_max(erase, labels, dim=0, out=out_erase)
        out_add = torch.zeros_like(unique_labels, dtype=torch.float)
        out_add, argmax = scatter_max(add, labels, dim=0, out=out_add)

        self.memory.memory = (1 - out_erase) * self.memory.memory + out_add

        state = (reading_total, state_past, state_past_rel, state_future)

        return out_total, state

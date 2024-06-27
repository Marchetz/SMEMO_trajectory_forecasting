import pdb

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_max, scatter_add, scatter_mean
from .ntm_personal_sgan.memory import NTMMemory
import torch.nn.functional as F
from torch.nn import Parameter
torch.manual_seed(1000)

KEYS = ["memory", "state", "state_rel", "reading", "writing", "read", "erase", "add",
        "k_reading", "beta_reading", "k_writing", "beta_writing"]


class model_rel_gru_mask(nn.Module):
    def __init__(self, config):
        super(model_rel_gru_mask, self).__init__()

        self.name_model = 'smemo_rel_gru_mask'

        # Save args
        self.len_future = config.len_future
        self.num_input = config.num_input
        self.num_output = config.num_input
        self.controller_size = config.controller_size
        self.controller_layers = config.controller_layers
        self.num_heads = config.num_heads
        self.N = config.memory_n
        self.M = config.memory_m

        # self.dropout = nn.Dropout(0.25)

        # encoder_past
        self.embed_past = nn.Sequential(
            nn.Linear(self.num_input, 8),
            nn.ReLU(),
            #self.dropout,
            nn.Linear(8, 16)
        )
        self.embed_past_rel = nn.Sequential(
            nn.Linear(self.num_input, 8),
            nn.ReLU(),
            #self.dropout,
            nn.Linear(8, 16)
        )
        self.encoder_past = nn.GRU(16 + self.M, self.controller_size, 1, batch_first=True)
        self.encoder_past_rel = nn.GRU(16, self.controller_size, 1, batch_first=True)

        #decoder
        #self.decoder = nn.Linear(self.controller_size + self.M, 2)
        self.decoder = nn.GRU(self.controller_size + self.M, self.controller_size + self.M, 1, batch_first=True)
        self.FC_output = nn.Linear(self.controller_size + self.M, 2)

        #multipred
        self.read_lengths = [self.M, 1]
        self.fc_reads = nn.ModuleList([])
        dim_fc = self.controller_size
        for i in range(self.num_heads):
            self.fc_reads += [nn.Linear(dim_fc, sum(self.read_lengths))]

        self.write_lengths_new = [self.M, 1, self.M, self.M]
        self.fc_write_new = nn.Linear(dim_fc, sum(self.write_lengths_new))

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
        torch.nn.init.xavier_normal_(self.fc_write_new.weight)
        torch.nn.init.zeros_(self.fc_write_new.bias)
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
        init_r = init_r.cuda()
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

    def init_sequence(self, length):
        """Initializing the state."""
        self.batch_size = len(length)
        self.memory.reset(len(length))
        self.previous_state = self.create_new_state(sum(length))

    def create_new_state(self, size):
        init_r = self.init_r.clone().repeat(size, 1)
        lstm_h = self.lstm_h_bias.clone().repeat(1, size, 1).cuda()
        lstm_h_rel = self.lstm_h_bias_rel.clone().repeat(1, size, 1).cuda()
        lstm_h_fut = self.lstm_h_bias_fut.clone().repeat(1, size*self.num_heads, 1).cuda()
        #lstm_h_fut = self.lstm_h_bias_fut.clone().repeat(1, size, 1).cuda()
        return init_r, (lstm_h), (lstm_h_rel), (lstm_h_fut)

    def init_debug(self, length):
        debug_dict = {}
        for k in KEYS:
            debug_dict[k] = []
        debug_value = self.get_init_status(length)
        debug_dict = self.append_debug(debug_dict, debug_value)
        return debug_dict

    def final_debug(self, debug_dict):
        for k in KEYS:
            print(k)
            debug_dict[k] = torch.stack(debug_dict[k])
        return debug_dict

    def get_init_status(self, length):
        num_a = sum(length)
        prev_reads, prev_controller_state, prev_controller_state_rel, prev_controller_state_fut = self.previous_state
        read = prev_reads.unsqueeze(1).repeat(1,self.num_heads,1)

        return [self.memory.memory, prev_controller_state, prev_controller_state_rel,
                torch.zeros(num_a, self.num_heads, self.N), torch.zeros(num_a, self.N), read,
                torch.zeros(num_a, self.M), torch.zeros(num_a, self.M), torch.zeros(num_a, self.num_heads, self.M),
                torch.zeros(num_a, self.num_heads, 1), torch.zeros(num_a, self.M), torch.zeros(num_a, 1)]

    def append_debug(self, debug_dict, debug_value):
        for i, k in enumerate(KEYS):
            debug_dict[k].append(debug_value[i].cpu())
        return debug_dict

    def get_mask(self, length, space, device, mask_writing_flag, mask_reading_flag, mask_total):
        m_tot_writing = torch.Tensor()
        m_tot_reading = torch.Tensor()

        for i in range(max(length)+1):
            m_single_space = torch.zeros(1, self.N)
            m_single_space[:, i * space:i * space + space] = 1
            m_tot_writing = torch.cat((m_tot_writing, m_single_space), 0)

            m_single = torch.zeros(1, self.N)
            m_single[:, 0:i * space] = 1
            m_tot_reading = torch.cat((m_tot_reading, m_single), 0)

        m_index = torch.Tensor()
        for i in range(len(length)):
            m_index = torch.cat((m_index, torch.arange(length[i])), 0)

        # writing mask
        mask_writing = torch.zeros(sum(length), self.N)
        mask_writing[torch.arange(len(mask_writing)).long()] = m_tot_writing[m_index.long()]
        mask_writing = mask_writing.to(device)

        # reading masks
        mask_reading = torch.zeros(sum(length), self.N)
        m_index_reading = torch.Tensor(length).repeat_interleave(torch.Tensor(length).long())
        mask_reading[torch.arange(len(mask_reading)).long()] = m_tot_reading[m_index_reading.long()]
        mask_reading = mask_reading.to(device)

        if not mask_total:
            mask_reading -= mask_writing

        if not mask_writing_flag:
            mask_writing = None

        if not mask_reading_flag:
            mask_reading = None

        return mask_writing, mask_reading, m_tot_writing

    def get_config_forward(self, args):
        config = {}
        config['regressive'] = args.regressive
        config['mask_writing_flag'] = args.mask_writing_flag
        config['mask_reading_flag'] = args.mask_reading_flag
        config['mask_total'] = args.mask_total
        config['space'] = args.space
        return config


    def forward_abl(self, past, past_rel, length, debug=False, ablation=None):
        pred, debug_value = self.forward(past, past_rel, length, debug, ablation)
        return pred, debug_value


    def forward(self, past, past_rel, length, debug=False, ablation=None):
        print('modello rel_gru_mask')
        if ablation == None:
            ablation = {}
            ablation["write"] = None
            ablation["read"] = None
            ablation["move"] = None

        mask_writing_flag = True
        mask_reading_flag = True
        mask_total = False
        space = 5
        mask_writing, mask_reading, m_single_space = self.get_mask(length, space, torch.device('cuda'), mask_writing_flag,
                                                                   mask_reading_flag, mask_total)

        pred = torch.Tensor().cuda()
        pred_rel = torch.Tensor().cuda()
        self.init_sequence(length)
        if debug:
            debug_dict = self.init_debug(length)

        padding = torch.zeros(past.shape[0], self.len_future, 16, device=torch.device('cuda:0'))
        present = past[:, -1].unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        length = torch.Tensor(length).long().cuda()
        temp = torch.arange(len(length)).cuda()

        # Past
        for i in range(past.shape[1]):
            current = past[:, i, :].unsqueeze(1)
            current_rel = past_rel[:, i, :].unsqueeze(1)
            out, self.previous_state, debug_value = self.forward_single(current, current_rel, length, temp, True,
                                                                        mask_writing, mask_reading, m_single_space,
                                                                        ablation=ablation)

            if debug:
                debug_dict = self.append_debug(debug_dict, debug_value)

        # Future
        for i in range(self.len_future):
            current = padding[:, i, :].unsqueeze(1)
            current_rel = padding[:, i, :].unsqueeze(1)
            out, self.previous_state, debug_value = self.forward_single(current, current_rel, length, temp, False,
                                                                        mask_writing, mask_reading, m_single_space,
                                                                        ablation=ablation)
            present = present + out
            pred = torch.cat((pred, present), dim=2)
            pred_rel = torch.cat((pred_rel, out), dim=2)

            if debug:
                debug_dict = self.append_debug(debug_dict, debug_value)

        if debug:
            return pred, self.final_debug(debug_dict)
        else:
            return pred, pred_rel

    def forward_single(self, past, past_rel, length, temp, isPast, mask_writing, mask_reading, m_single_space, ablation):
        """
        Forward pass that encodes past and future and decodes the future.
        :param past: past trajectory
        :param future: future trajectory
        :return: decoded future
        """
        current = past.clone().squeeze(1)

        #INIT
        prev_reads, prev_controller_state, prev_controller_state_rel, prev_controller_state_fut = self.previous_state
        if isPast:
            past = self.embed_past(past)
            past_rel = self.embed_past_rel(past_rel)

        #gru_rel
        output_past_rel, state_past_rel = self.encoder_past_rel(past_rel, prev_controller_state_rel)

        #gru_abs
        input = torch.cat((past, prev_reads.unsqueeze(1)), dim=2)
        output_past, state_past = self.encoder_past(input, prev_controller_state)

        memory_total = torch.repeat_interleave(self.memory.memory, length, dim=0)
        info_label = output_past

        if not ablation["read"]:
            o_total = torch.Tensor().cuda()
            for i in range(len(self.fc_reads)):
                o = self.fc_reads[i](info_label)
                o_total = torch.cat((o_total, o), 0)
            memory_total_2 = memory_total.repeat(self.num_heads, 1, 1)
            k, β = self._split_cols(o_total.squeeze(1), self.read_lengths)
            β = F.softplus(β)


            #mask
            #distance = torch.nn.functional.pdist(current)
            # mask_reading_new = torch.Tensor()
            # for i in range(len(length)):
            #     start = seq_start_end[i][0]
            #     end = seq_start_end[i][1]
            #     distance = torch.norm(current[start:end, None] - current[start:end], dim=2)
            #     sort, sort_ind = torch.sort(distance, dim=1)
            #     select = sort_ind[:,1:6]
            #     m_read = m_single_space[select].sum(1)
            #     mask_reading_new = torch.cat((mask_reading_new, m_read),0) #new mask for nb (top-5)

            if mask_reading is None:
                w_read = self.memory.similarity(k, β, memory_total_2)
            else:
                w_read = self.memory.similarity_mask_inf(k, β, memory_total_2, mask_reading.repeat(self.num_heads,1))

            reading_total = torch.matmul(w_read.unsqueeze(1), memory_total_2).squeeze(1)
            k_reading = k
            beta_reading = β
        else:
            k_reading = torch.Tensor([0]).cuda()
            beta_reading = torch.Tensor([0]).cuda()
            w_read = torch.zeros(past.shape[0], 128).cuda()
            if ablation["read"] == 'zeros':
                reading_total = torch.zeros(past.shape[0] * self.num_heads, 20).cuda()
            if ablation["read"] == 'rand':
                reading_total = torch.rand(past.shape[0] * self.num_heads, 20).cuda()

        if not ablation["move"]:
            output_past_total = output_past_rel.repeat(self.num_heads, 1, 1)
        else:
            if ablation["move"] == 'zeros':
                output_past_total = torch.zeros(output_past_rel.shape).cuda().repeat(self.num_heads, 1, 1)
            if ablation["move"] == 'rand':
                output_past_total = torch.rand(output_past_rel.shape).cuda().repeat(self.num_heads, 1, 1)
        input_decoder = torch.cat((output_past_total, reading_total.unsqueeze(1)), dim=2)
        output_future, state_future = self.decoder(input_decoder, prev_controller_state_fut)
        out_total = self.FC_output(output_future)
        out_total = out_total.view(self.num_heads, past.shape[0], 1, 2).permute(1, 0, 2, 3)

        # WRITING
        o = self.fc_write_new(info_label)
        k, β, e, a = self._split_cols(o.squeeze(1), self.write_lengths_new)
        e = torch.sigmoid(e)
        β = F.softplus(β)
        if mask_writing is None:
            w_write = self.memory.similarity(k, β, memory_total)
        else:
            w_write = self.memory.similarity_mask_inf(k, β, memory_total, mask_writing)

        k_writing = k
        beta_writing = β

        erase = torch.matmul(w_write.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w_write.unsqueeze(-1), a.unsqueeze(1))

        temp_long = torch.repeat_interleave(temp, length, dim=0)
        labels = temp_long.view(temp_long.size(0), 1, 1).expand(-1, erase.size(1), erase.size(2))
        unique_labels = temp.view(temp.size(0), 1, 1).expand(-1, erase.size(1), erase.size(2))

        out_erase = torch.zeros_like(unique_labels, dtype=torch.float)
        out_erase, argmax = scatter_max(erase, labels, dim=0, out=out_erase)
        out_add = torch.zeros_like(unique_labels, dtype=torch.float)
        out_add, argmax = scatter_max(add, labels, dim=0, out=out_add)

        if not ablation["write"]:
            self.memory.memory = (1 - out_erase) * self.memory.memory + out_add
        reading = reading_total.view(self.num_heads, past.shape[0], -1).permute(1,0,2)
        reading_prev = reading.mean(1)
        state = (reading_prev, state_past, state_past_rel, state_future)

        debug_value = [self.memory.memory, state_past, state_past_rel,
                       w_read.view(self.num_heads, past.shape[0], -1).permute(1,0,2), w_write, reading, e, a,
                       k_reading.view(self.num_heads, past.shape[0], -1).permute(1,0,2),
                       beta_reading.view(self.num_heads, past.shape[0], -1).permute(1,0,2), k_writing, beta_writing]

        return out_total, state, debug_value






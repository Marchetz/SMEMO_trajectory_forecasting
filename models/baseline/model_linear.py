import torch
import torch.nn as nn

class model_linear(nn.Module):
    def __init__(self, settings):
        super(model_linear, self).__init__()

        self.name_model = 'linear'

        #parameters
        self.len_past = settings.len_past
        self.len_future = settings.len_future

        #layers
        self.linear = torch.nn.Linear(self.len_past * 2, self.len_future * 2)

        # weight initialization: xavier
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, past, past_rel=None, length=None, debug=False):

        dim_batch = past.size()[0]
        past = past.view(dim_batch, -1)
        output = self.linear(past)
        output = output.view(dim_batch, self.len_future, 2).unsqueeze(1)
        return output, None, None

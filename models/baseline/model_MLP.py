import torch
import torch.nn as nn


class model_MLP(nn.Module):
    def __init__(self, settings):
        super(model_MLP, self).__init__()

        self.name_model = 'MLP'

        #parameters
        self.len_past = settings.len_past
        self.len_future = settings.len_future

        #layers
        #self.firstLayer = torch.nn.Linear((self.len_past - 1) * 2, self.len_past * 4)
        self.firstLayer = torch.nn.Linear((self.len_past ) * 2, self.len_past * 4)
        self.secondLayer = torch.nn.Linear(self.len_past * 4, self.len_future * 2)
        self.relu = nn.ReLU()

        # weight initialization: xavier
        torch.nn.init.xavier_uniform_(self.firstLayer.weight)
        torch.nn.init.xavier_uniform_(self.secondLayer.weight)
        torch.nn.init.zeros_(self.firstLayer.bias)
        torch.nn.init.zeros_(self.secondLayer.bias)

    def forward(self, past, past_rel=None, length=None, debug=False):

        dim_batch = past.size()[0]
        past = past.view(dim_batch, -1)
        hidden = self.relu(self.firstLayer(past))
        output = self.secondLayer(hidden)
        output = output.view(dim_batch, self.len_future, 2).unsqueeze(1)
        return output, None, None

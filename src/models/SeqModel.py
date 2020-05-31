import torch
import torch.nn as nn

class SeqModel(nn.Module):
    def __init__(self, num_layers, hidden_size, num_class=4, embedding=False):
        super().__init__()
        self.embedding = nn.Embedding(100, 2) if embedding else None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear = nn.Linear(hidden_size, num_class)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        raise NotImplementedError('SeqModel is an abstract class')

    def init_internal_states(self, bs):
        raise NotImplementedError('SeqModel is an abstract class')

    def compute_embeddings(self, xs):
        return self.embedding(xs) if self.embedding else []


class LSTM(SeqModel):
    def __init__(self, num_layers, hidden_size, num_class=4, embedding=False):
        super().__init__(num_layers, hidden_size, num_class=num_class, embedding=embedding)
        self.lstm = self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers)

    def init_internal_states(self, bs):
        hidden_state = torch.zeros(self.num_layers, bs, self.hidden_size).to('cuda')
        cell_state = torch.zeros(self.num_layers, bs, self.hidden_size).to('cuda')
        return (hidden_state, cell_state)

    def forward(self, x):
        if self.embedding:
            x = torch.einsum('bs->sb', x)
            x = self.embedding(x)
        else:
            x = torch.einsum('bsi->sbi', x)

        _, (y, _) = self.lstm(x, self.init_internal_states(x.shape[1]))
        return self.linear(y[0])

class GRU(SeqModel):
    def __init__(self, num_layers, hidden_size, num_class=4, embedding=False):
        super().__init__(num_layers, hidden_size, num_class=num_class, embedding=embedding)
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, num_layers=num_layers)

    def init_internal_states(self, bs):
        return torch.zeros(self.num_layers, bs, self.hidden_size).to('cuda')

    def forward(self, x):
        if self.embedding:
            x = torch.einsum('bs->sb', x)
            x = self.embedding(x)
        else:
            x = torch.einsum('bsi->sbi', x)

        _, y = self.gru(x, self.init_internal_states(x.shape[1]))
        return self.linear(y[0])

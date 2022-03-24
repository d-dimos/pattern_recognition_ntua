import torch
import torch.nn as nn

class CustomLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=0):
        super(CustomLSTM, self).__init__()
        # structural info
        self.bidirectional = bidirectional
        self.rnn_size      = rnn_size
        self.num_layers = num_layers

        self.feature_size = 2 * self.rnn_size   if self.bidirectional else self.rnn_size
        self.num_layers   = 2 * self.num_layers if self.bidirectional else self.num_layers

        self.lstm = nn.LSTM(input_dim, rnn_size, num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        # initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.rnn_size).double().to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.rnn_size).double().to(DEVICE)

        # forward pass
        output, _ = self.lstm(x, (h0, c0))
        last_outputs = self.fc(self.last_timestep(output, lengths, self.bidirectional))

        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1).to(DEVICE)
        return outputs.gather(1, idx).squeeze()
        
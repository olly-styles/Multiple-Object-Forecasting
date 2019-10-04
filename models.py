import torch.nn as nn
import torch


class EncoderRNN(nn.Module):
    def __init__(self, device, num_hidden, num_layers):
        super(EncoderRNN, self).__init__()
        self.num_hidden = num_hidden
        self.encoder1 = nn.GRUCell(8, self.num_hidden)
        self.device = device
        self.num_layers = num_layers
        if num_layers > 1:
            self.encoder2 = nn.GRUCell(8, self.num_hidden)
        if num_layers > 2:
            self.encoder3 = nn.GRUCell(8, self.num_hidden)

    def forward(self, input, val=False):
        context = torch.zeros(input.size(0), self.num_hidden, dtype=torch.float).to(self.device)

        for i in range(input.size()[-1]):
            inp = input[:, :, i]
            context = self.encoder1(inp, context)
            if self.num_layers > 1:
                context = self.encoder2(inp, context)
            if self.num_layers > 2:
                context = self.encoder3(inp, context)

        return context


class DecoderRNN(nn.Module):
    def __init__(self, device, num_hidden, dropout_p, num_layers):
        super(DecoderRNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.decoder1 = nn.GRUCell(self.num_hidden, self.num_hidden)
        self.out = nn.Linear(self.num_hidden, 4)
        self.dropout_context = nn.Dropout(p=dropout_p)
        self.dropout_dtp_features = nn.Dropout(p=dropout_p)
        self.relu_context = nn.ReLU()
        self.relu_dtp_features = nn.ReLU()
        self.device = device
        self.context_encoder = nn.Linear(self.num_hidden, int(self.num_hidden / 2))
        self.dtp_encoder = nn.Linear(2048, int(self.num_hidden / 2))
        if num_layers > 1:
            self.decoder2 = nn.GRUCell(self.num_hidden, self.num_hidden)
        if num_layers > 2:
            self.decoder3 = nn.GRUCell(self.num_hidden, self.num_hidden)

    def forward(self, context, dtp_features, val=False):
        outputs = []
        h_t = torch.zeros(context.size(0), self.num_hidden, dtype=torch.float).to(self.device)
        # Fully connected
        context = self.context_encoder(context)
        encoded_dtp_features = self.dtp_encoder(dtp_features)
        # Dropout
        context = self.dropout_context(context)
        encoded_dtp_features = self.dropout_dtp_features(encoded_dtp_features)
        # Relu
        context = self.relu_context(context)
        encoded_dtp_features = self.relu_dtp_features(encoded_dtp_features)
        context = torch.cat((context, encoded_dtp_features), 1)
        # Decode
        for i in range(60):
            h_t = self.decoder1(context, h_t)
            if self.num_layers > 1:
                h_t = self.decoder2(context, h_t)
            if self.num_layers > 2:
                h_t = self.decoder3(context, h_t)
            output = self.out(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 2)
        outputs = outputs.view(-1, 240)
        return outputs

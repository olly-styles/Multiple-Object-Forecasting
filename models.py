import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self,device):
        super(GRU, self).__init__()
        self.gru1 = nn.GRUCell(4, 32)
        self.linear = nn.Linear(32, 4)
        self.device = device
    def forward(self, input,val=False):
        outputs = []
        hiddens = []
        h_t = torch.zeros(input.size(0), 32, dtype=torch.float).to(self.device)
        for i, input_t in enumerate(input.chunk(input.size(2), dim=2)):
            input_t = input_t[:,:,0]
            # Use GT inputs for first 23 steps, then use models own outputs
            if val:
                if i < 29:
                    inp = input_t
                else:
                    inp = output
            else:
                if i < 29:
                    inp = input_t
                else:
                    inp = output

            h_t = self.gru1(inp, h_t)
            output = self.linear(h_t)
            outputs += [output]

            # Hidden state at the final timestep
            if i == 28:
                hiddens += [h_t]
        outputs = torch.stack(outputs, 2)
        return outputs,hiddens

class EncoderRNN(nn.Module):
    def __init__(self,device,num_hidden,num_layers):
        super(EncoderRNN, self).__init__()
        self.num_hidden = num_hidden
        self.encoder1 = nn.GRUCell(8, self.num_hidden)
        self.device = device
        self.num_layers = num_layers
        if num_layers > 1:
            self.encoder2 = nn.GRUCell(8, self.num_hidden)
        if num_layers > 2:
            self.encoder3 = nn.GRUCell(8, self.num_hidden)

    def forward(self, input,val=False):
        outputs = []
        hiddens = []
        context = torch.zeros(input.size(0), self.num_hidden, dtype=torch.float).to(self.device)

        for i in range(input.size()[-1]):
            inp = input[:,:,i]
            context = self.encoder1(inp,context)
            if self.num_layers > 1:
                context = self.encoder2(inp,context)
            if self.num_layers > 2:
                context = self.encoder3(inp,context)

        return context

class DecoderRNN(nn.Module):
    def __init__(self,device,num_hidden,dropout_p,num_layers):
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
        self.context_encoder = nn.Linear(self.num_hidden,int(self.num_hidden/2))
        self.dtp_encoder = nn.Linear(2048,int(self.num_hidden/2))
        if num_layers > 1:
            self.decoder2 = nn.GRUCell(self.num_hidden, self.num_hidden)
        if num_layers > 2:
            self.decoder3 = nn.GRUCell(self.num_hidden, self.num_hidden)

    def forward(self, context,dtp_features,val=False):
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
        context = torch.cat((context,encoded_dtp_features),1)
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
        outputs = outputs.view(-1,240)
        return outputs

class DecoderRNN_BB_ONLY(nn.Module):
    def __init__(self,device,num_hidden,dropout_p,num_layers):
        super(DecoderRNN_BB_ONLY, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.decoder1 = nn.GRUCell(self.num_hidden, self.num_hidden)
        self.out = nn.Linear(self.num_hidden, 4)
        self.dropout_context = nn.Dropout(p=dropout_p)
        self.dropout_dtp_features = nn.Dropout(p=dropout_p)
        self.relu_context = nn.ReLU()
        self.device = device
        self.context_encoder = nn.Linear(self.num_hidden,int(self.num_hidden))
        if num_layers > 1:
            self.decoder2 = nn.GRUCell(self.num_hidden, self.num_hidden)
        if num_layers > 2:
            self.decoder3 = nn.GRUCell(self.num_hidden, self.num_hidden)

    def forward(self, context,dtp_features,val=False):
        outputs = []
        h_t = torch.zeros(context.size(0), self.num_hidden, dtype=torch.float).to(self.device)
        # Fully connected
        context = self.context_encoder(context)
        # Dropout
        context = self.dropout_context(context)
        # Relu
        context = self.relu_context(context)
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
        outputs = outputs.view(-1,240)
        return outputs

class DecoderRNN_DTP_ONLY(nn.Module):
    def __init__(self,device,num_hidden,dropout_p,num_layers):
        super(DecoderRNN_DTP_ONLY, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.decoder1 = nn.GRUCell(self.num_hidden, self.num_hidden)
        self.out = nn.Linear(self.num_hidden, 4)
        self.dropout_dtp_features = nn.Dropout(p=dropout_p)
        self.relu_dtp_features = nn.ReLU()
        self.device = device
        self.dtp_encoder = nn.Linear(2048,int(self.num_hidden))
        if num_layers > 1:
            self.decoder2 = nn.GRUCell(self.num_hidden, self.num_hidden)
        if num_layers > 2:
            self.decoder3 = nn.GRUCell(self.num_hidden, self.num_hidden)

    def forward(self, context,dtp_features,val=False):
        outputs = []
        h_t = torch.zeros(context.size(0), self.num_hidden, dtype=torch.float).to(self.device)
        # Fully connected
        encoded_dtp_features = self.dtp_encoder(dtp_features)
        # Dropout
        encoded_dtp_features = self.dropout_dtp_features(encoded_dtp_features)
        # Relu
        encoded_dtp_features = self.relu_dtp_features(encoded_dtp_features)
        context = encoded_dtp_features
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
        outputs = outputs.view(-1,240)
        return outputs


class EncoderLSTM(nn.Module):
    def __init__(self,device,num_hidden,num_layers):
        super(EncoderLSTM, self).__init__()
        self.num_hidden = num_hidden
        self.encoder1 = nn.LSTMCell(4, self.num_hidden)
        self.device = device
        self.num_layers = num_layers
        if num_layers > 1:
            self.encoder2 = nn.LSTMCell(4, self.num_hidden)
        if num_layers > 2:
            self.encoder3 = nn.LSTMCell(4, self.num_hidden)

    def forward(self, input,val=False):
        outputs = []
        hiddens = []
        context = torch.zeros(input.size(0), self.num_hidden, dtype=torch.float).to(self.device)
        c_t = torch.zeros(context.size(0), self.num_hidden, dtype=torch.float).to(self.device)

        for i in range(input.size()[-1]):
            inp = input[:,:,i]
            context,c_t = self.encoder1(inp,(context,c_t))
            if self.num_layers > 1:
                context,c_t = self.encoder2(inp,(context,c_t))
            if self.num_layers > 2:
                context,c_t = self.encoder3(inp,(context,c_t))

        return context

class DecoderLSTM(nn.Module):
    def __init__(self,device,num_hidden,dropout_p,num_layers):
        super(DecoderLSTM, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.decoder1 = nn.LSTMCell(self.num_hidden, self.num_hidden)
        self.out = nn.Linear(self.num_hidden, 4)
        self.dropout_context = nn.Dropout(p=dropout_p)
        self.dropout_dtp_features = nn.Dropout(p=dropout_p)
        self.relu_context = nn.ReLU()
        self.relu_dtp_features = nn.ReLU()
        self.device = device
        self.context_encoder = nn.Linear(self.num_hidden,int(self.num_hidden/2))
        self.dtp_encoder = nn.Linear(2048,int(self.num_hidden/2))
        if num_layers > 1:
            self.decoder2 = nn.LSTMCell(self.num_hidden, self.num_hidden)
        if num_layers > 2:
            self.decoder3 = nn.LSTMCell(self.num_hidden, self.num_hidden)

    def forward(self, context,dtp_features,val=False):
        outputs = []
        h_t = torch.zeros(context.size(0), self.num_hidden, dtype=torch.float).to(self.device)
        c_t = torch.zeros(context.size(0), self.num_hidden, dtype=torch.float).to(self.device)

        # Fully connected
        context = self.context_encoder(context)
        encoded_dtp_features = self.dtp_encoder(dtp_features)
        # Dropout
        context = self.dropout_context(context)
        encoded_dtp_features = self.dropout_dtp_features(encoded_dtp_features)
        # Relu
        context = self.relu_context(context)
        encoded_dtp_features = self.relu_dtp_features(encoded_dtp_features)
        context = torch.cat((context,encoded_dtp_features),1)
        # Decode
        for i in range(60):
            h_t,c_t = self.decoder1(context, (h_t,c_t))
            if self.num_layers > 1:
                h_t,c_t = self.decoder2(context, (h_t,c_t))
            if self.num_layers > 2:
                h_t,c_t = self.decoder3(context, (h_t,c_t))
            output = self.out(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 2)
        outputs = outputs.view(-1,240)
        return outputs

class GRU_decoder(nn.Module):
    def __init__(self,device):
        super(GRU_decoder, self).__init__()
        self.encoder = nn.GRUCell(4, 128)
        self.relu = nn.ReLU()
        self.gru1 = nn.GRUCell(128, 128)
        self.out = nn.Linear(128, 4)
        self.device = device
    def forward(self, input,val=False):
        outputs = []
        hiddens = []
        context = torch.zeros(input.size(0), 128, dtype=torch.float).to(self.device)

        for i in range(input.size()[-1]):
            inp = input[:,:,i]
            context = self.encoder(inp,context)
        h_t = torch.zeros(input.size(0), 128, dtype=torch.float).to(self.device)

        for i in range(60):
            h_t = self.gru1(context, h_t)
            output = self.out(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 2)
        outputs = outputs.view(-1,240)

        return outputs,hiddens


class SimpleLinear(nn.Module):
    def __init__(self,device):
        super(SimpleLinear, self).__init__()
        self.fc1 = nn.Linear(120, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 240)
        self.device = device
    def forward(self, input):
        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Conv1D(nn.Module):
    def __init__(self,device):
        super(Conv1D, self).__init__()
        # Encoder
        self.conv1l =   nn.Conv1d(in_channels=4,out_channels=32,kernel_size=2)   # Location
        self.bn1 =      nn.BatchNorm1d(32)
        self.conv2 =    nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2)
        self.bn2 =      nn.BatchNorm1d(64)
        self.conv3 =    nn.Conv1d(in_channels=64,out_channels=128,kernel_size=2)
        self.bn3 =      nn.BatchNorm1d(128)
        self.conv4 =    nn.Conv1d(in_channels=128,out_channels=128,kernel_size=1)
        self.bn4 =      nn.BatchNorm1d(128)

        # Concat
        self.conv5 =    nn.Conv1d(in_channels=128,out_channels=128,kernel_size=1)
        self.bn5 =      nn.BatchNorm1d(128)
        self.conv6 =    nn.Conv1d(in_channels=128,out_channels=128,kernel_size=1)
        self.bn6 =      nn.BatchNorm1d(128)

        # Decoder
        self.deconv1 =  nn.ConvTranspose1d(in_channels=128,out_channels=128,kernel_size=9)
        self.bn7 =      nn.BatchNorm1d(128)
        self.deconv2 =  nn.ConvTranspose1d(in_channels=128,out_channels=128,kernel_size=6)
        self.bn8 =      nn.BatchNorm1d(128)
        self.deconv3 =  nn.ConvTranspose1d(in_channels=128,out_channels=64,kernel_size=6)
        self.bn9 =      nn.BatchNorm1d(64)
        self.deconv4 =  nn.ConvTranspose1d(in_channels=64,out_channels=64,kernel_size=6)
        self.bn10 =      nn.BatchNorm1d(64)
        self.deconv5 =  nn.ConvTranspose1d(in_channels=64,out_channels=64,kernel_size=6)
        self.bn11 =      nn.BatchNorm1d(64)
        self.deconv6 =  nn.ConvTranspose1d(in_channels=64,out_channels=32,kernel_size=6)
        self.bn12 =      nn.BatchNorm1d(32)

        self.convfin =    nn.Conv1d(in_channels=32,out_channels=4,kernel_size=1)


    def forward(self, loc):

        # Location encoder
        x = self.conv1l(loc)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.deconv1(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = self.bn8(x)
        x = F.relu(x)

        x = self.deconv3(x)
        x = self.bn9(x)
        x = F.relu(x)

        x = self.deconv4(x)
        x = self.bn10(x)
        x = F.relu(x)

        x = self.deconv5(x)
        x = self.bn11(x)
        x = F.relu(x)

        x = self.deconv6(x)
        x = self.bn12(x)
        x = F.relu(x)

        x = self.convfin(x)

        return x

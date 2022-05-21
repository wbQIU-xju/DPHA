"""
---------------------------------------------------------------------
    Project_name: model
    File Name:  model.py
    Description:
    Author:  qwb
    date:    2021/5/21
---------------------------------------------------------------------
    Change Activity: 22:19
---------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import overlap_and_add
from torch.autograd import Variable
from torch.nn.modules.activation import MultiheadAttention
from model_part.SingleGlu import Adptive_SE_Gated4
from model_part.sfr_layers import SFR
from.model_part.GLN import GlobalLayerNorm
import args_parameter as parser
EPS = 1e-8

class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """

    def __init__(self, W=2, E=64):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.W, self.E = W, E
        # Components
        self.conv1d_U = nn.Conv1d(1, E, kernel_size=W, stride=W // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [B, 1, T], B is batch size, T is #samples
        Returns:
            mixture_w: [B, E, L], where L = (T-W)/(W/2)+1 = 2T/W-1
            L is the number of time steps
        """
        # mixture = torch.unsqueeze(mixture, 1)  # [B, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [B, E, L]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, E, W):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.E, self.W = E, W  # E:enc_dim  W: windows_length
        # Components
        self.basis_signals = nn.Linear(E, W, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [B, E, L]      #B:batch_size ,  E:enc_dim , L: L = (T-W)/(W/2)+1 = 2T/W-1
            est_mask: [B, C, E, L]    # C : numpsk
        Returns:
            est_source: [B, C, T]    #T is #samples
        """
        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask        # [B,1,E,L]*[B, C, E, L] --> [B, C, E, L]
        source_w = torch.transpose(source_w, 2, 3)                 # [B, C, E, L]->[B, C, L, E]
        # S = DV
        est_source = self.basis_signals(source_w)   # [B, C, L, E] -> [B, C, L, W]
        est_source = overlap_and_add(est_source, self.W // 2)
        return est_source


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout,
                                         batch_first=True, bidirectional=bidirectional)
        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        """

        Args:
            input: input shape: batch, seq, dim

        Returns:

        """
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output


# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, output_size, segment_size,
                 dropout=0, num_layers=1, bidirectional=True, n_head=1):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.num_layers = num_layers
        # dual-path RNN
        self.row_rnn1 = nn.ModuleList([])
        self.col_rnn1 = nn.ModuleList([])
        self.row_rnn2 = nn.ModuleList([])
        self.col_rnn2 = nn.ModuleList([])
        self.row_att_norm = nn.ModuleList([])
        self.col_att_norm = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        self.row_attention = nn.ModuleList([])
        self.col_attention = nn.ModuleList([])
        self.row_se = nn.ModuleList([])
        self.col_se = nn.ModuleList([])
        self.linear1 = nn.ModuleList([])
        self.linear2 = nn.ModuleList([])
        self.linear3 = nn.ModuleList([])
        self.linear4 = nn.ModuleList([])
        self.SFR = nn.ModuleList([])
        self.concat_linear = nn.ModuleList([])


        for i in range(num_layers):
            self.row_att_norm.append(GlobalLayerNorm(input_size))
            self.col_att_norm.append(GlobalLayerNorm(input_size))
            self.row_attention.append(MultiheadAttention(input_size, n_head, dropout=0.1))
            self.col_attention.append(MultiheadAttention(input_size, n_head, dropout=0.1))
            self.row_rnn1.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.col_rnn1.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.row_rnn2.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.col_rnn2.append(SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional))
            self.row_se.append(Adptive_SE_Gated4(input_size, 8, 2 * segment_size, 2 * 6))
            self.col_se.append(Adptive_SE_Gated4(input_size, 8, 2 * segment_size, 2 * 6))
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.linear1.append(nn.Linear(2 * input_size, input_size, bias=False))
            self.linear2.append(nn.Linear(2 * input_size, input_size, bias=False))
            self.linear3.append(nn.Linear(2 * input_size, input_size, bias=False))
            self.linear4.append(nn.Linear(2 * input_size, input_size, bias=False))
            self.SFR.append(SFR(in_channels=2 * input_size, out_channels=input_size,
                                kernel_size=1, stride=1, padding=0, dilation=1,
                                groups=16, activation='ReLU', bn_momentum=0.1))
            self.concat_linear.append(nn.Linear((i+2) * input_size, input_size, bias=False))
            self.bn=nn.BatchNorm2d(input_size)
            self.relu = nn.ReLU()
    def forward(self, input):
        """

        Args:
            input: shape: batch, N, dim1, dim2
                  [B, N, L ,K]   dim1=L segment_size   dim2=K   为论文中 S=sqrt(2T)=K
                   apply RNN on dim1 first and then dim2
        Returns:
            output shape: B, output_size, dim1, dim2
        """

        batch_size, _, dim1, dim2 = input.shape
        output = input
        out_data = []
        out_data_2 = []
        out_data.append(output)

        for i in range(self.num_layers):
            row_input = output.permute(0, 3, 2, 1)                                          # B,dim2, dim1, N
            row_input1 = row_input.reshape(batch_size * dim2, dim1, -1)                     # B*dim2, dim1, N
            row_input1 = row_input1.permute(1, 0, 2)                                        # dim1,B*dim2, N
            row_input2 = self.row_att_norm[i](output).permute(0, 3, 2, 1)                   # B,dim2, dim1, N
            row_input2 = row_input2.reshape(batch_size * dim2, dim1, -1)                    # B*dim2, dim1, N
            row_input2 = row_input2.permute(1, 0, 2)                                        # dim1,B*dim2, N
            row_input3 = self.row_attention[i](row_input2, row_input2, row_input2, attn_mask=None,
                                               key_padding_mask=None)[0]
            row_input4 = torch.cat([row_input1, row_input3], dim=2)                         # dim1,B*dim2, 2*N
            row_input4 = self.linear1[i](row_input4)                                        # dim1,B*dim2, N
            row_input5 = row_input4.permute(1, 0, 2)                                        # B*dim2, dim1, N
            row_output1 = self.row_rnn1[i](row_input5)                                      # B*dim2, dim1, N
            row_output2 = self.row_rnn2[i](row_input5)                                      # B*dim2, dim1, N
            row_output = torch.sigmoid(row_output1) * row_output2                           # B*dim2, dim1, N
            row_output3 = torch.cat([row_output, row_input5], dim=2)                        # B*dim2, dim1, 2*N
            row_output = self.linear3[i](row_output3)                                       # B*dim2, dim1, N
            row_output = self.row_se[i](row_output)                                         # B*dim2, dim1, N
            row_output = row_output.reshape(batch_size, dim2, dim1, -1)                     # B,dim2, dim1, N
            row_output = row_output + row_input                                             # B,dim2, dim1, N
            row_output = self.row_norm[i](row_output.permute(0, 3, 2, 1))                   # B,dim2, dim1, N
            output = output + row_output

            col_input = output.permute(0, 2, 3, 1)                                          # B, dim1, dim2, N
            col_input1 = col_input.reshape(batch_size * dim1, dim2, -1)                     # B*dim1,dim2, N
            col_input1 = col_input1.permute(1, 0, 2)                                        # dim2,B*dim1, N
            col_input2 = self.col_att_norm[i](output).permute(0, 2, 3, 1)                   # B, dim1, dim2, N
            col_input2 = col_input2.reshape(batch_size * dim1, dim2, -1)                    # B*dim1, dim2, N
            col_input2 = col_input2.permute(1, 0, 2)                                        # dim2,B*dim1, N
            col_input3 = self.col_attention[i](col_input2, col_input2, col_input2, attn_mask=None,
                                               key_padding_mask=None)[0]                    # dim2,B*dim1, N
            col_input4 = torch.cat([col_input1, col_input3], dim=2)                         # dim2,B*dim1, 2*N
            col_input4 = self.linear2[i](col_input4)                                        # dim2,B*dim1, N
            col_input5 = col_input4.permute(1, 0, 2)                                        # B*dim1,dim2, N
            col_output1 = self.col_rnn1[i](col_input5)                                      # B*dim1,dim2, N
            col_output2 = self.col_rnn2[i](col_input5)                                      # B*dim1,dim2, N
            col_output = torch.sigmoid(col_output1) * col_output2                           # B*dim1,dim2, N
            col_output3 = torch.cat([col_output, col_input5], dim=2)                        # B*dim1,dim2, 2*N
            col_output = self.linear4[i](col_output3)                                       # B*dim1,dim2, N
            col_output = self.col_se[i](col_output)                                         # B*dim1,dim2, N
            col_output = col_output.reshape(batch_size, dim1, dim2, -1)                     # B,dim1,dim2, N
            col_output = col_output + col_input                                             # B,dim1,dim2, N
            col_output = self.col_norm[i](col_output.permute(0, 3, 1, 2))                   # B,dim1,dim2, N
            output = output + col_output                                                    # B, N,dim1,dim2
            if i >= 1 and i <= (self.num_layers - 2):
                out_data.append(output)
                out_put = torch.cat([*[out_data_2[_] for _ in range(i)], out_data[i], output], dim=1)    # [B, N, K, S]
                out_put2 = self.SFR[i](torch.cat([out_data[i], output], dim=1))
                out_data_2.append(out_put2)
                out_put = out_put.permute(0, 2, 3, 1)
                out_put = self.relu(self.bn(self.concat_linear[i](out_put)))
                out_put = out_put.permute(0, 3, 1, 2)
                output = out_put
            elif i == 0:
                out_data.append(output)
                out_put = torch.cat([out_data[0], output], dim=1)  # [B, N, K, S]
                out_put2 = self.SFR[i](torch.cat([out_data[0], output], dim=1))
                out_data_2.append(out_put2)
                out_put = out_put.permute(0, 2, 3, 1)
                out_put = self.relu(self.bn(self.concat_linear[i](out_put)))
                out_put = out_put.permute(0, 3, 1, 2)
                output = out_put
            else:
                out_data.append(output)

        return out_data[1:]


# base module for deep DPRNN
class DPRNN_base(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_spk=2, layer=6,
                 segment_size=250, bidirectional=True, rnn_type='LSTM'):
        super(DPRNN_base, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.input_size = feature_dim
        self.output_size = feature_dim*num_spk
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.eps = 1e-8

        self.DPRNN_base = DPRNN(rnn_type, self.feature_dim, self.hidden_dim,
                                self.feature_dim * self.num_spk,
                                segment_size=self.segment_size, num_layers=layer,
                                bidirectional=bidirectional)
        # output layer
        self.output = nn.ModuleList([])
        for i in range(self.layer):
            self.output.append(nn.Sequential(nn.PReLU(),
                               nn.Conv2d(self.input_size, self.output_size, 1)))

    def forward(self, input):
        output1, output2 = [], []
        output1 = self.DPRNN_base(input)
        for i in range(self.layer):
            output = self.output[i](output1[i])
            output2.append(output)   # B, output_size, dim1, dim2
        return output2


class BF_module(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, num_spk=2, layer=6, segment_size=250,
                 bidirectional=True, rnn_type='GRU'):
        super(BF_module, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)  # [B,E,L,K] ---> [B,N,L,K]

        self.DPRNN_base = DPRNN_base(self.input_dim, self.feature_dim, self.hidden_dim, self.num_spk,
                                     self.layer, self.segment_size, self.bidirectional, self.rnn_type)

        # gated output layer
        self.output = nn.ModuleList([])
        self.output_gate = nn.ModuleList([])

        for i in range(layer):
            self.output.append(nn.Sequential(
                nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                nn.Tanh()
            ))
            self.output_gate.append(nn.Sequential(
                nn.Conv1d(self.feature_dim, self.feature_dim, 1),
                nn.Sigmoid()
            ))


    def pad_segment(self, input, segment_size):

        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(input.type())
        if segment_size == 128:
            input = torch.cat([pad_aux, input, pad_aux, pad_aux], 2)
        else:
            input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        """
           # split the feature into chunks of segment size
        :param input: input is the features: (B, N, L)
        :param segment_size:
        :return:
        """
        input,rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        if segment_size == 128:
            segments1 = input[:, :, :].contiguous().view(batch_size, dim, -1, segment_size)
            segments2 = input[:, :, :].contiguous().view(batch_size, dim, -1, segment_size)
        else:
            segments1 = input[:, :, :-segment_stride].contiguous().view(batch_size, dim, -1, segment_size)
            segments2 = input[:, :, segment_stride:].contiguous().view(batch_size, dim, -1, segment_size)
        segments = torch.cat([segments1, segments2], 3).view(batch_size, dim, -1, segment_size).transpose(2, 3)

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        """
        merge the splitted features into full utterance
        :param input: input is the features: [B, N, K, S]
        :param rest:
        :return:
        """

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = input.transpose(2, 3).contiguous().view(batch_size, dim, -1, segment_size * 2)  # B, N, S, K
        if segment_size == 128:
            input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride*2:-segment_stride]
            input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:-segment_stride*2]
        else:
            input1 = input[:, :, :, :segment_size].contiguous().view(batch_size, dim, -1)[:, :, segment_stride:]
            input2 = input[:, :, :, segment_size:].contiguous().view(batch_size, dim, -1)[:, :, :-segment_stride]

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):
        # input: (B, E, T)
        output1, output2 = [], []
        batch_size, E, seq_length = input.shape
        enc_feature = self.BN(input)  # (B, E, L)-->(B, N, L)
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)
        output1 = self.DPRNN_base(enc_segments)
        for i in range(len(output1)):
            output = output1[i].view(batch_size * self.num_spk, self.feature_dim, self.segment_size, -1)
            output = self.merge_feature(output, enc_rest)  # B*n_spk, N, T
            bf_filter = self.output[i](output) * self.output_gate[i](output)  # B*n_spk, N, T
            bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, self.num_spk, -1,
                                                                    self.feature_dim)  # B, n_spk, T, N
            output2.append(bf_filter)
        return output2


# base module for FaSNet
class FaSNet_base(nn.Module):
    def __init__(self, enc_dim=256, feature_dim=64, hidden_dim=128, layer=2,
                 segment_size=180, nspk=2, win_len=4):
        super(FaSNet_base, self).__init__()

        # parameters
        self.window = win_len
        self.stride = self.window // 2
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size
        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

        self.encoder = Encoder(win_len, enc_dim)  # [B,1,T]-->[B E L]
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)  # [B E L]-->[B E L]
        self.separator = BF_module(self.enc_dim, self.feature_dim, self.hidden_dim, self.num_spk, self.layer,
                                   self.segment_size)
        # [B, N, L] -> [B, E, L]
        self.mask_conv1x1 = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(layer):
            self.mask_conv1x1.append(nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False))
            self.decoder.append(Decoder(enc_dim, win_len))


    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, n, nsample = input.shape
        stride = window // 2
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, n, rest).type(input.type())
            input = torch.cat([input, pad], -1)
        pad_aux = torch.zeros(batch_size, n, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], -1)

        return input, rest

    def forward(self, input):
        """
        input: shape (batch,1,T)
        """
        # pass to a DPRNN
        output, output2 = [], []
        B,c,t= input.size()  # [B,1,T]  # [B,1,T]
        mixture_w = self.encoder(input)  # B, E, L
        score_1 = self.enc_LN(mixture_w)  # B, E, L
        output = self.separator(score_1)  # B, n_spk, T, N
        for i in range(len(output)):
            score_3 = output[i].view(B * self.num_spk, -1, self.feature_dim).transpose(1, 2).contiguous()
            score_4 = self.mask_conv1x1[i](score_3)  # [B*n_spk, N, L] -> [B*n_spk, E, L]
            score = score_4.view(B, self.num_spk, self.enc_dim, -1)  # [B*n_spk, E, L] -> [B, n_spk, E, L]
            est_mask = F.relu(score)
            est_source = self.decoder[i](mixture_w, est_mask)  # [B, E, L] + [B, n_spk, E, L]--> [B, n_spk, T]
            output2.append(est_source)
        return output2


    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None, best_val_loss=None):
        package = {
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
            package['best_val_loss'] = best_val_loss
        return package


def test_net():
    x = torch.rand(2, 32000)
    nnet = FaSNet_base()
    print(nnet)
    x = nnet(x)
    print(x.shape())



if __name__ == "__main__":
     test_net()

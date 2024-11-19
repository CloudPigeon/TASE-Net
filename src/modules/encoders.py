import torch
import torch.nn.functional as F
import time
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig,RobertaModel,RobertaConfig

def add_noise(x, intens=1e-7):
    return x + torch.rand(x.size()) * intens

class LanguageEmbeddingLayer(nn.Module):
    """Embed input text with "glove" or "Bert"
    """
    def __init__(self, hp):
        super(LanguageEmbeddingLayer, self).__init__()
        bertconfig = BertConfig.from_pretrained('/home/u2023111000/MSA/src/premodels/bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('/home/u2023111000/MSA/src/premodels/bert-base-uncased', config=bertconfig)


    def forward(self,bert_sent, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                attention_mask=bert_sent_mask)
        bert_output = bert_output[0]
        return bert_output   # return head (sequence representation)

class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text'):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        y_1 = torch.tanh(self.linear_1(x))
        fusion = self.linear_2(y_1)
        y_2 = torch.tanh(fusion)
        y_3 = self.linear_3(y_2)
        return y_2, y_3


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)
        self.linear_2 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        lengths = lengths.to(torch.int64)
        # x = x.permute(1, 0, 2)
        bs = x.size(0)

        packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
        packed_output, final_states = self.rnn(packed_sequence)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)#得到一个形状为(batch_size, max_sequence_length, num_features)的张量，output张量中的每一行对应于一个序列，每一列对应于一个时间步的输出，而output_lengths告诉你每个序列的实际长度，以便在后续处理中可以忽略填充的部分。
        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0],final_states[0][1]),dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        output = self.linear_2(output)
        return y_1,output


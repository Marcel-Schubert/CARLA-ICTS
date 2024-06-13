import math

import torch
import torch.nn as nn
from torch import Tensor

from P3VI.model import TimeDistributed, LinearReLu


class CI3PP(nn.Module):
    def __init__(self, n_observed_frames, n_predict_frames):
        super(CI3PP, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        self.tdfc_traj = TimeDistributed(LinearReLu(2, 50), batch_first=False)
        self.tdfc_cf = TimeDistributed(LinearReLu(2, 50), batch_first=False)

        self.encoder_traj = nn.GRU(input_size=50, hidden_size=128)
        self.encoder_cf = nn.GRU(input_size=50, hidden_size=128)

        self.pos_enc = PositionalEncoding(128)
        # generate query, keys and values
        self.fc_query_traj = nn.Linear(128, 128)
        self.fc_key_traj = nn.Linear(128, 128)
        self.fc_value_traj = nn.Linear(128, 128)

        self.fc_query_cf = nn.Linear(128, 128)
        self.fc_key_cf = nn.Linear(128, 128)
        self.fc_value_cf = nn.Linear(128, 128)

        # attention network
        self.mha_traj = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        self.mha_cf = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        self.enc_com = nn.GRU(input_size=256, hidden_size=256)

        self.decoder_linear = LinearReLu(256, 128)
        self.decoder_gru = nn.GRU(input_size=128, hidden_size=128)

        self.prediction_head = TimeDistributed(nn.Linear(128, 2))

    def forward(self, x_traj, x_cf):
        td_cf = self.tdfc_cf(x_cf)
        td_traj = self.tdfc_traj(x_traj)

        enc_traj_seq, enc_traj = self.encoder_traj(td_traj)
        enc_cf_seq, enc_cf = self.encoder_cf(td_cf)

        enc_traj_seq = self.pos_enc(enc_traj_seq)
        enc_cf_seq = self.pos_enc(enc_cf_seq)

        query_traj = self.fc_query_traj(enc_traj_seq)
        key_traj = self.fc_key_traj(enc_traj_seq)
        value_traj = self.fc_value_traj(enc_traj_seq)


        query_cf = self.fc_query_cf(enc_cf_seq)
        key_cf = self.fc_key_cf(enc_cf_seq)
        value_cf = self.fc_value_cf(enc_cf_seq)

        traj_attention, _ = self.mha_traj(query_cf, key_traj, value_traj)
        cf_attention, _ = self.mha_cf(query_traj, key_cf, value_cf)

        stacked = torch.cat((traj_attention, cf_attention), dim=-1).sum(dim=0)

        decoded_lin = self.decoder_linear(stacked).repeat(self.n_pred, 1, 1)
        decoded_gru, _ = self.decoder_gru(decoded_lin)

        pred = self.prediction_head(decoded_gru)

        return pred




### https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

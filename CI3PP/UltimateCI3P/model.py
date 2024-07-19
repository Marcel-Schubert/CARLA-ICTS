import math

import torch
import torch.nn as nn
from torch import Tensor

from P3VI.model import TimeDistributed, LinearReLu


class CI3P_CAR(nn.Module):

    def __init__(self, n_observed_frames,
                 n_predict_frames,
                 embed_dim=128):

        super(CI3P_CAR, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        emeding = 64

        # self.emb_traj = TimeDistributed(LinearReLu(2, emeding))
        # self.emb_cf = TimeDistributed(LinearReLu(2, emeding))
        # self.emb_car = TimeDistributed(LinearReLu(2, emeding))

        self.embedder_traj = LinearReLu(2, emeding)
        self.embedder_cf = LinearReLu(2, emeding)
        self.embedder_car = LinearReLu(2, emeding)

        # self.traj_gru = nn.GRU(input_size=2, hidden_size=emeding)
        # self.cf_gru = nn.GRU(input_size=2, hidden_size=emeding)
        # self.car_gru = nn.GRU(input_size=2, hidden_size=emeding)

        self.pos_enc = PositionalEncoding(64, dropout=0.0)

        self.mha_traj_x_cf = nn.MultiheadAttention(embed_dim=emeding, num_heads=4)
        self.mha_traj_x_car = nn.MultiheadAttention(embed_dim=emeding, num_heads=4)

        self.mha_cf_x_traj = nn.MultiheadAttention(embed_dim=emeding, num_heads=4)
        self.mha_cf_x_car = nn.MultiheadAttention(embed_dim=emeding, num_heads=4)

        self.mha_car_x_traj = nn.MultiheadAttention(embed_dim=emeding, num_heads=4)
        self.mha_car_x_cf = nn.MultiheadAttention(embed_dim=emeding, num_heads=4)

        # self.tdfc = TimeDistributed(LinearReLu(emeding * 6, 256))

        self.pos_enc_concat = PositionalEncoding(emeding * 6, dropout=0.0)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emeding * 6, nhead=4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2, norm=nn.LayerNorm(emeding * 6))


        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=emeding * 6, nhead=4)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2, norm=nn.LayerNorm(emeding * 6))



        self.decoder_linear = LinearReLu(emeding * 6, 256)
        self.decoder_gru = nn.GRU(input_size=emeding * 6, hidden_size=128)

        self.prediction_head = nn.Sequential(
            LinearReLu(128, 128),
            LinearReLu(128, 128),
            nn.Linear(128, 2)
        )

    def forward(self, x_traj, x_cf, x_car):

        untouched = x_traj

        # x_traj = self.emb_traj(x_traj)
        # x_cf = self.emb_cf(x_cf)
        # x_car = self.emb_car(x_car)

        # x_traj, _ = self.traj_gru(x_traj)
        # x_cf, _ = self.cf_gru(x_cf)
        # x_car, _ = self.car_gru(x_car)

        x_traj = self.embedder_traj(x_traj)
        x_cf = self.embedder_cf(x_cf)
        x_car = self.embedder_car(x_car)

        x_traj = self.pos_enc(x_traj)
        x_cf = self.pos_enc(x_cf)
        x_car = self.pos_enc(x_car)

        mh_traj_x_cf, _ = self.mha_traj_x_cf(x_traj, x_cf, x_cf)
        mh_traj_x_car, _ = self.mha_traj_x_car(x_traj, x_car, x_car)

        mh_cf_x_traj, _ = self.mha_cf_x_traj(x_cf, x_traj, x_traj)
        mh_cf_x_car, _ = self.mha_cf_x_car(x_cf, x_car, x_car)

        mh_car_x_traj, _ = self.mha_car_x_traj(x_car, x_traj, x_traj)
        mh_car_x_cf, _ = self.mha_car_x_cf(x_car, x_cf, x_cf)

        stacked = torch.cat((mh_traj_x_cf, mh_traj_x_car, mh_cf_x_traj, mh_cf_x_car, mh_car_x_traj, mh_car_x_cf), dim=-1)

        stacked = self.pos_enc_concat(stacked)
        # stacked = self.tdfc(stacked)

        enc = self.encoder(stacked)
        # _, enc = self.encoder_GRU(enc)

        tgt = enc[-1, :, :].repeat((80, 1, 1))

        # dec = self.decoder(tgt, enc)

        # decoded_lin = self.decoder_linear(enc).repeat(self.n_pred, 1, 1)

        decoded_gru, _ = self.decoder_gru(tgt)
        #
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

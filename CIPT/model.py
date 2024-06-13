import torch
import torch.nn as nn

from P3VI.model import TimeDistributed, LinearReLu


class CIPT(nn.Module):

    def __init__(self, n_observed_frames, n_predict_frames):
        super(CIPT, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        self.tdfc_traj = TimeDistributed(LinearReLu(2, 32), batch_first=False)
        self.tdfc_cf = TimeDistributed(LinearReLu(2, 32), batch_first=False)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=64, dropout=0.1, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)

        self.decoder_linear = LinearReLu(64, 64)
        self.decoder_gru = nn.GRU(input_size=64, hidden_size=64)

        self.prediction_head = TimeDistributed(nn.Linear(64, 2))

    def forward(self, x_traj, x_cf):
        td_cf = self.tdfc_cf(x_cf)
        td_traj = self.tdfc_traj(x_traj)

        td_stack = torch.cat((td_traj, td_cf), dim=-1)

        enc = self.transformer_encoder(td_stack)

        decoded_lin = self.decoder_linear(enc).sum(dim=0).repeat(self.n_pred, 1, 1)
        decoded_gru, _ = self.decoder_gru(decoded_lin)

        pred = self.prediction_head(decoded_gru)

        return pred

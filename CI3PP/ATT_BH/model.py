import torch
import torch.nn as nn

from P3VI.model import TimeDistributed, LinearReLu


class CI3P_ATT_BH(nn.Module):

    def __init__(self, n_observed_frames, n_predict_frames):
        super(CI3P_ATT_BH, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        self.tdfc_traj = TimeDistributed(LinearReLu(2, 50), batch_first=False)
        self.tdfc_cf = TimeDistributed(LinearReLu(2, 50), batch_first=False)

        self.encoder_traj = nn.GRU(input_size=50, hidden_size=128)
        self.encoder_cf = nn.GRU(input_size=50, hidden_size=128)

        # attention network
        self.mha_traj = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.mha_cf = nn.MultiheadAttention(embed_dim=128, num_heads=4)

        self.decoder_linear = LinearReLu(256, 128)
        self.decoder_gru = nn.GRU(input_size=128, hidden_size=128)

        self.prediction_head = TimeDistributed(nn.Linear(128, 2))

    def forward(self, x_traj, x_cf):
        td_cf = self.tdfc_cf(x_cf)
        td_traj = self.tdfc_traj(x_traj)

        enc_traj_seq, enc_traj = self.encoder_traj(td_traj)
        enc_cf_seq, enc_cf = self.encoder_cf(td_cf)

        traj_attention, _ = self.mha_traj(enc_cf, enc_traj_seq, enc_traj_seq)
        cf_attention, _ = self.mha_cf(enc_traj, enc_cf_seq, enc_cf_seq)

        stacked = torch.cat((traj_attention, cf_attention), dim=-1)
        decoded_lin = self.decoder_linear(stacked).repeat(self.n_pred, 1, 1)
        decoded_gru, _ = self.decoder_gru(decoded_lin)

        pred = self.prediction_head(decoded_gru)

        return pred

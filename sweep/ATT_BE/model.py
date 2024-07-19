import torch
import torch.nn as nn

from P3VI.model import TimeDistributed, LinearReLu


class CI3P_ATT_BE(nn.Module):

    def __init__(self, n_observed_frames,
                 n_predict_frames,
                 embed_dim=128,
                 n_heads=4):

        super(CI3P_ATT_BE, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        self.tdfc_traj = TimeDistributed(LinearReLu(2, embed_dim), batch_first=False)
        self.tdfc_cf = TimeDistributed(LinearReLu(2, embed_dim), batch_first=False)

        # attention network
        self.mha_traj = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)
        self.mha_cf = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)

        self.encoder_traj = nn.GRU(input_size=embed_dim, hidden_size=embed_dim)
        self.encoder_cf = nn.GRU(input_size=embed_dim, hidden_size=embed_dim)

        self.decoder_linear = LinearReLu(embed_dim*2, 128)
        self.decoder_gru = nn.GRU(input_size=128, hidden_size=128)

        self.prediction_head = TimeDistributed(nn.Linear(128, 2))

    def forward(self, x_traj, x_cf):

        traj_fc = self.tdfc_traj(x_traj)
        cf_fc = self.tdfc_cf(x_cf)

        traj_attention, _ = self.mha_traj(cf_fc, traj_fc, traj_fc)
        cf_attention, _ = self.mha_cf(traj_fc, cf_fc, cf_fc)

        _, enc_traj = self.encoder_traj(traj_attention)
        _, enc_cf = self.encoder_cf(cf_attention)

        stacked = torch.cat((enc_traj, enc_cf), dim=-1)
        decoded_lin = self.decoder_linear(stacked).repeat(self.n_pred, 1, 1)
        decoded_gru, _ = self.decoder_gru(decoded_lin)

        pred = self.prediction_head(decoded_gru)

        return pred

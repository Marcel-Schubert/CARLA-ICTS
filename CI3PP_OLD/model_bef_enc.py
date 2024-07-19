import torch
import torch.nn as nn

from P3VI.model import TimeDistributed, LinearReLu


class CI3PP_BEF_ENC(nn.Module):

    def __init__(self, n_observed_frames, n_predict_frames):
        super(CI3PP_BEF_ENC, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        self.tdfc_traj = TimeDistributed(LinearReLu(2, 50), batch_first=False)
        self.tdfc_cf = TimeDistributed(LinearReLu(2, 50), batch_first=False)

        self.encoder_traj = nn.GRU(input_size=64, hidden_size=128)
        self.encoder_cf = nn.GRU(input_size=64, hidden_size=128)

        # generate query, keys and values
        self.fc_query_traj = nn.Linear(50, 64)
        self.fc_key_traj = nn.Linear(50, 64)
        self.fc_value_traj = nn.Linear(50, 64)

        self.fc_query_cf = nn.Linear(50, 64)
        self.fc_key_cf = nn.Linear(50, 64)
        self.fc_value_cf = nn.Linear(50, 64)

        # attention network
        self.mha_traj = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.mha_cf = nn.MultiheadAttention(embed_dim=64, num_heads=4)

        self.decoder_linear = LinearReLu(256, 128)
        self.decoder_gru = nn.GRU(input_size=128, hidden_size=128)

        self.prediction_head = TimeDistributed(nn.Linear(128, 2))

    def forward(self, x_traj, x_cf):
        td_cf = self.tdfc_cf(x_cf)
        td_traj = self.tdfc_traj(x_traj)

        query_traj = self.fc_query_traj(td_traj)
        key_traj = self.fc_key_traj(td_traj)
        value_traj = self.fc_value_traj(td_traj)

        query_cf = self.fc_query_cf(td_cf)
        key_cf = self.fc_key_cf(td_cf)
        value_cf = self.fc_value_cf(td_cf)

        traj_attention, _ = self.mha_traj(query_cf, key_traj, value_traj)
        cf_attention, _ = self.mha_cf(query_traj, key_cf, value_cf)

        enc_traj_seq, enc_traj = self.encoder_traj(traj_attention)
        enc_cf_seq, enc_cf = self.encoder_cf(cf_attention)

        stacked = torch.cat((enc_traj, enc_cf), dim=-1)
        decoded_lin = self.decoder_linear(stacked).repeat(self.n_pred, 1, 1)
        decoded_gru, _ = self.decoder_gru(decoded_lin)

        pred = self.prediction_head(decoded_gru)

        return pred

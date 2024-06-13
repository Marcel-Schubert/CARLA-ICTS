import torch
import torch.nn as nn

from P3VI.model import TimeDistributed, LinearReLu


class CI3PP(nn.Module):

    def __init__(self, n_observed_frames, n_predict_frames):
        super(CI3PP, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        self.tdfc_traj = TimeDistributed(LinearReLu(2, 50), batch_first=False)
        self.tdfc_cf = TimeDistributed(LinearReLu(2, 50), batch_first=False)

        self.encoder_traj = nn.GRU(input_size=50, hidden_size=64)
        self.encoder_cf = nn.GRU(input_size=50, hidden_size=64)

        # generate query, keys and values
        self.fc_query_traj = nn.Linear(64, 64)
        self.fc_key_traj = nn.Linear(64, 64)
        self.fc_value_traj = nn.Linear(64, 64)

        self.fc_query_cf = nn.Linear(64, 64)
        self.fc_key_cf = nn.Linear(64, 64)
        self.fc_value_cf = nn.Linear(64, 64)

        # attention network
        self.mha_traj = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.mha_cf = nn.MultiheadAttention(embed_dim=64, num_heads=4)

        self.decoder_linear = LinearReLu(128, 64)
        self.decoder_gru = nn.GRU(input_size=64, hidden_size=64)

        self.prediction_head = TimeDistributed(nn.Linear(64, 2))

    def forward(self, x_traj, x_cf):
        td_cf = self.tdfc_cf(x_cf)
        td_traj = self.tdfc_traj(x_traj)

        enc_traj_seq, enc_traj = self.encoder_traj(td_traj)
        enc_cf_seq, enc_cf = self.encoder_cf(td_cf)

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

import torch
import torch.nn as nn

from P3VI.model import TimeDistributed, LinearReLu


class SocialCi3p(nn.Module):

    def __init__(self, n_observed_frames, n_predict_frames):
        super(SocialCi3p, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        embedded = 128

        self.embedder_traj = TimeDistributed(LinearReLu(2, 50), batch_first=True)
        self.embedder_cf = TimeDistributed(LinearReLu(2, 50), batch_first=True)
        self.embedder_car = TimeDistributed(LinearReLu(2, 50), batch_first=True)

        self.traj_gru = nn.GRU(input_size=50, hidden_size=embedded, batch_first=True)
        self.cf_gru = nn.GRU(input_size=50, hidden_size=embedded, batch_first=True)
        self.car_gru = nn.GRU(input_size=50, hidden_size=embedded, batch_first=True)

        self.mha_traj_x_cf = nn.MultiheadAttention(embed_dim=embedded, num_heads=4, batch_first=True)
        self.mha_traj_x_car = nn.MultiheadAttention(embed_dim=embedded, num_heads=4, batch_first=True)

        self.mha_cf_x_traj = nn.MultiheadAttention(embed_dim=embedded, num_heads=4, batch_first=True)
        self.mha_cf_x_car = nn.MultiheadAttention(embed_dim=embedded, num_heads=4, batch_first=True)

        self.mha_car_x_traj = nn.MultiheadAttention(embed_dim=embedded, num_heads=4, batch_first=True)
        self.mha_car_x_cf = nn.MultiheadAttention(embed_dim=embedded, num_heads=4, batch_first=True)

        self.decoder_linear = LinearReLu(embedded*6, 128)
        self.decoder_gru = nn.GRU(input_size=128, hidden_size=128, batch_first=True)

        self.prediction_head = TimeDistributed(nn.Linear(128, 2), batch_first=True)

    def forward(self, x_traj, x_cf, x_car):
        x_traj = self.embedder_traj(x_traj)
        x_cf = self.embedder_cf(x_cf)
        x_car = self.embedder_car(x_car)

        traj_seq, traj_q = self.traj_gru(x_traj)
        cf_seq, cf_q = self.cf_gru(x_cf)
        car_seq, car_q = self.car_gru(x_car)

        traj_q = traj_q.transpose(0, 1)
        cf_q = cf_q.transpose(0, 1)
        car_q = car_q.transpose(0, 1)

        mh_traj_x_cf, _ = self.mha_traj_x_cf(traj_q, cf_seq, cf_seq)
        mh_traj_x_car, _ = self.mha_traj_x_car(traj_q, car_seq, car_seq)

        mh_cf_x_traj, _ = self.mha_cf_x_traj(cf_q, traj_seq, traj_seq)
        mh_cf_x_car, _ = self.mha_cf_x_car(cf_q, car_seq, car_seq)

        mh_car_x_traj, _ = self.mha_car_x_traj(car_q, traj_seq, traj_seq)
        mh_car_x_cf, _ = self.mha_car_x_cf(car_q, cf_seq, cf_seq)

        stacked = torch.cat((mh_traj_x_cf, mh_traj_x_car, mh_cf_x_traj, mh_cf_x_car, mh_car_x_traj, mh_car_x_cf), dim=-1)

        decoded_lin = self.decoder_linear(stacked).repeat(1, self.n_pred, 1)
        decoded_gru, _ = self.decoder_gru(decoded_lin)

        pred = self.prediction_head(decoded_gru)

        return pred

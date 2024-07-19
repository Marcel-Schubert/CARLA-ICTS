"""
Author: Dikshant Gupta
Time: 22.01.22 12:15
"""

import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class LinearTanh(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTanh, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)


class CI3PP_CVAE_ATT_BE(nn.Module):
    def __init__(self, latent_dim=24, predict_frames=20, n_heads=4, embed_dim=64):
        super(CI3PP_CVAE_ATT_BE, self).__init__()
        self.latent_dim = latent_dim
        self.predict_frame_num = predict_frames
        self.input_latent_traj = TimeDistributed(LinearTanh(2, embed_dim))
        self.input_latent_ci = TimeDistributed(LinearTanh(2, embed_dim))

        self.h_traj = nn.GRU(embed_dim, 128)
        self.h_ci = nn.GRU(embed_dim, 128)

        # attention network
        self.mha_traj = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)
        self.mha_cf = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)

        self.y_encoder = nn.Sequential(TimeDistributed(LinearTanh(2, 128)), nn.GRU(128, 256))
        self.mu = nn.Linear(256 + 128*2, self.latent_dim)
        self.var = nn.Linear(256 + 128*2, self.latent_dim)

        self.decoder_linear = LinearTanh(self.latent_dim + 128*2, 128)
        self.decoder_gru = nn.GRU(128, 256)
        self.decoder = TimeDistributed(nn.Linear(256, 2))

    def forward(self, x):
        x1, x2 = x

        # split into trajectory and cognitive information
        tj, ci = x1[:, :, 0:2], x1[:, :, 2:]
        tj = self.input_latent_traj(tj)
        ci = self.input_latent_ci(ci)

        traj_attention, _ = self.mha_traj(ci, tj, tj)
        cf_attention, _ = self.mha_cf(tj, ci, ci)

        enc_traj_seq, enc_traj = self.h_traj(traj_attention)
        enc_cf_seq, enc_cf = self.h_ci(cf_attention)


        x1 = torch.cat((enc_traj, enc_cf), dim=-1)


        _, x2 = self.y_encoder(x2)
        cat_x = torch.cat((x1, x2), dim=-1)
        mean = self.mu(cat_x)
        log_var = self.var(cat_x)
        z = self.sample(mean, log_var)

        decoder_x = torch.cat((x1, z), dim=-1)
        decoder_x = self.decoder_linear(decoder_x)
        decoder_x = decoder_x.repeat(self.predict_frame_num, 1, 1)
        out, _ = self.decoder_gru(decoder_x)
        output = self.decoder(out)
        return output, mean, log_var

    def inference(self, x):
        z = torch.normal(torch.zeros((self.latent_dim,)), torch.ones((self.latent_dim,))).cuda()

        tj, ci = x[:, :, 0:2], x[:, :, 2:]
        tj = self.input_latent_traj(tj)
        ci = self.input_latent_ci(ci)

        traj_attention, _ = self.mha_traj(ci, tj, tj)
        cf_attention, _ = self.mha_cf(tj, ci, ci)

        enc_traj_seq, enc_traj = self.h_traj(traj_attention)
        enc_cf_seq, enc_cf = self.h_ci(cf_attention)

        x1 = torch.cat((enc_traj, enc_cf), dim=-1)

        z = z.unsqueeze(dim=0).unsqueeze(dim=0).repeat(1, x1.shape[1], 1)
        decoder_x = torch.cat((x1, z), dim=-1)
        decoder_x = self.decoder_linear(decoder_x)
        decoder_x = decoder_x.repeat(self.predict_frame_num, 1, 1)
        out, _ = self.decoder_gru(decoder_x)
        output = self.decoder(out)

        return output

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

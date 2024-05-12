import time

import numpy as np
import torch
import torch.nn.functional as F

from CI3PP.model import CI3PP
from P3VI.utils import load_data
from sklearn.model_selection import train_test_split
from datetime import datetime as dt

path_int = "./P3VI/data/int_new_prelim.npy"
path_non_int = "./P3VI/data/non_int_new_prelim.npy"

observed_frame_num = 15
predicting_frame_num = 20
batch_size = 512
epochs = 1000


class CI3PPWrapper:
    def __init__(self, model_path=None,
                 n_obs=observed_frame_num,
                 n_pred=predicting_frame_num):
        self.model = CI3PP(observed_frame_num, predicting_frame_num).cuda()

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.optim = torch.optim.Adam(lr=0.00005, params=self.model.parameters())
        self.n_obs = n_obs
        self.n_pred = n_pred

        self.save_path = (f'./_out/weights/CI3PP_'
                          f'{epochs}e_'
                          f'{batch_size}b_'
                          f'{self.n_obs}obs_'
                          f'{self.n_pred}pred_'
                          f'{dt.today().strftime("%Y-%m-%d_%H-%M-%S")}.pth')

        print(f"Save path will be: {self.save_path}")

    def train(self):
        # load data from files
        obs_train_int, pred_train_int = load_data(path_int, self.n_obs, self.n_pred)
        obs_train_non_int, pred_train_non_int = load_data(path_non_int, self.n_obs, self.n_pred)

        # concat interactive and non-interactive scenarios
        obs_train = np.concatenate((obs_train_int, obs_train_non_int))
        pred_train = np.concatenate((pred_train_int, pred_train_non_int))

        # convert to np array and float32
        input_train = np.array(obs_train[:, :, :], dtype=np.float32)
        output_train = np.array(pred_train[:, :, :], dtype=np.float32)

        # create train test split
        input_train, input_test, output_train, output_test = train_test_split(input_train, output_train, test_size=0.15,
                                                                              random_state=0)

        # make output relative to the last observed frame
        i_t = input_train[:, self.n_obs - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, self.n_pred, axis=1)
        output_train = output_train - i_t

        i_t = input_test[:, self.n_obs - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, self.n_pred, axis=1)
        output_test = output_test - i_t

        # reshape tensors
        input_train = np.transpose(input_train, (1, 0, 2))
        output_train = np.transpose(output_train, (1, 0, 2))
        input_test = np.transpose(input_test, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))

        # eval variables
        count = 0
        best_eval = np.Inf
        best_eval_fde = np.Inf

        # epoch loop:
        for epoch in range(epochs):
            n_batches = int(np.floor(input_train.shape[1] / batch_size))
            print(f"Batches: {n_batches}")
            ckp_loss = 0


            # batch loop:
            t_before = time.time()
            for i in range(n_batches):

                # get current batch:
                x = input_train[:, i * batch_size: i * batch_size + batch_size, :]
                y = output_train[:, i * batch_size: i * batch_size + batch_size, :]
                # convert to tensor:
                x = torch.from_numpy(x).cuda()
                y = torch.from_numpy(y).cuda()

                # slice to trajectory and cognitive information:
                x_traj = x[:, :, 0:2]
                x_cf = x[:, :, 2:]

                # prediction:
                y_pred = self.model(x_traj, x_cf)
                recons_loss = F.mse_loss(y_pred, y)

                self.optim.zero_grad()
                recons_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optim.step()
                ckp_loss += recons_loss.item()

                if i % 100 == 0:
                    print(f"Time Taken for Batch: {(time.time() - t_before)}")
                    t_before = time.time()
                    print("Epoch: {:4}, Batch: {:4} Loss: {:.4f}".format(epoch + 1, i, ckp_loss / 100))
                    ckp_loss = 0
                    eval_loss = 0
                    fde_loss = 0
                    test_batches = int(np.floor(input_test.shape[1] / batch_size))

                    for j in range(test_batches):
                        x = input_test[:, j * batch_size: j * batch_size + batch_size, :]
                        y = output_test[:, j * batch_size: j * batch_size + batch_size, :]
                        x = torch.from_numpy(x).cuda()
                        y = torch.from_numpy(y).cuda()

                        x_traj = x[:, :, 0:2]
                        x_cf = x[:, :, 2:]
                        mse, fde = self.evaluate(x_traj, x_cf, y)
                        eval_loss += mse
                        fde_loss += fde
                    eval_loss /= test_batches * self.n_pred
                    fde_loss /= test_batches
                    if eval_loss < best_eval and fde_loss < best_eval_fde:
                        print(f"Saving Model with loss:{eval_loss, fde_loss}")
                        torch.save(self.model.state_dict(), self.save_path)
                        best_eval = eval_loss
                        best_eval_fde = fde_loss
                    count += 1
    def test(self, path=None):
        input_test, output_test = load_data(path, self.n_obs, self.n_pred)
        input_test = np.array(input_test[:, :, :], dtype=np.float32)
        output_test = np.array(output_test[:, :, :], dtype=np.float32)
        i_t = input_test[:, self.n_obs - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, self.n_pred, axis=1)
        output_test = output_test - i_t

        input_test = np.transpose(input_test, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))

        test_batches = int(np.floor(input_test.shape[1] / batch_size))
        eval_loss = 0
        fde_loss = 0
        for j in range(test_batches):
            x = input_test[:, j * batch_size: j * batch_size + batch_size, :]
            y = output_test[:, j * batch_size: j * batch_size + batch_size, :]
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()

            x_traj = x[:, :, 0:2]
            x_cf = x[:, :, 2:]
            mse, fde = self.evaluate(x_traj, x_cf, y)
            eval_loss += mse
            fde_loss += fde

        eval_loss /= test_batches * self.n_pred * batch_size
        fde_loss /= test_batches * batch_size

        print("MSE", round(eval_loss, 2))
        print("FDE", round(fde_loss, 2))

        return round(eval_loss, 2), round(fde_loss, 2)

    def evaluate(self, x_traj, x_cf, y_test):
        with (torch.no_grad()):
            y_pred = self.model(x_traj, x_cf)
            return torch.square(y_pred - y_test).sum(2).sqrt().sum().item(), torch.square(
                (y_pred[-1, :, :] - y_test[-1, :, :])).sum(1).sqrt().sum().item()

if __name__ == '__main__':
    model = CI3PPWrapper()
    model.train()
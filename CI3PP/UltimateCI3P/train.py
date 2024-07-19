import os
import time
import sys

from matplotlib import pyplot as plt

sys.path.append("/workspace/data/CARLA-ICTS")


from CI3PP.UltimateCI3P.model import CI3P_CAR
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from datetime import datetime as dt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

n_obs = 60
n_pred = 80
epochs = 2000
lr = 0.001
batch_size = 512


# path_non_int = "./P3VI/data/car_dump/all_non_int.npy"
# path_int = "./P3VI/data/car_dump/all_int.npy"
#
# path_int_car = "./P3VI/data/car_dump/all_int_car.npy"
# path_non_int_car = "./P3VI/data/car_dump/all_non_int_car.npy"

path_non_int = "./P3VI/data/car_dump/01_non_int.npy"
path_int = "./P3VI/data/car_dump/01_int.npy"
#
path_int_car = "./P3VI/data/car_dump/01_int_car.npy"
path_non_int_car = "./P3VI/data/car_dump/01_non_int_car.npy"

num_modes = 1


class CI3P_ULTIMATE_WRAPPER():

    def __init__(self, path=None):
        start_time_str = dt.today().strftime("%Y-%m-%d_%H-%M-%S")
        obs_str = f'obs{n_obs}_pred{n_pred}'
        self.base_path = f'./_out/{self.__class__.__name__}/{obs_str}/{start_time_str}'
        os.makedirs(self.base_path, exist_ok=True)

        self.model = CI3P_CAR(
            n_predict_frames=n_pred,
            n_observed_frames=n_obs,
            embed_dim=128
        ).cuda()

        if path is not None:
            self.model.load_state_dict(torch.load(path))

    def split_data(self, path_int, path_non_int, path_int_car, path_non_int_car, n_obs, n_pred):
        # load data from files
        obs_train_int, pred_train_int = self.load_data(path_int, n_obs, n_pred)
        obs_train_non_int, pred_train_non_int = self.load_data(path_non_int, n_obs, n_pred)

        obs_train_int_car, pred_train_int_car = self.load_data(path_int_car, n_obs, n_pred, is_car=True)
        obs_train_non_int_car, pred_train_non_int_car = self.load_data(path_non_int_car, n_obs, n_pred, is_car=True)

        obs_train_int = np.concatenate((obs_train_int, obs_train_int_car), axis=-1)
        obs_train_non_int = np.concatenate((obs_train_non_int, obs_train_non_int_car), axis=-1)
        pred_train_int = np.concatenate((pred_train_int, pred_train_int_car), axis=-1)
        pred_train_non_int = np.concatenate((pred_train_non_int, pred_train_non_int_car), axis=-1)

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
        i_t = input_train[:, n_obs - 1, 0:2]
        i_t = np.concatenate((i_t, input_train[:, n_obs - 1, 4:]), axis=-1)
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, n_pred, axis=1)
        output_train = output_train - i_t

        i_t = input_test[:, n_obs - 1, 0:2]
        i_t = np.concatenate((i_t, input_test[:, n_obs - 1, 4:]), axis=-1)
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, n_pred, axis=1)
        output_test = output_test - i_t

        # reshape tensors
        input_train = np.transpose(input_train, (1, 0, 2))
        output_train = np.transpose(output_train, (1, 0, 2))
        input_test = np.transpose(input_test, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))

        return input_train, input_test, output_train, output_test

    def split_data_test(self, path_int, path_int_car, n_obs, n_pred):
        # load data from files
        obs_train, pred_train = self.load_data(path_int, n_obs, n_pred)
        # obs_train_non_int, pred_train_non_int = self.load_data(path_non_int, n_obs, n_pred)

        obs_train_car, pred_train_car = self.load_data(path_int_car, n_obs, n_pred, is_car=True)
        # obs_train_non_int_car, pred_train_non_int_car = self.load_data(path_non_int_car, n_obs, n_pred, is_car=True)

        obs_train = np.concatenate((obs_train, obs_train_car), axis=-1)
        # obs_train_non_int = np.concatenate((obs_train_non_int, obs_train_non_int_car), axis=-1)
        pred_train = np.concatenate((pred_train, pred_train_car), axis=-1)
        # pred_train_non_int = np.concatenate((pred_train_non_int, pred_train_non_int_car), axis=-1)

        # concat interactive and non-interactive scenarios
        # obs_train = np.concatenate((obs_train_int, obs_train_non_int))
        # pred_train = np.concatenate((pred_train_int, pred_train_non_int))

        # convert to np array and float32
        input_train = np.array(obs_train[:, :, :], dtype=np.float32)
        output_train = np.array(pred_train[:, :, :], dtype=np.float32)

        # create train test split
        input_train, input_test, output_train, output_test = train_test_split(input_train, output_train, test_size=0.15,
                                                                              random_state=0)
        # make output relative to the last observed frame
        i_t = input_train[:, n_obs - 1, 0:2]
        i_t = np.concatenate((i_t, input_train[:, n_obs - 1, 4:]), axis=-1)
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, n_pred, axis=1)
        output_train = output_train - i_t

        i_t = input_test[:, n_obs - 1, 0:2]
        i_t = np.concatenate((i_t, input_test[:, n_obs - 1, 4:]), axis=-1)
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, n_pred, axis=1)
        output_test = output_test - i_t

        # reshape tensors
        input_train = np.transpose(input_train, (1, 0, 2))
        output_train = np.transpose(output_train, (1, 0, 2))
        input_test = np.transpose(input_test, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))

        return input_train, input_test, output_train, output_test

    def load_data(self, file, n_oberserved_frames, n_predict_frames, is_car=False):

        with open(file, 'rb') as f:
            raw = np.load(f, allow_pickle=True)

        enum_conv = lambda t: t.value
        vfunc = np.vectorize(enum_conv)
        if not is_car:
            raw[:, :, 2] = vfunc(raw[:, :, 2])
            raw[:, :, 3] = vfunc(raw[:, :, 3])
        raw = raw.astype(np.double)
        # print(raw[1][0:200])
        window = raw.shape[1] - n_oberserved_frames - n_predict_frames

        observed_data, predict_data = [], []
        for k in range(0, window, 2):
            observed = raw[:, k:n_oberserved_frames + k, :]
            pred = raw[:, k + n_oberserved_frames:n_predict_frames + n_oberserved_frames + k, 0:2]

            observed_data.append(observed)
            predict_data.append(pred)

        observed_data = np.concatenate(observed_data, axis=0)
        predict_data = np.concatenate(predict_data, axis=0)

        return torch.tensor(observed_data).float(), torch.tensor(predict_data).float()

    def train(self):

        input_train, input_test, output_train, output_test = self.split_data(path_int,
                                                                             path_non_int,
                                                                             path_int_car,
                                                                             path_non_int_car,
                                                                             n_obs, n_pred)

        # input_train = np.transpose(input_train, (1, 0, 2))
        # input_test = np.transpose(input_test, (1, 0, 2))
        # output_train = np.transpose(output_train, (1, 0, 2))
        # output_test = np.transpose(output_test, (1, 0, 2))
        #
        # input_train = np.float32(input_train)
        # output_train = np.float32(output_train)
        # input_test = np.float32(input_test)
        # output_test = np.float32(output_test)

        # eval variables
        best_eval = np.Inf
        best_eval_fde = np.Inf
        last_best_epoch = 0
        self.optimiser = optim.Adam(self.model.parameters(), lr=lr,
                                    eps=1e-4)

        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=[5, 10, 15, 20], gamma=0.5,
                                               verbose=True)

        for epoch in range(0, epochs):
            print("Epoch:", epoch)
            did_epoch_better = False

            n_batches = int(np.floor(input_train.shape[1] / batch_size))
            ckp_loss = 0

            # batch loop:
            t_before = time.time()

            for i in range(n_batches):
                # before_prep = time.time()
                print(f"\rBatch: {i:5}/{n_batches}", end="", flush=True)

                ego_in = input_train[:, i * batch_size: i * batch_size + batch_size, :2]
                ego_cf = input_train[:, i * batch_size: i * batch_size + batch_size, 2:4]
                agents_in = input_train[:, i * batch_size: i * batch_size + batch_size, 4:]
                # agents_out = output_train[i * batch_size: i * batch_size + batch_size, :, 2:].unsqueeze(-1)
                ego_out = output_train[:, i * batch_size: i * batch_size + batch_size, :2]


                ego_in = torch.from_numpy(ego_in).float().cuda()
                agents_in = torch.from_numpy(agents_in).float().cuda()
                ego_out = torch.from_numpy(ego_out).float().cuda()
                ego_cf = torch.from_numpy(ego_cf).float().cuda()

                pred_obs = self.model(ego_in,ego_cf, agents_in)

                # ade, fde = self.evaluate(ego_out, pred_obs)
                # recon_loss = nn.MSELoss()(pred_obs, ego_out)
                recon_loss = torch.square(pred_obs - ego_out).sum(2).sqrt().sum()

                self.optimiser.zero_grad()
                recon_loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimiser.step()

                if i % 100 == 0 and i != 0:
                    print(
                        f"\rEpoch: {epoch:4}, Batch: {i:4} Loss: {recon_loss:.6f} Time: {(time.time() - t_before): 4.4f}")
                    t_before = time.time()
                    eval_loss = 0
                    fde_loss = 0

                    test_batches = int(np.floor(input_test.shape[1] / batch_size))
                    self.model.eval()
                    with torch.no_grad():

                        for j in range(test_batches):
                            print(f"\rValidation Batch: {j:5}/{test_batches}", end="", flush=True)
                            ego_in = input_train[:, j * batch_size: j * batch_size + batch_size, :2]
                            ego_cf = input_train[:, j * batch_size: j * batch_size + batch_size, 2:4]
                            agents_in = input_train[:, j * batch_size: j * batch_size + batch_size, 4:]
                            ego_out = output_train[:, j * batch_size: j * batch_size + batch_size, :2]

                            ego_in = torch.from_numpy(ego_in).float().cuda()
                            agents_in = torch.from_numpy(agents_in).float().cuda()
                            ego_out = torch.from_numpy(ego_out).float().cuda()
                            ego_cf = torch.from_numpy(ego_cf).float().cuda()

                            pred_obs = self.model(ego_in, ego_cf, agents_in)

                            ade_loss, fde_loss = self.evaluate(pred_obs, ego_out)
                            eval_loss += ade_loss/n_pred
                            fde_loss += fde_loss


                        eval_loss /= test_batches * batch_size
                        fde_loss /= test_batches * batch_size


                        if eval_loss < best_eval and fde_loss < best_eval_fde:
                            best_eval = eval_loss
                            best_eval_fde = fde_loss
                            did_epoch_better = True
                            print(f"\rSaving Model with loss:{eval_loss:.4f},{fde_loss:.4f}")
                            torch.save(self.model.state_dict(), self.base_path + f"/model_{epoch}.pth")

                    self.model.train()

                if epoch % 5 == 0 and i % 100 == 0 and i != 0 and epoch != 0:
                    plt.plot(ego_out[:, 0, 0].cpu().numpy(), ego_out[:, 0, 1].cpu().numpy())
                    plt.plot(pred_obs[:, 0, 0].cpu().numpy(), pred_obs[:, 0, 1].cpu().numpy())
                    plt.show()


            if did_epoch_better:
                print(f"Epoch {epoch} was better than last best epoch({last_best_epoch})")
                last_best_epoch = epoch
            if epoch - last_best_epoch > 10:
                print(f"Stopping training, no improvement in 10 epochs saved{last_best_epoch}")
                break
            self.optimiser_scheduler.step()




    def test(self, path, path_car, n_obs, n_pred, batch_size_fn):

        _, input_test, _, output_test = self.split_data_test(path,
                                                             path_car,
                                                             n_obs, n_pred)

        input_test = np.transpose(input_test, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))

        input_test = torch.from_numpy(input_test).float().cuda()
        output_test = torch.from_numpy(output_test).float().cuda()

        test_batches = int(np.floor(input_test.shape[0] / batch_size_fn))
        eval_loss = 0
        fde_loss = 0
        # print("Test batches:", test_batches)
        self.model.cuda()
        self.model.eval()
        for j in range(test_batches):
            # print("Batch:", j)
            j = j + 100
            ego_in = input_test[j * batch_size_fn: j * batch_size_fn + batch_size_fn, :, :2]
            agents_in = input_test[j * batch_size_fn: j * batch_size_fn + batch_size_fn, :, 4:]
            # agents_out = output_train[i * batch_size: i * batch_size + batch_size, :, 2:].unsqueeze(-1)
            ego_out = output_test[j * batch_size_fn: j * batch_size_fn + batch_size_fn, :, :2]
            map_lanes = torch.zeros((batch_size_fn, 1, 1))

            ex_mask = torch.ones((ego_in.shape[0], ego_in.shape[1], 1)).cuda()
            ego_in = torch.concatenate((ego_in, ex_mask), dim=-1)
            agents_in = torch.concatenate((agents_in, ex_mask), dim=-1).unsqueeze(-2)

            pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes)

            ade_losses, fde_losses, a, f = self._compute_ego_errors(pred_obs, ego_out, ego_in)

            eval_loss += a / n_pred
            fde_loss += f

        eval_loss /= test_batches * batch_size_fn
        fde_loss /= test_batches * batch_size_fn

        print("MSE", round(eval_loss, 2))
        print("FDE", round(fde_loss, 2))

        return round(eval_loss, 2), round(fde_loss, 2)

    def evaluate(self, ego_out, pred_obs):


            # # make output relative to the last observed frame
            # i_t = x_test[60 - 1:, :, 0:2].detach().cpu().numpy()
            # i_t = np.expand_dims(i_t, axis=1)
            # i_t = np.repeat(i_t, 80, axis=1)
            # i_t = i_t.squeeze(0)
            #
            # output_train = y_test.detach().cpu().numpy() + i_t
            # out_put_pred = y_pred.detach().cpu().numpy() + i_t
            #
            # plt.plot(x_test[:, 0, 0].cpu().numpy(), x_test[:, 0, 1].cpu().numpy())
            # plt.plot(output_train[:, 0, 0], output_train[:, 0, 1])
            # plt.plot(out_put_pred[:, 0, 0], out_put_pred[:, 0, 1])
            # plt.show()

            return torch.square(pred_obs - ego_out).sum(2).sqrt().sum().item(), torch.square(
                (pred_obs[-1, :, :] - ego_out[-1, :, :])).sum(1).sqrt().sum().item()


if __name__ == '__main__':
    wrapper = CI3P_ULTIMATE_WRAPPER()
    wrapper.train()
    # wrapper.train(wrapper.model, wrapper.optimizer, epochs, batch_size, n_obs, n_pred, path_int, path_non_int, save_path, logger, is_cvae=False, is_m2p3=False)
    # wrapper.test(path_int, n_obs, n_pred, batch_size, wrapper.model, is_cvae=False, is_m2p3=False)

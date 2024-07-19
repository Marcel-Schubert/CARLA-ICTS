import os
import time
import sys

from matplotlib import pyplot as plt

sys.path.append("/workspace/data/CARLA-ICTS")

from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from autobots.AutoBots.models.autobot_ego import AutoBotEgo
from datetime import datetime as dt
from autobots.AutoBots.utils.train_helpers import nll_loss_multimodes_joint, nll_loss_multimodes
import numpy as np
import torch

from sklearn.model_selection import train_test_split

n_obs = 60
n_pred = 80
epochs = 2000
lr = 0.001
batch_size = 512
kl_weight = 20.0
entropy_weight = 40.0

# path_non_int = "./P3VI/data/car_dump/all_non_int.npy"
# path_int = "./P3VI/data/car_dump/all_int.npy"
#
# path_int_car = "./P3VI/data/car_dump/all_int_car.npy"
# path_non_int_car = "./P3VI/data/car_dump/all_non_int_car.npy"

# path_int = "./P3VI/data/new_car/01_int_cleaned.npy"
# path_non_int = "./P3VI/data/new_car/01_non_int_cleaned.npy"
#
# path_int_car = "./P3VI/data/new_car/01_int_cleaned_car.npy"
# path_non_int_car = "./P3VI/data/new_car/01_non_int_cleaned_car.npy"


path_int = "./P3VI/data/new_car/all_int.npy"
path_non_int = "./P3VI/data/new_car/all_non_int.npy"

path_int_car = "./P3VI/data/new_car/all_int_car.npy"
path_non_int_car = "./P3VI/data/new_car/all_non_int_car.npy"

num_modes = 1


class AutoBotWrapper:

    def __init__(self, path=None):
        start_time_str = dt.today().strftime("%Y-%m-%d_%H-%M-%S")
        obs_str = f'obs{n_obs}_pred{n_pred}'
        self.base_path = f'./_out/{self.__class__.__name__}/{obs_str}/{start_time_str}'
        os.makedirs(self.base_path, exist_ok=True)

        self.model = AutoBotEgo(
            k_attr=2,
            d_k=128,
            _M=1,
            c=1,
            T=n_pred,
            L_enc=2,
            dropout=0.1,
            num_heads=4,
            L_dec=2,
            tx_hidden_size=128,
            use_map_img=False,
            use_map_lanes=False).cuda()

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

        # enum_conv = lambda t: t.value
        # vfunc = np.vectorize(enum_conv)
        # if not is_car:
        #     raw[:, :, 2] = vfunc(raw[:, :, 2])
        #     raw[:, :, 3] = vfunc(raw[:, :, 3])
        # raw = raw.astype(np.double)
        # # print(raw[1][0:200])
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

    def autobotjoint_train(self):

        input_train, input_test, output_train, output_test = self.split_data(path_int,
                                                                             path_non_int,
                                                                             path_int_car,
                                                                             path_non_int_car,
                                                                             n_obs, n_pred)

        input_train = np.transpose(input_train, (1, 0, 2))
        input_test = np.transpose(input_test, (1, 0, 2))
        output_train = np.transpose(output_train, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))

        input_train = np.float32(input_train)
        output_train = np.float32(output_train)
        input_test = np.float32(input_test)
        output_test = np.float32(output_test)





        # eval variables
        best_eval = np.Inf
        best_eval_fde = np.Inf
        last_best_epoch = 0
        self.optimiser = optim.Adam(self.model.parameters(), lr=lr,
                                    eps=1e-4)
        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=[5, 10, 15, 20], gamma=0.5,
                                               verbose=True)

        steps = 0
        for epoch in range(0, epochs):
            print("Epoch:", epoch)
            # epoch_ade_losses = []
            # epoch_fde_losses = []
            # epoch_mode_probs = []
            did_epoch_better = False

            # self.model.double()
            self.model.cuda()

            n_batches = int(np.floor(input_train.shape[0] / batch_size))
            ckp_loss = 0

            # batch loop:
            t_before = time.time()

            for i in range(n_batches):
                # before_prep = time.time()
                print(f"\rBatch: {i:5}/{n_batches}", end="", flush=True)



                ego_in = input_train[i * batch_size: i * batch_size + batch_size, :, :2]
                agents_in = input_train[i * batch_size: i * batch_size + batch_size, :, 4:]
                # agents_out = output_train[i * batch_size: i * batch_size + batch_size, :, 2:].unsqueeze(-1)
                ego_out = output_train[i * batch_size: i * batch_size + batch_size, :, :2]
                map_lanes = torch.zeros((batch_size, 1, 1))

                # ego_in = np.transpose(ego_in, (1, 0, 2))
                # agents_in = np.transpose(agents_in, (1, 0, 2))
                # agents_out = np.transpose(agents_out, (1, 0, 2))
                # ego_out = np.transpose(ego_out, (1, 0, 2))

                ex_mask = np.ones((ego_in.shape[0], ego_in.shape[1], 1))
                ego_in = np.concatenate((ego_in, ex_mask), axis=-1)
                agents_in = np.concatenate((agents_in, ex_mask), axis=-1)
                # print("Prep time:", time.time() - before_prep)
                #
                ego_in = torch.from_numpy(ego_in).float().cuda()
                agents_in = torch.from_numpy(agents_in).float().cuda().unsqueeze(-2)
                # agents_out = torch.from_numpy(agents_out).cuda().unsqueeze(-1)
                ego_out = torch.from_numpy(ego_out).float().cuda()
                map_lanes = map_lanes.float().cuda()


                # # get current batch:
                # x = input_train[:, i * batch_size: i * batch_size + batch_size, :]
                # y = output_train[:, i * batch_size: i * batch_size + batch_size, :]
                #
                # # convert to tensor:
                # x = torch.from_numpy(x).cuda()
                # y = torch.from_numpy(y).cuda()
                # before_pred = time.time()
                pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes)
                # ego_in, ego_out, agents_in, agents_out, map_lanes, agent_types = self._data_to_device(data)
                # pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes, agent_types)

                nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, ego_out[:, :, :2],
                                                                                   mode_probs,
                                                                                   entropy_weight=entropy_weight,
                                                                                   kl_weight=kl_weight,
                                                                                   use_FDEADE_aux_loss=True)

                self.optimiser.zero_grad()
                (nll_loss + adefde_loss + kl_loss).backward()
                print("NLL", nll_loss.item(), "ADEFDE", adefde_loss.item(), "KL", kl_loss.item())
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimiser.step()
                # print("Pred time:", time.time() - before_pred)

                # self.writer.add_scalar("Loss/nll", nll_loss.item(), steps)
                # self.writer.add_scalar("Loss/adefde", adefde_loss.item(), steps)
                # self.writer.add_scalar("Loss/kl", kl_loss.item(), steps)

                # with torch.no_grad():
                    # ade_losses, fde_losses, a, f = self._compute_ego_errors(pred_obs, ego_out)
                    # epoch_ade_losses.append(ade_losses)
                    # epoch_fde_losses.append(fde_losses)
                    # epoch_mode_probs.append(mode_probs.detach().cpu().numpy())
                # torch.cuda.empty_cache()
                if i % 100 == 0 and i != 0:
                    print(
                        f"\rEpoch: {epoch:4}, Batch: {i:4} Loss: {nll_loss + adefde_loss + kl_loss:.6f} Time: {(time.time() - t_before): 4.4f}")
                    t_before = time.time()
                    eval_loss = 0
                    fde_loss = 0

                    test_batches = int(np.floor(input_test.shape[0] / batch_size))
                    self.model.eval()
                    with torch.no_grad():
                        # val_ade_losses = []
                        # val_fde_losses = []
                        # val_mode_probs = []

                        for j in range(test_batches):
                            print(f"\rValidation Batch: {j:5}/{test_batches}", end="", flush=True)
                            ego_in = input_test[j * batch_size: j * batch_size + batch_size, :, :2]
                            agents_in = input_test[j * batch_size: j * batch_size + batch_size, :, 4:]
                            # agents_out = output_train[i * batch_size: i * batch_size + batch_size, :, 2:].unsqueeze(-1)
                            ego_out = output_test[j * batch_size: j * batch_size + batch_size, :, :2]
                            map_lanes = torch.zeros((batch_size, 1, 1))

                            ex_mask = np.ones((ego_in.shape[0], ego_in.shape[1], 1))
                            ego_in = np.concatenate((ego_in, ex_mask), axis=-1)
                            agents_in = np.concatenate((agents_in, ex_mask), axis=-1)
                            # print("Prep time:", time.time() - before_prep)
                            #
                            ego_in = torch.from_numpy(ego_in).float().cuda()
                            agents_in = torch.from_numpy(agents_in).float().cuda().unsqueeze(-2)
                            # agents_out = torch.from_numpy(agents_out).cuda().unsqueeze(-1)
                            ego_out = torch.from_numpy(ego_out).float().cuda()
                            map_lanes = map_lanes.float().cuda()

                            pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes)

                            ade_losses, fde_losses, a, f = self._compute_ego_errors(pred_obs, ego_out)
                            eval_loss += a / n_pred
                            fde_loss += f
                            # val_ade_losses.append(ade_losses)
                            # val_fde_losses.append(fde_losses)
                            # val_mode_probs.append(mode_probs.detach().cpu().numpy())

                        eval_loss /= test_batches * batch_size
                        fde_loss /= test_batches * batch_size

                        # val_ade_losses = np.concatenate(val_ade_losses)
                        # val_fde_losses = np.concatenate(val_fde_losses)
                        # val_mode_probs = np.concatenate(val_mode_probs)

                        # val_minade_1 = min_xde_K(val_ade_losses, val_mode_probs, K=1)
                        # val_minfde_1 = min_xde_K(val_fde_losses, val_mode_probs, K=1)

                        # print("ADE", eval_loss, "minFDE 1:", val_minfde_1[0])

                        if eval_loss < best_eval and fde_loss < best_eval_fde:
                            best_eval = eval_loss
                            best_eval_fde = fde_loss
                            did_epoch_better = True
                            print(f"\rSaving Model with loss:{eval_loss:.4f},{fde_loss:.4f}")
                            torch.save(self.model.state_dict(), self.base_path + f"/model_{epoch}.pth")

                        # best_eval = min(best_eval, val_minade_1[0])
                        # best_eval_fde = min(best_eval_fde, val_minfde_1[0])
                        # self.model.train()
                        # torch.save(self.model.state_dict(), save_path)
                        # self.save_model(minade_k=val_minade_c[0], minfde_k=val_minfde_c[0])
                    self.model.train()

                steps += 1

            if did_epoch_better:
                print(f"Epoch {epoch} was better than last best epoch({last_best_epoch})")
                last_best_epoch = epoch
            if epoch - last_best_epoch > 10:
                print(f"Stopping training, no improvement in 10 epochs saved{last_best_epoch}")
                break

            # ade_losses = np.concatenate(epoch_ade_losses)
            # fde_losses = np.concatenate(epoch_fde_losses)
            # mode_probs = np.concatenate(epoch_mode_probs)

            # train_minade_c = min_xde_K(ade_losses, mode_probs, K=1)
            # train_minade_10 = min_xde_K(ade_losses, mode_probs, K=min(1, 10))
            # train_minade_5 = min_xde_K(ade_losses, mode_probs, K=min(1, 5))
            # train_minade_1 = min_xde_K(ade_losses, mode_probs, K=1)
            # train_minfde_c = min_xde_K(fde_losses, mode_probs, K=min(1, 10))
            # train_minfde_1 = min_xde_K(fde_losses, mode_probs, K=1)
            # print("Train minADE c:", train_minade_c[0], "Train minADE 1:", train_minade_1[0], "Train minFDE c:",
            #       train_minfde_c[0])

            self.optimiser_scheduler.step()
            # print("Best minADE c", self.smallest_minade_k, "Best minFDE c", self.smallest_minfde_k)

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

    def _compute_ego_errors(self, ego_preds, ego_gt, ego_in=None):
        with (torch.no_grad()):
            ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
            ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1).transpose(0,
                                                                                                                          1).cpu().numpy()
            fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0, 1).cpu().numpy()

            a, f = torch.square(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2]).sum(-1).sqrt().sum().item(), torch.square((ego_preds[:, -1:, :, :2] - ego_gt[:, -1:, :, :2])).sum(-1).sqrt().sum().item()

            # # make output relative to the last observed frame
            # i_t = ego_in[:, 60 - 1:, 0:2].detach().cpu().numpy()
            # i_t = np.expand_dims(i_t, axis=1)
            # i_t = np.repeat(i_t, 80, axis=1)
            # i_t = i_t.squeeze(0)
            #
            # ego_gt = ego_gt[:,:,:,:2].cpu().numpy() + i_t
            # ego_preds = ego_preds[:,:,:,:2].cpu().numpy() + i_t
            #
            # plt.plot(ego_in[0, :,  0].cpu().numpy(), ego_in[0, :,  1].cpu().numpy())
            # plt.plot(ego_gt[0, :, 0, 0], ego_gt[0, :, 0, 1])
            # plt.plot(ego_preds[0, :, 0, 0], ego_preds[0, :, 0, 1])
            # plt.show()

        return ade_losses, fde_losses, a, f


if __name__ == '__main__':
    wrapper = AutoBotWrapper()
    wrapper.autobotjoint_train()
    # wrapper.train(wrapper.model, wrapper.optimizer, epochs, batch_size, n_obs, n_pred, path_int, path_non_int, save_path, logger, is_cvae=False, is_m2p3=False)
    # wrapper.test(path_int, n_obs, n_pred, batch_size, wrapper.model, is_cvae=False, is_m2p3=False)

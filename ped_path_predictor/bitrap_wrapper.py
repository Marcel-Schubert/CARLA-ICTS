import sys
import time



sys.path.append("/workspace/data/CARLA-ICTS")
sys.path.append("/workspace/data/CARLA-ICTS/BiTrap")
sys.path.append("./BiTrap")

from BiTrap.bitrap.utils.scheduler import ParamScheduler, sigmoid_anneal
from BiTrap.bitrap.modeling.bitrap_np import BiTraPNP
from BiTrap.configs import cfg
from P3VI.utils import load_data
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import torch
import datetime

path_int = './P3VI/data/ICTS2_int.npy'
path_non_int = './P3VI/data/ICTS2_non_int.npy'
n_obs = 60
n_pred = 80
epochs = 1000
batch_size = 512


def build_optimizer(cfg, model):
    all_params = model.parameters()
    optimizer = torch.optim.Adam(all_params, lr=cfg.SOLVER.LR)
    return optimizer


def split_data(path_int, path_non_int, n_obs, n_pred):
    # load data from files
    obs_train_int, pred_train_int = load_data(path_int, n_obs, n_pred)
    obs_train_non_int, pred_train_non_int = load_data(path_non_int, n_obs, n_pred)

    # concat interactive and non-interactive scenarios
    obs_train = np.concatenate((obs_train_int, obs_train_non_int))
    pred_train = np.concatenate((pred_train_int, pred_train_non_int))

    # convert to np array and float32
    input_train = np.array(obs_train[:, :, :2], dtype=np.float32)
    output_train = np.array(pred_train[:, :, :], dtype=np.float32)

    # create train test split
    input_train, input_test, output_train, output_test = train_test_split(input_train, output_train, test_size=0.15,
                                                                          random_state=0)
    # make output relative to the last observed frame
    i_t = input_train[:, n_obs - 1, 0:2]
    i_t = np.expand_dims(i_t, axis=1)
    i_t = np.repeat(i_t, n_pred, axis=1)
    output_train = output_train - i_t

    i_t = input_test[:, n_obs - 1, 0:2]
    i_t = np.expand_dims(i_t, axis=1)
    i_t = np.repeat(i_t, n_pred, axis=1)
    output_test = output_test - i_t

    # reshape tensors
    input_train = np.transpose(input_train, (1, 0, 2))
    output_train = np.transpose(output_train, (1, 0, 2))
    input_test = np.transpose(input_test, (1, 0, 2))
    output_test = np.transpose(output_test, (1, 0, 2))

    return input_train, input_test, output_train, output_test


class BiTrapWrapper:
    def __init__(self, model_path=None, observed_frame_num=n_obs,
                 predicting_frame_num=n_pred):
        cfg.merge_from_file('./BiTrap/bitrap_np_ICTS.yml')
        self.model = BiTraPNP(cfg.MODEL, dataset_name=cfg.DATASET.NAME).cuda()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.optim = build_optimizer(cfg, self.model)
        self.observed_frame_num = observed_frame_num
        self.predicting_frame_num = predicting_frame_num
        self.save_path = f'./_out/weights/bitrap_relative_{epochs}e_{batch_size}b_{self.observed_frame_num}obs_{self.predicting_frame_num}pred_{datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.pth'

    def train(self):

        # load data
        input_train, input_test, output_train, output_test = split_data(path_int, path_non_int, n_obs, n_pred)

        self.model.param_scheduler = ParamScheduler()
        self.model.param_scheduler.create_new_scheduler(
            name='kld_weight',
            annealer=sigmoid_anneal,
            annealer_kws={
                'device': cfg.DEVICE,
                'start': 0,
                'finish': 100.0,
                'center_step': 400.0,
                'steps_lo_to_hi': 100.0,
            })

        self.model.param_scheduler.create_new_scheduler(
            name='z_logit_clip',
            annealer=sigmoid_anneal,
            annealer_kws={
                'device': cfg.DEVICE,
                'start': 0.05,
                'finish': 5.0,
                'center_step': 300.0,
                'steps_lo_to_hi': 300.0 / 5.
            })

        count = 0
        best_eval = np.Inf
        best_eval_fde = np.Inf

        last_best_epoch = 0

        with torch.set_grad_enabled(True):
            self.model.train()
            for epoch in range(epochs):
                did_epoch_better = False
                num_batches = int(np.floor(input_train.shape[1] / batch_size))
                print(f"Batches: {num_batches}")
                ckp_loss = 0
                t_before = time.time()
                for i in range(num_batches):
                    print(f"\rBatch: {i:5}/{num_batches}", end="", flush=True)
                    x = input_train[:, i * batch_size: i * batch_size + batch_size, :]
                    y = output_train[:, i * batch_size: i * batch_size + batch_size, :]
                    # transpose to (batch, seq, feature)
                    x = np.transpose(x, (1, 0, 2))
                    y = np.transpose(y, (1, 0, 2))
                    x = torch.from_numpy(x).cuda()
                    y = torch.from_numpy(y).cuda()

                    pred_goal, y_pred, loss_dict, _, _ = self.model(x, target_y=y)

                    loss = loss_dict['loss_goal'] + \
                           loss_dict['loss_traj'] + \
                           self.model.param_scheduler.kld_weight * loss_dict['loss_kld']
                    self.model.param_scheduler.step()
                    loss_dict = {k: v.item() for k, v in loss_dict.items()}
                    loss_dict['lr'] = self.optim.param_groups[0]['lr']

                    # optimize
                    self.optim.zero_grad()  # avoid gradient accumulate from loss.backward()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
                    self.optim.step()

                    # y_pred = y_pred.squeeze()
                    #
                    # recons_loss = F.mse_loss(y_pred, y)

                    # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2), dim=1)

                    # loss = loss_dict['loss_goal'] + \
                    #        loss_dict['loss_traj'] + \
                    #        model.param_scheduler.kld_weight * loss_dict['loss_kld']
                    #
                    # optim.zero_grad()
                    # loss.backward()
                    # optim.step()
                    # ckp_loss += recons_loss.item()
                    # writer.add_scalar("loss", loss.item(), epoch * num_batches + i)
                    if i % 100 == 0 and i != 0:
                        print(
                            f"\rEpoch: {epoch:4}, Batch: {i:4} Loss: {loss:.6f} Time: {(time.time() - t_before): 4.4f}")
                        t_before = time.time()
                        # ckp_loss = 0

                        eval_loss = 0
                        fde_loss = 0
                        test_batches = int(np.floor(input_test.shape[1] / batch_size)/2)
                        self.model.eval()
                        with torch.no_grad():
                            for j in range(test_batches):
                                print(f"\rValidation Batch: {j:5}/{test_batches}", end="", flush=True)

                                # print("Test batch: ", j)
                                x = input_test[:, j * batch_size: j * batch_size + batch_size, :]
                                y = output_test[:, j * batch_size: j * batch_size + batch_size, :]
                                x = np.transpose(x, (1, 0, 2))
                                y = np.transpose(y, (1, 0, 2))
                                x = torch.from_numpy(x).cuda()
                                y = torch.from_numpy(y).cuda()

                                y_pred, mse, fde = self.evaluate(x, y, self.model)
                                eval_loss += mse / n_pred
                                fde_loss += fde
                                # print("x", x[-1,0,:])
                                # print("y", y[0,0,:])
                                # print("y_pred", y_pred[0,0,:],"\n")
                        eval_loss /= test_batches * batch_size
                        fde_loss /= test_batches * batch_size
                        self.model.train()
                        self.optim.zero_grad()
                        if eval_loss < best_eval and fde_loss < best_eval_fde:
                            did_epoch_better = True
                            print(f"\r\033[91mSaving Model with loss:{eval_loss:.4f},{fde_loss:.4f} \033[0m")
                            # save_path = './_out/weights/new_{}_{}_all_seed_0_BiTraP_{}_{}_{}.pth'.format(epochs, batch_size,"best",self.observed_frame_num, self.predicting_frame_num)
                            # print(save_path)
                            torch.save(self.model.state_dict(), self.save_path)
                            best_eval = eval_loss
                            best_eval_fde = fde_loss
                        # writer.add_scalar("eval_loss", eval_loss, count)
                        # writer.add_scalar("fde_loss", fde_loss, count)
                        count += 1
                if did_epoch_better:
                    print(f"Epoch {epoch} was better than last best epoch({last_best_epoch})")
                    last_best_epoch = epoch
                if epoch - last_best_epoch > 10:
                    print(f"Stopping training, no improvement in 10 epochs saved{last_best_epoch}")
                    break

    def evaluate(self, x_test, y_test, model):
        with ((torch.no_grad())):
            # y_pred = model(x_test)
            pred_goal, y_pred, loss_dict, _, _ = model(x_test)
            y_pred = y_pred.squeeze()

            # y_pred = y_pred.squeeze()
            # y_test = torch.transpose(y_test, 0,1)
            # y_pred = torch.transpose(y_pred, 0,1)

            return y_pred, torch.square(y_pred - y_test).sum(2).sqrt().sum().item(),\
                torch.square((y_pred[:, -1, :] - y_test[:, -1, :])).sum(1).sqrt().sum().item()
        # return y_pred, torch.sum(torch.square(y_pred - y_test)).item(), fde(y_pred, y_test)

    def fde(self, y_pred, y_test):
        # print(100*"-")
        # for i in range(128):
        #    print(y_pred[-1,i,:],y_test[-1,i,:])
        # return torch.sum(torch.square((y_pred[-1,:,:] - y_test[-1,:,:]))).item()
        return torch.square((y_pred[:, -1, :] - y_test[:, -1, :])).sum(1).sqrt().sum().item()

    def test(self, path=None):
        input_test, output_test = load_data(path, n_obs, n_pred)
        input_test = np.array(input_test[:, :, 0:2], dtype=np.float32)
        output_test = np.array(output_test[:, :, :], dtype=np.float32)

        # make output relative to the last observed frame
        i_t = input_test[:, n_obs - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, n_pred, axis=1)
        output_test = output_test - i_t

        input_test = np.transpose(input_test, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))
        test_batches = int(np.floor(input_test.shape[1] / batch_size))
        eval_loss = 0
        fde_loss = 0
        for j in range(test_batches):
            x = input_test[:, j * batch_size: j * batch_size + batch_size, :]
            y = output_test[:, j * batch_size: j * batch_size + batch_size, :]
            x = np.transpose(x, (1, 0, 2))
            y = np.transpose(y, (1, 0, 2))
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()

            _, mse, fde = self.evaluate(x, y, self.model)
            eval_loss += mse
            fde_loss += fde
            # print("x", x[-1,0,:])
            # print("y", y[0,0,:])
            # print("y_pred", y_pred[0,0,:],"\n")
        eval_loss /= test_batches * n_pred * batch_size
        fde_loss /= test_batches * batch_size

        print("MSE", round(eval_loss, 2))
        print("FDE", round(fde_loss, 2))

        return round(eval_loss, 2), round(fde_loss, 2)


if __name__ == "__main__":
    bitrap = BiTrapWrapper()
    bitrap.train()

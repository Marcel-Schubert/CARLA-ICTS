import datetime
import sys

from BiTrap.bitrap.utils.scheduler import ParamScheduler, sigmoid_anneal

sys.path.append('./BiTrap')

from BiTrap.bitrap.modeling.bitrap_np import BiTraPNP
from BiTrap.configs import cfg
from P3VI.utils import load_data
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import torch

path_int = './P3VI/data/int_new_prelim.npy'
path_non_int = './P3VI/data/non_int_new_prelim.npy'
observed_frame_num = 60
predicting_frame_num = 80
epochs = 100
batch_size = 512


def build_optimizer(cfg, model):
    all_params = model.parameters()
    optimizer = torch.optim.Adam(all_params, lr=cfg.SOLVER.LR)
    return optimizer


class BiTrapWrapperAbsolute:
    def __init__(self, model_path=None, observed_frame_num=observed_frame_num,
                 predicting_frame_num=predicting_frame_num):
        cfg.merge_from_file('./BiTrap/bitrap_np_ICTS.yml')
        self.model = BiTraPNP(cfg.MODEL, dataset_name=cfg.DATASET.NAME).cuda()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.optim = build_optimizer(cfg, self.model)
        self.observed_frame_num = observed_frame_num
        self.predicting_frame_num = predicting_frame_num
        self.save_path = f'./_out/weights/bitrap_absolute_{epochs}e_{batch_size}b_{self.observed_frame_num}obs_{self.predicting_frame_num}pred_{datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.pth'

    def train(self):
        obs_train_int, pred_train_int = load_data(path_int, observed_frame_num, predicting_frame_num)
        obs_train_non_int, pred_train_non_int = load_data(path_non_int, observed_frame_num,
                                                          predicting_frame_num)

        obs_train = np.concatenate((obs_train_int, obs_train_non_int))
        pred_train = np.concatenate((pred_train_int, pred_train_non_int))
        print(obs_train.shape)
        print(pred_train.shape)
        input_train = np.array(obs_train[:, :, 0:2], dtype=np.float32)
        output_train = np.array(pred_train[:, :, :], dtype=np.float32)
        input_train, input_test, output_train, output_test = train_test_split(input_train, output_train, test_size=0.15,
                                                                              random_state=0)

        # # make output relative to the last observed frame
        # i_t = input_train[:, observed_frame_num - 1, :]
        # i_t = np.expand_dims(i_t, axis=1)
        # i_t = np.repeat(i_t, predicting_frame_num, axis=1)
        # output_train = output_train - i_t
        # print(np.mean(output_train))
        # i_t = input_test[:, observed_frame_num - 1, :]
        # i_t = np.expand_dims(i_t, axis=1)
        # i_t = np.repeat(i_t, predicting_frame_num, axis=1)
        # output_test = output_test - i_t

        # input_train = np.transpose(input_train, (1, 0, 2))
        # output_train = np.transpose(output_train, (1, 0, 2))
        # input_test = np.transpose(input_test, (1, 0, 2))
        # output_test = np.transpose(output_test, (1, 0, 2))
        print("Input train shape =", input_train.shape)
        print("Output train shape =", output_train.shape)

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

        with torch.set_grad_enabled(True):
            self.model.train()
            for epoch in range(epochs):
                num_batches = int(np.floor(input_train.shape[0] / batch_size))
                print(f"Batches: {num_batches}")
                ckp_loss = 0
                for i in range(num_batches):
                    x = input_train[i * batch_size: i * batch_size + batch_size, :,
                        :]  # observed_frame_num x batch_size x 2
                    y = output_train[i * batch_size: i * batch_size + batch_size, :, :]
                    x = torch.from_numpy(x).cuda()
                    y = torch.from_numpy(y).cuda()
                    # print(100*"-")
                    # print(x)
                    # print(y)

                    pred_goal, y_pred, loss_dict, _, _ = self.model(x, target_y=y)
                    y_pred = y_pred.squeeze()
                    loss = loss_dict['loss_goal'] + \
                            loss_dict['loss_traj'] + \
                           self.model.param_scheduler.kld_weight * loss_dict['loss_kld']
                    self.model.param_scheduler.step()
                    loss_dict = {k: v.item() for k, v in loss_dict.items()}
                    loss_dict['lr'] = self.optim.param_groups[0]['lr']
                    recons_loss = F.mse_loss(y_pred, y)
                    loss += recons_loss

                    # optimize
                    self.optim.zero_grad()  # avoid gradient accumulate from loss.backward()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
                    self.optim.step()

                    # y_pred = y_pred.squeeze()
                    #

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
                    if i % 100 == 0:
                        print("Epoch: {}, batch: {} Loss: {:.4f}".format(epoch + 1, i, recons_loss / 10))
                        # ckp_loss = 0

                        eval_loss = 0
                        fde_loss = 0
                        test_batches = int(np.floor(input_test.shape[0] / batch_size))
                        for j in range(test_batches):
                            x = input_test[j * batch_size: j * batch_size + batch_size, :, :]
                            y = output_test[j * batch_size: j * batch_size + batch_size, :, :]
                            x = torch.from_numpy(x).cuda()
                            y = torch.from_numpy(y).cuda()

                            y_pred, mse, fde = self.evaluate(x, y, self.model)
                            eval_loss += mse
                            fde_loss += fde
                            # print("x", x[-1,0,:])
                            # print("y", y[0,0,:])
                            # print("y_pred", y_pred[0,0,:],"\n")
                        eval_loss /= test_batches * predicting_frame_num
                        fde_loss /= test_batches
                        if eval_loss < best_eval and fde_loss < best_eval_fde:
                            # print(save_path)
                            torch.save(self.model.state_dict(), self.save_path)
                            best_eval = eval_loss
                            best_eval_fde = fde_loss
                        # writer.add_scalar("eval_loss", eval_loss, count)
                        # writer.add_scalar("fde_loss", fde_loss, count)
                        count += 1
    def evaluate(self, x_test, y_test, model):
        with torch.no_grad():
            # y_pred = model(x_test)
            pred_goal, y_pred, loss_dict, _, _ = model(x_test)
            y_pred = y_pred.squeeze()
            return y_pred, torch.square(y_pred - y_test).sum(2).sqrt().sum().item(), self.fde(y_pred, y_test)
        # return y_pred, torch.sum(torch.square(y_pred - y_test)).item(), fde(y_pred, y_test)

    def fde(self, y_pred, y_test):
        # print(100*"-")
        # for i in range(128):
        #    print(y_pred[-1,i,:],y_test[-1,i,:])
        # return torch.sum(torch.square((y_pred[-1,:,:] - y_test[-1,:,:]))).item()
        return torch.square((y_pred[-1, :, :] - y_test[-1, :, :])).sum(1).sqrt().sum().item()

    def test(self, path=None):
        input_test, output_test = load_data(path, observed_frame_num, predicting_frame_num)
        input_test = np.array(input_test[:, :, 0:2], dtype=np.float32)
        output_test = np.array(output_test[:, :, :], dtype=np.float32)

        # make output relative to the last observed frame
        # i_t = input_test[:, observed_frame_num - 1, 0:2]
        # i_t = np.expand_dims(i_t, axis=1)
        # i_t = np.repeat(i_t, predicting_frame_num, axis=1)
        # output_test = output_test - i_t

        # input_test = np.transpose(input_test, (1, 0, 2))
        # output_test = np.transpose(output_test, (1, 0, 2))
        test_batches = int(np.floor(input_test.shape[0] / batch_size))
        eval_loss = 0
        fde_loss = 0
        for j in range(test_batches):
            x = input_test[j * batch_size: j * batch_size + batch_size, :, :]
            y = output_test[j * batch_size: j * batch_size + batch_size, :, :]
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()

            _, mse, fde = self.evaluate(x, y, self.model)
            eval_loss += mse
            fde_loss += fde
            # print("x", x[-1,0,:])
            # print("y", y[0,0,:])
            # print("y_pred", y_pred[0,0,:],"\n")
        eval_loss /= test_batches * predicting_frame_num * batch_size
        fde_loss /= test_batches * batch_size

        print("MSE", round(eval_loss, 2))
        print("FDE", round(fde_loss, 2))

        return round(eval_loss, 2), round(fde_loss, 2)

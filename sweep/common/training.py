import time
import sys

import wandb
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append("/workspace/data/CARLA-ICTS")

import numpy as np
import torch
import torch.nn.functional as F

from P3VI.utils import load_data
from sklearn.model_selection import train_test_split


def split_data(path_int, path_non_int, n_obs, n_pred):
    # load data from files
    obs_train_int, pred_train_int = load_data(path_int, n_obs, n_pred)
    obs_train_non_int, pred_train_non_int = load_data(path_non_int, n_obs, n_pred)

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


def train(model,
          optimizer,
          epochs,
          batch_size,
          n_obs,
          n_pred,
          path_int,
          path_non_int,
          save_path,
          logger,
          is_cvae=False,
          is_m2p3=False):
    # load data
    input_train, input_test, output_train, output_test = split_data(path_int, path_non_int, n_obs, n_pred)
    optimiser_scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.5,
                                           verbose=True)
    # eval variables
    best_eval = np.Inf
    best_eval_fde = np.Inf

    last_best_epoch = 0

    # epoch loop:
    for epoch in range(epochs):

        did_epoch_better = False

        n_batches = int(np.floor(input_train.shape[1] / batch_size))
        logger.info(f"Batches: {n_batches}")
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

            if is_cvae:
                if not is_m2p3:
                    y_pred, mu, log_var = model([x, y])
                else:
                    y_pred, mu, log_var = model([x[:, :, 0:2], y])
                recons_loss = F.mse_loss(y_pred, y)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2), dim=1)
                loss = kld_loss + recons_loss
            else:
                # slice to trajectory and cognitive information:
                x_traj = x[:, :, 0:2]
                x_cf = x[:, :, 2:]

                # prediction:
                y_pred = model(x_traj, x_cf)
                recons_loss = F.mse_loss(y_pred, y)

                def l2_loss_fde(pred, data):
                    fde_loss = torch.norm(pred[-1, :, :2] - data[-1:, :, :2], 2, dim=-1).squeeze()
                    ade_loss = torch.norm(pred[:, :, :2] - data[:, :, :2], 2, dim=-1).mean(0)
                    loss = (fde_loss + ade_loss)
                    return 100.0 * loss.mean()

                loss = l2_loss_fde(y_pred, y)

                # ade = torch.square(y_pred - y).sum(2).sqrt().sum().item()
                # loss = recons_loss

            optimizer.zero_grad()
            loss.backward()

            if not is_cvae:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            ckp_loss += recons_loss.item()

            if i % 100 == 0 and i != 0:
                logger.info(
                    f"Epoch: {epoch:4}, Batch: {i:4} Loss: {ckp_loss / 100:.6f} Time: {(time.time() - t_before): 4.4f}")
                t_before = time.time()
                eval_loss = 0
                fde_loss = 0
                test_batches = int(np.floor(input_test.shape[1] / batch_size)/2)

                for j in range(test_batches):
                    x = input_test[:, j * batch_size: j * batch_size + batch_size, :]
                    y = output_test[:, j * batch_size: j * batch_size + batch_size, :]
                    x = torch.from_numpy(x).cuda()
                    y = torch.from_numpy(y).cuda()

                    mse, fde = evaluate(x, y, model, is_cvae, is_m2p3)
                    eval_loss += mse
                    fde_loss += fde

                eval_loss /= test_batches * n_pred * batch_size
                fde_loss /= test_batches * batch_size

                if eval_loss < best_eval and fde_loss < best_eval_fde:
                    did_epoch_better = True
                    logger.info(f"Saving Model with loss: {eval_loss:.4f}, {fde_loss:.4f}")
                    torch.save(model.state_dict(), save_path)
                    best_eval = eval_loss
                    best_eval_fde = fde_loss
                # wandb.log({
                #     "epoch": epoch,
                #     "val_loss": eval_loss,
                #     "ckpt_loss": ckp_loss / 100,
                # })
                ckp_loss = 0

        if did_epoch_better:
            logger.info(f"Epoch {epoch} was better than last best epoch({last_best_epoch})")
            last_best_epoch = epoch
        if epoch - last_best_epoch > 10:
            logger.info(f"Stopping training, no improvement in 10 epochs saved{last_best_epoch}")
            # wandb.log({
            #     "epoch": epoch,
            #     "val_loss": best_eval,
            #     "ckpt_loss": ckp_loss / 100,
            # })
            break
        optimiser_scheduler.step()


def test(path, n_obs, n_pred, batch_size, model, is_cvae=False, is_m2p3=False):

    _, input_test, _, output_test = split_data(path, path, n_obs, n_pred)


    # input_test, output_test = load_data(path, n_obs, n_pred)
    # input_test = np.array(input_test[:, :, :], dtype=np.float32)
    # output_test = np.array(output_test[:, :, :], dtype=np.float32)
    # i_t = input_test[:, n_obs - 1, 0:2]
    # i_t = np.expand_dims(i_t, axis=1)
    # i_t = np.repeat(i_t, n_pred, axis=1)
    # output_test = output_test - i_t
    #
    # input_test = np.transpose(input_test, (1, 0, 2))
    # output_test = np.transpose(output_test, (1, 0, 2))

    test_batches = int(np.floor(input_test.shape[1] / batch_size)/2)
    eval_loss = 0
    fde_loss = 0
    for j in range(test_batches):
        x = input_test[:, j * batch_size: j * batch_size + batch_size, :]
        y = output_test[:, j * batch_size: j * batch_size + batch_size, :]

        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()

        mse, fde = evaluate(x, y, model, is_cvae, is_m2p3)
        eval_loss += mse / n_pred
        fde_loss += fde

    eval_loss /= test_batches * batch_size
    fde_loss /= test_batches * batch_size

    print("MSE", round(eval_loss, 2))
    print("FDE", round(fde_loss, 2))

    return round(eval_loss, 2), round(fde_loss, 2)


def evaluate(x_test, y_test, model, is_cvae=False, is_m2p3=False):
    with torch.no_grad():
        if is_cvae:
            if not is_m2p3:
                y_pred = model.inference(x_test)
            else:
                y_pred = model.inference(x_test[:, :, 0:2])
        else:
            x_traj = x_test[:, :, 0:2]
            x_cf = x_test[:, :, 2:]
            y_pred = model(x_traj, x_cf)

        # make output relative to the last observed frame
        # i_t = x_test[60 - 1:, : , 0:2].detach().cpu().numpy()
        # i_t = np.expand_dims(i_t, axis=1)
        # i_t = np.repeat(i_t, 80, axis=1)
        # i_t = i_t.squeeze(0)


        # output_train = y_test.detach().cpu().numpy() + i_t
        # out_put_pred = y_pred.detach().cpu().numpy() + i_t
        #
        # plt.plot(x_test[:, 0, 0].cpu().numpy(), x_test[:, 0, 1].cpu().numpy())
        # plt.plot(output_train[:, 0, 0], output_train[:, 0, 1])
        # plt.plot(out_put_pred[:, 0, 0], out_put_pred[:, 0, 1])
        # plt.show()

        return torch.square(y_pred - y_test).sum(2).sqrt().sum().item(), torch.square(
            (y_pred[-1, :, :] - y_test[-1, :, :])).sum(1).sqrt().sum().item()

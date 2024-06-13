"""
Author: Dikshant Gupta
Time: 22.01.22 10:57
"""
import os
import time
import sys
sys.path.append("/workspace/data/CARLA-ICTS")

from NotificationService.notification_service import telegram_bot_sendtext
from CI3PP.CVAE_ATT.model import CI3PP_CVAE_ATT
from P3VI.utils import load_data

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


from ped_path_predictor.utils import *
import os
from datetime import datetime as dt


path_int = "./P3VI/data/ICTS2_int.npy"
path_non_int = "./P3VI/data/ICTS2_non_int.npy"
observed_frame_num = 60
predicting_frame_num = 80
batch_size = 1024
epochs = 2000
lr = 0.00005
latent_dim = 24


class CI3PP_CVAE_ATT_WRAPPER:
    def __init__(self, model_path=None,observed_frame_num=observed_frame_num,predicting_frame_num=predicting_frame_num):
        self.model = CI3PP_CVAE_ATT(latent_dim=latent_dim, predict_frames=predicting_frame_num).cuda()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.optim = torch.optim.Adam(lr=lr, params=self.model.parameters())
        self.observed_frame_num = observed_frame_num
        self.predicting_frame_num = predicting_frame_num


        export_dir = './_out/weights/CI3PP_CVAE_ATT'
        self.save_path = (f'{export_dir}/CI3PP_CVAE_ATT'
                          f'{observed_frame_num}o_'
                          f'{predicting_frame_num}p_'
                          f'{epochs}e_' 
                          f'{batch_size}b_'
                          f'{lr}lr_'
                          f'{dt.today().strftime("%Y-%m-%d_%H-%M-%S")}.pth')
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        print(f"Save path will be: {self.save_path}")



    def test(self, test=False, path=None):
        if not test:
            obs_train_int, pred_train_int = load_data(path_int, self.observed_frame_num, self.predicting_frame_num)
            obs_train_non_int, pred_train_non_int = load_data(path_non_int, self.observed_frame_num, self.predicting_frame_num)

            obs_train = np.concatenate((obs_train_int, obs_train_non_int))
            pred_train = np.concatenate((pred_train_int, pred_train_non_int))


            print(obs_train.shape)
            print(pred_train.shape)
            input_train = np.array(obs_train[:, :,:], dtype=np.float32)
            output_train = np.array(pred_train[:, :, :], dtype=np.float32)
            input_train, input_test, output_train, output_test = train_test_split(input_train, output_train, test_size=0.15,random_state=0)
    

            # make output relative to the last observed frame
            i_t = input_train[:, self.observed_frame_num - 1, :]
            i_t = np.expand_dims(i_t, axis=1)
            i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
            output_train = output_train - i_t
            print(np.mean(output_train))
            i_t = input_test[:, self.observed_frame_num - 1, :]
            i_t = np.expand_dims(i_t, axis=1)
            i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
            output_test = output_test - i_t

            input_train = np.transpose(input_train, (1, 0, 2))
            output_train = np.transpose(output_train, (1, 0, 2))
            input_test = np.transpose(input_test, (1, 0, 2))
            output_test = np.transpose(output_test, (1, 0, 2))
            print("Input train shape =", input_train.shape)
            print("Output train shape =", output_train.shape)
        else:
            input_test, output_test = load_data(path, self.observed_frame_num, self.predicting_frame_num)
            input_test = np.array(input_test[:, :, :], dtype=np.float32)
            output_test = np.array(output_test[:, :, :], dtype=np.float32)
            i_t = input_test[:, self.observed_frame_num - 1, 0:2]
            i_t = np.expand_dims(i_t, axis=1)
            i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
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

            _, mse, fde = self.evaluate(x, y)
            eval_loss += mse
            fde_loss += fde
            #print("x", x[-1,0,:])
            #print("y", y[0,0,:])
            #print("y_pred", y_pred[0,0,:],"\n")
        eval_loss /= test_batches * self.predicting_frame_num * batch_size
        fde_loss /= test_batches * batch_size

        print("MSE", round(eval_loss,2))
        print("FDE", round(fde_loss,2))
                    
        return round(eval_loss,2), round(fde_loss,2)
    
    def train(self):
        # load data from files
        obs_train_int, pred_train_int = load_data(path_int, self.observed_frame_num, self.predicting_frame_num)
        obs_train_non_int, pred_train_non_int = load_data(path_non_int, self.observed_frame_num, self.predicting_frame_num)

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
        i_t = input_train[:, self.observed_frame_num - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
        output_train = output_train - i_t

        i_t = input_test[:, self.observed_frame_num - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
        output_test = output_test - i_t

        # reshape tensors
        input_train = np.transpose(input_train, (1, 0, 2))
        output_train = np.transpose(output_train, (1, 0, 2))
        input_test = np.transpose(input_test, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))
        # print("Input train shape =", input_train.shape)
        # print("Output train shape =", output_train.shape)

        count = 0
        best_eval = np.Inf
        best_eval_fde = np.Inf

        last_best_epoch = 0
        telegram_bot_sendtext("CI3P CVAE ATT training started")

        for epoch in range(epochs):

            did_epoch_better = False

            num_batches = int(np.floor(input_train.shape[1] / batch_size))
            ckp_loss = 0
            print("batches: ", num_batches)

            t_before = time.time()

            for i in range(num_batches):
                x = input_train[:, i * batch_size: i * batch_size + batch_size, :] # observed_frame_num x batch_size x 2
                y = output_train[:, i * batch_size: i * batch_size + batch_size, :]
                x = torch.from_numpy(x).cuda()
                y = torch.from_numpy(y).cuda()
                #print(100*"-")
                #print(x)
                #print(y)
                y_pred, mu, log_var = self.model([x, y])
                recons_loss = F.mse_loss(y_pred, y)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2), dim=1)
                #print(y_pred)
                loss = kld_loss + recons_loss
                #print(100*"-")
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                ckp_loss += recons_loss.item()
                if i % 100 == 0:
                    print(f"Time Taken for Batch: {(time.time() - t_before)}")
                    t_before = time.time()

                    print("Epoch: {}, batch: {} Loss: {:.4f}".format(epoch + 1, i, recons_loss / 10))
                    ckp_loss = 0

                    eval_loss = 0
                    fde_loss = 0
                    test_batches = int(np.floor(input_test.shape[1] / batch_size))
                    for j in range(test_batches):
                        x = input_test[:, j * batch_size: j * batch_size + batch_size, :]
                        y = output_test[:, j * batch_size: j * batch_size + batch_size, :]
                        x = torch.from_numpy(x).cuda()
                        y = torch.from_numpy(y).cuda()

                        y_pred, mse, fde = self.evaluate(x, y)
                        eval_loss += mse
                        fde_loss += fde
                        #print("x", x[-1,0,:])
                        #print("y", y[0,0,:])
                        #print("y_pred", y_pred[0,0,:],"\n")
                    eval_loss /= test_batches * self.predicting_frame_num
                    fde_loss /= test_batches
                    if eval_loss < best_eval and fde_loss < best_eval_fde:
                        did_epoch_better = True
                        print(f"Saving Model with loss:{eval_loss, fde_loss}")
                        # save_path = './_out/weights/new_{}_{}_all_seed_0_m2p3_{}_{}_{}.pth'.format(epochs, batch_size,"best",self.observed_frame_num, self.predicting_frame_num)
                        #print(save_path)
                        torch.save(self.model.state_dict(), self.save_path)
                        best_eval = eval_loss
                        best_eval_fde = fde_loss
                    count += 1
            if did_epoch_better:
                print(f"Epoch {epoch} was better")
                last_best_epoch = epoch
            if epoch - last_best_epoch > 10:
                telegram_bot_sendtext(f"CI3P CVAE ATT training stopped, no improvement in 10 epochs at epoch {epoch}\nEvalLoss: {best_eval}\nFDELoss: {best_eval_fde}")
                print(f"Stopping training, no improvement in 10 epochs")
                break

    def evaluate(self, x_test, y_test):
        with torch.no_grad():
            y_pred = self.model.inference(x_test)
            return y_pred, torch.square(y_pred - y_test).sum(2).sqrt().sum().item(), self.fde(y_pred, y_test)
           # return y_pred, torch.sum(torch.square(y_pred - y_test)).item(), self.fde(y_pred, y_test)
        
    def fde(self, y_pred, y_test):
        #print(100*"-")
        #for i in range(128):
        #    print(y_pred[-1,i,:],y_test[-1,i,:])
        #return torch.sum(torch.square((y_pred[-1,:,:] - y_test[-1,:,:]))).item()
        return torch.square((y_pred[-1,:,:] - y_test[-1,:,:])).sum(1).sqrt().sum().item()

    def get_single_prediction(self, x):
        # input of the size (observed_frame_num, 2)
        i_t = x[self.observed_frame_num - 1, :]
        x = np.array(x, dtype=np.float32)
        x = x.reshape((self.observed_frame_num, 1, 2))
        x = torch.from_numpy(x).cuda()
        with torch.no_grad():
            y = self.model.inference(x)

        # y shape = (predicted_frame_num, 1, 2)
        y = y.squeeze().cpu().numpy()
        i_t = np.expand_dims(i_t, axis=0)
        i_t = np.repeat(i_t, self.predicting_frame_num, axis=0)
        y = y + i_t
        return y


if __name__ == "__main__":
    model = CI3PP_CVAE_ATT_WRAPPER()
    model.train()

import sys



sys.path.append("/workspace/data/CARLA-ICTS")
from autobots.AutoBots.models.autobot_ego_cog import AutoBotEgoCog
import logging
import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime as dt
from P3VI.new_util import getDataloaders, singleDatasets
from autobots.AutoBots.models.autobot_ego import AutoBotEgo
from autobots.AutoBots.utils.train_helpers import nll_loss_multimodes

# path_int = "./P3VI/data/new_car/01_int_cleaned.npy"
# path_non_int = "./P3VI/data/new_car/01_non_int_cleaned.npy"
#
# path_int_car = "./P3VI/data/new_car/01_int_cleaned_car.npy"
# path_non_int_car = "./P3VI/data/new_car/01_non_int_cleaned_car.npy"


path_int = "./P3VI/data/new_car/all_int.npy"
path_non_int = "./P3VI/data/new_car/all_non_int.npy"

path_int_car = "./P3VI/data/new_car/all_int_car.npy"
path_non_int_car = "./P3VI/data/new_car/all_non_int_car.npy"


n_obs = 60
n_pred = 80
batch_size = 512
lr = 0.001

kl_weight = 20.0
entropy_weight = 40.0

epoch_limit = 1000


class AutoBotWrapperCogATT:

    def __init__(self, path=None):
        start_time_str = dt.today().strftime("%Y-%m-%d_%H-%M-%S")
        obs_str = f'obs{n_obs}_pred{n_pred}'
        self.base_path = f'./_out/{self.__class__.__name__}/{obs_str}/{start_time_str}'
        os.makedirs(self.base_path, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO,
                            filename=f'{self.base_path}/train.log',
                            format='%(asctime)s %(name)s: %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.logger.info(f"Model: AutoBotWrapperCogATT")
        self.logger.info(f"Observation Frames: {n_obs}")
        self.logger.info(f"Prediction Frames: {n_pred}")
        self.logger.info(f"Batch Size: {batch_size}")
        self.logger.info(f"Initital Learning Rate: {lr}")



        self.model = AutoBotEgoCog(
            k_attr=2,
            d_k=128,
            _M=1,
            c=1,
            T=n_pred,
            L_enc=1,
            dropout=0.0,
            num_heads=16,
            L_dec=1,
            tx_hidden_size=384,
            use_map_img=False,
            use_map_lanes=False).cuda()

        self.optimiser = optim.Adam(self.model.parameters(), lr=lr, eps=1e-4)
        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=[5, 10, 15, 20], gamma=0.5,
                                               verbose=True)

        if path is not None:
            self.model.load_state_dict(torch.load(path))

        self.train_loader, self.test_loader, self.val_loader = getDataloaders(path_int, path_non_int, path_int_car,
                                                                              path_non_int_car, n_obs, n_pred,
                                                                              batch_size=batch_size)

        self.logger.info(f"Train-Batches {len(self.train_loader)}")
        self.logger.info(f"Test-Batches {len(self.test_loader)}")

    def transform(self, x, y):
        ego_in = x[:, :, :2]
        agent_in = x[:, :, 4:]

        cf_as_agent = x[:, :, 2:4].cuda()

        ego_out = y.cuda()

        map_lanes = torch.zeros((batch_size, 1, 1)).cuda()

        ex_mask = torch.ones((ego_in.shape[0], ego_in.shape[1], 1)).float()
        ego_in = torch.concatenate((ego_in, ex_mask), dim=-1).cuda()

        agents_in = (torch.concatenate((agent_in, ex_mask), dim=-1).unsqueeze(-2)).cuda()

        return ego_in, agents_in, map_lanes, ego_out, cf_as_agent

    def _compute_ego_errors(self, ego_preds, ego_gt, ego_in=None):
        with torch.no_grad():
            ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
            ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1),
                                    dim=1).transpose(0,
                                                     1).cpu().numpy()
            fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0,
                                                                                                         1).cpu().numpy()

            a, f = torch.square(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2]).sum(-1).sqrt().sum().item(), torch.square(
                (ego_preds[:, -1:, :, :2] - ego_gt[:, -1:, :, :2])).sum(-1).sqrt().sum().item()

        return ade_losses, fde_losses, a, f

    def train(self):

        # eval variables
        best_eval = np.Inf
        best_eval_fde = np.Inf
        last_best_epoch = 0

        for epoch in range(0, 1000):
            print(f'Epoch {epoch}')
            did_epoch_better = False

            self.model.train()

            t_before = time.time()
            for i, (x, y) in enumerate(self.train_loader):
                print(f'\rBatch {i}/{len(self.train_loader)}', end='')

                ego_in, agents_in, map_lanes, ego_out, cf_in = self.transform(x, y)

                pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes, cf_in)

                nll_loss, kl_loss, post_entropy, ade_fde_loss = nll_loss_multimodes(pred_obs, ego_out[:, :, :2],
                                                                                    mode_probs,
                                                                                    entropy_weight=entropy_weight,
                                                                                    kl_weight=kl_weight,
                                                                                    use_FDEADE_aux_loss=True)

                self.optimiser.zero_grad()
                (nll_loss + ade_fde_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimiser.step()

                if i % 100 == 0:
                    self.logger.info(
                        f"Epoch: {epoch:4}, Batch: {i:4} Loss: {ade_fde_loss:.6f} Time: {(time.time() - t_before): 4.4f}")
                    t_before = time.time()

                    eval_loss, fde_loss = self.eval(self.val_loader)

                    if eval_loss < best_eval and fde_loss < best_eval_fde:
                        best_eval = eval_loss
                        best_eval_fde = fde_loss
                        did_epoch_better = True
                        self.logger.info(f"Saving Model with loss:{eval_loss:.4f},{fde_loss:.4f}")
                        torch.save(self.model.state_dict(), self.base_path + f"/model_{epoch}.pth")
                    self.model.train()

            if did_epoch_better:
                self.logger.info(f"Epoch {epoch} was better than last best epoch({last_best_epoch})")
                last_best_epoch = epoch
            if epoch - last_best_epoch > 10:
                self.logger.info(f"Stopping training, no improvement in 10 epochs saved{last_best_epoch}")
                break
            self.optimiser_scheduler.step()

    def eval(self, dataloader):
        eval_loss = 0
        fde_loss = 0
        self.model.eval()
        with torch.no_grad():
            for j, (x_val, y_val) in enumerate(dataloader):
                print(f'\rBatch {j}/{len(dataloader)}', end='')
                ego_in, agents_in, map_lanes, ego_out, cf = self.transform(x_val, y_val)

                pred_obs, mode_probs = self.model(ego_in, agents_in, map_lanes, cf)

                ade_losses, fde_losses, a, f = self._compute_ego_errors(pred_obs, ego_out)

                eval_loss += a / n_pred
                fde_loss += f
        self.model.train()
        eval_loss /= len(dataloader) * batch_size
        fde_loss /= len(dataloader) * batch_size
        return eval_loss, fde_loss


if __name__ == '__main__':
    # abw = AutoBotWrapperCogATT()
    # abw.train()

    abw_eval = AutoBotWrapperCogATT(path='./_out/AutoBotWrapperCogATT/obs60_pred80/2024-07-07_19-43-32/model_57.pth')

    # dl = singleDatasets(("./P3VI/data/new_car/01_int_cleaned.npy", "./P3VI/data/new_car/01_int_cleaned_car.npy"), 60, 80, 512)
    #
    # print("\nINT-1", abw_eval.eval(dl))
    # dl = singleDatasets(("./P3VI/data/new_car/02_int_cleaned.npy", "./P3VI/data/new_car/02_int_cleaned_car.npy"), 60, 80, 512)
    # print("\nINT-2", abw_eval.eval(dl))
    #
    # dl = singleDatasets(("./P3VI/data/new_car/03_int_cleaned.npy", "./P3VI/data/new_car/03_int_cleaned_car.npy"), 60,
    #                     80, 512)
    # print("\nINT-3", abw_eval.eval(dl))
    #
    # dl = singleDatasets(("./P3VI/data/new_car/04_int_cleaned.npy", "./P3VI/data/new_car/04_int_cleaned_car.npy"), 60,
    #                     80, 512)
    # print("\nINT-4", abw_eval.eval(dl))
    #
    #
    # dl = singleDatasets(("./P3VI/data/new_car/05_int_cleaned.npy", "./P3VI/data/new_car/05_int_cleaned_car.npy"), 60,
    #                     80, 512)
    # print("\nINT-5", abw_eval.eval(dl))
    #
    # dl = singleDatasets(("./P3VI/data/new_car/06_int_cleaned.npy", "./P3VI/data/new_car/06_int_cleaned_car.npy"), 60,
    #                     80, 512)
    # print("\nINT-6", abw_eval.eval(dl))
    #
    # print("\npure test", abw_eval.eval(abw_eval.test_loader))

    dl = singleDatasets(("./P3VI/data/new_car/01_non_int_cleaned.npy", "./P3VI/data/new_car/01_non_int_cleaned_car.npy"), 60, 80, 512)
    print("\nNON_INT-1", abw_eval.eval(dl))

    dl = singleDatasets(("./P3VI/data/new_car/02_non_int_cleaned.npy", "./P3VI/data/new_car/02_non_int_cleaned_car.npy"), 60, 80, 512)
    print("\nNON_INT-2", abw_eval.eval(dl))

    dl = singleDatasets(("./P3VI/data/new_car/03_non_int_cleaned.npy", "./P3VI/data/new_car/03_non_int_cleaned_car.npy"), 60,80, 512)
    print("\nNON_INT-3", abw_eval.eval(dl))

    dl = singleDatasets(("./P3VI/data/new_car/04_non_int_cleaned.npy", "./P3VI/data/new_car/04_non_int_cleaned_car.npy"), 60,80, 512)
    print("\nNON_INT-4", abw_eval.eval(dl))

    dl = singleDatasets(("./P3VI/data/new_car/05_non_int_cleaned.npy", "./P3VI/data/new_car/05_non_int_cleaned_car.npy"), 60,80, 512)
    print("\nNON_INT-5", abw_eval.eval(dl))

    dl = singleDatasets(("./P3VI/data/new_car/06_non_int_cleaned.npy", "./P3VI/data/new_car/06_non_int_cleaned_car.npy"), 60,80, 512)
    print("\nNON_INT-6", abw_eval.eval(dl))

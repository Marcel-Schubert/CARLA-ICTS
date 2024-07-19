import sys
sys.path.append("/workspace/data/CARLA-ICTS")
from sweep.CVAE.model import CI3PP_CVAE
from sweep.common.training import *
from datetime import datetime as dt
from wandb import AlertLevel
import logging
import os
import wandb

n_obs = 15
n_pred = 20
epochs = 2000

path_int = "./P3VI/data/ICTS2_int.npy"
path_non_int = "./P3VI/data/ICTS2_non_int.npy"

def run_sweep():
    run = wandb.init()
    wandb.alert(title='New Sweep Run CI3P CVAE',
                text=f'Starting new sweep run with config: {run.config}',
                level=AlertLevel.INFO)

    lr = wandb.config.lr
    batch_size = wandb.config.batch_size

    model = CI3PP_CVAE(predict_frames=n_pred)
    obs_str = f'obs{n_obs}_pred{n_pred}'
    config_str = f'_batch{batch_size}_lr{lr}'

    #
    #
    #

    start_time_str = dt.today().strftime("%Y-%m-%d_%H-%M-%S")

    base_path = f'./_out/{model.__class__.__name__}/{obs_str}/{config_str}/{start_time_str}'
    os.makedirs(base_path, exist_ok=True)

    logger = logging.getLogger(model.__class__.__name__)
    logging.basicConfig(level=logging.INFO,
                        filename=f'{base_path}/train.log',
                        format='%(asctime)s %(name)s: %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Observation Frames: {n_obs}")
    logger.info(f"Prediction Frames: {n_pred}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {lr}")

    save_path = f'{base_path}/model_{config_str}.pth'
    logger.info(f"Save path will be: {save_path}")

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(model=model,
          optimizer=optimizer,
          epochs=epochs,
          batch_size=batch_size,
          n_obs=n_obs,
          n_pred=n_pred,
          path_int=path_int,
          path_non_int=path_non_int,
          save_path=save_path,
          logger=logger,
          is_cvae=True)
    wandb.alert(title='Sweep Run Completed',
                text=f'Sweep run with config: {run.config} has completed',
                level=AlertLevel.INFO)


if __name__ == '__main__':
    wandb.agent('carla-icts-pagi/CI3P CVAE/hz7mmans', function=run_sweep)

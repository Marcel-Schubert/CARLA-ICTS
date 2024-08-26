import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# from CI3PP.model import CI3PP
from P3VI.utils import load_data
# from CI3PP.train_60_80 import CI3PPWrapper as CI3PPWrapper_60_80
# from ped_path_predictor.m2p3_60_80 import PathPredictor
from ped_path_predictor.model import M2P3

data_ped = "./P3VI/data/walking_cleaned.npy"
data_car = './P3VI/data/walking_cleaned_car.npy'

# data_ped = "./ped_path_predictor/data/new_car/01_int_cleaned.npy"
# data_car = './ped_path_predictor/data/new_car/01_int_cleaned_car.npy'


# data_ped = "./ped_path_predictor/data/fake01.npy"
# data_car = './ped_path_predictor/data/fake01_car.npy'

t_start = 200
t_obs = 60
t_pred = 80

if __name__ == '__main__':
    with open(data_ped, "rb") as f_ped, open(data_car, "rb") as f_car:
        d_ped = np.load(f_ped, allow_pickle=True)
        d_car = np.load(f_car, allow_pickle=True)
    print(d_ped.shape)

    fig, ax = plt.subplots()
    i = 0
    # same scale for x and y axis
    ax.set_aspect(1)
    # observed pedestrian
    ax.plot(d_ped[i,t_start:t_start+t_obs,0], d_ped[i,t_start:t_start+t_obs,1], ".", markevery=4, color="blue")
    # ground truth pedestrian during prediction horizon
    ax.plot(d_ped[i,t_start+t_obs:t_start+t_obs+t_pred,0], d_ped[i,t_start+t_obs:t_start+t_obs+t_pred,1], ".", markevery=4, color="blue", alpha=0.4)
    # observed car
    ax.plot(d_car[i,t_start:t_start+t_obs,0], d_car[i,t_start:t_start+t_obs,1], ".", markevery=4, color="orange")
    # ground truth car during prediction horizon
    ax.plot(d_car[i,t_start+t_obs:t_start+t_obs+t_pred,0], d_car[i,t_start+t_obs:t_start+t_obs+t_pred,1], ".", markevery=4, color="orange", alpha=0.4)
    plt.show()

    print(d_ped[1,:,:])
    

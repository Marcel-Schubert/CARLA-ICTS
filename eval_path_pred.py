import numpy as np
import torch

import train_BiTrap
from BiTrap.bitrap.modeling import BiTraPNP
from BiTrap.configs import cfg
from P3VI.train import P3VIWrapper
from P3VI.utils import load_data
from ped_path_predictor.bitrap_wrapper import BiTrapWrapper
from ped_path_predictor.bitrap_wrapper_absolute import BiTrapWrapperAbsolute
from ped_path_predictor.m2p3 import PathPredictor
import os
import signal

data_paths = [
    # "./P3VI/data/OLD/01_int_test.npy",
    # "./P3VI/data/OLD/02_int_test.npy",
    # "./P3VI/data/OLD/03_int_test.npy",
    # "./P3VI/data/OLD/01_non_int_test.npy",
    # "./P3VI/data/OLD/02_non_int_test.npy",
    # "./P3VI/data/OLD/03_non_int_test.npy",
    # "./P3VI/data/ICTS2_int_new_prelim.npy",
    "./P3VI/data/SINGLE_EXPORT/01_int_prelim.npy",
    "./P3VI/data/SINGLE_EXPORT/02_int_prelim.npy",
    "./P3VI/data/SINGLE_EXPORT/03_int_prelim.npy",
    "./P3VI/data/SINGLE_EXPORT/04_int_prelim.npy",
    "./P3VI/data/SINGLE_EXPORT/05_int_prelim.npy",
    "./P3VI/data/SINGLE_EXPORT/06_int_prelim.npy"
    # "./P3VI/data/OLD/01_non_int_test.npy",
    # "./P3VI/data/OLD/02_non_int_test.npy",
    # "./P3VI/data/OLD/03_non_int_test.npy",
]

m2p3_mse = []
m2p3_fde = []

p3vi_mse_new = []
p3vi_fde_new = []

p3vi_mse = []
p3vi_fde = []

bitrap_mse = []
bitrap_fde = []

for p in data_paths:
    print(20*"#")
    print(p)
    # print("M2P3")
    # #best_60_80.pth",60,80)
    # #best_15_20.pth",15,20)
    # m2p3 = PathPredictor("./_out/weights/m2p3-new-icts2-2000e-512b-60-80.pth",60, 80)
    # mse, fde = m2p3.test(True, p)
    # m2p3_mse.append(mse)
    # m2p3_fde.append(fde)
    # print(20 * "#", "\n")

    print("M2P3")
    #best_60_80.pth",60,80)
    #best_15_20.pth",15,20)
    m2p3 = PathPredictor("./_out/weights/m2p3-new-icts2-2000e-512b-60-80.pth",60, 80)
    mse, fde = m2p3.test(True, p)
    m2p3_mse.append(mse)
    m2p3_fde.append(fde)
    print(20 * "#", "\n")


    #
    # # print("P3VI stock")
    # # p3vi = P3VIWrapper("./_out/weights/p3vi-stock-2000e-256b-15-20.pth", 15, 20)
    # # mse, fde = p3vi.test(True, p)
    # # p3vi_mse.append(mse)
    # # p3vi_fde.append(fde)
    # # print(20 * "#", "\n")

    # print("P3VI new")
    # p3vi = P3VIWrapper("./_out/weights/p3vi-new-2000e-512b-60-80.pth", 60, 80)
    # mse, fde = p3vi.test(True, p)
    # p3vi_mse_new.append(mse)
    # p3vi_fde_new.append(fde)
    # print(20 * "#", "\n")


    print("BiTrap-relative")
    bitrap = BiTrapWrapper(model_path="./_out/weights/bitrap_relative_100_512_60_80.pth", observed_frame_num=60, predicting_frame_num=80)
    mse, fde = bitrap.test(p)
    bitrap_mse.append(mse)
    bitrap_fde.append(fde)
    print(20 * "#", "\n")

    print("BiTrap-absolute")
    bitrap = BiTrapWrapperAbsolute(model_path="./_out/weights/bitrap_absolute_100_512_60_80.pth", observed_frame_num=60,
                           predicting_frame_num=80)
    mse, fde = bitrap.test(p)
    bitrap_mse.append(mse)
    bitrap_fde.append(fde)
    print(20 * "#", "\n")


    # print("P3VI")
    # p3vi = P3VIWrapper("./_out/weights/new_2000_256_all_seed_0_p3vi_best_15_20.pth",60,20)
    # mse, fde = p3vi.test(True,p)
    # p3vi_mse.append(mse)
    # p3vi_fde.append(fde)
    # print(20*"#","\n")

# string = ""
# for mse,fde in zip(m2p3_mse, m2p3_fde):
#     string += "M2P3 " + str(mse) +"/"+str(fde)
# print(string)
#
# string = ""
# for mse,fde in zip(p3vi_mse_new, p3vi_fde_new):
#     string += "P3VI new" + str(mse) +"/"+str(fde)
# print(string)

# string = ""
# for mse,fde in zip(p3vi_mse, p3vi_fde):
#     string += "P3VI stock" + str(mse) +"/"+str(fde)
# print(string)

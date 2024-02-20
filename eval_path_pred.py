from P3VI.train import P3VIWrapper
from ped_path_predictor.m2p3 import PathPredictor
import os
import signal

data_paths = [
    "./P3VI/data/02_non_int_test.npy",

]

m2p3_mse = []
m2p3_fde = []

p3vi_mse = []
p3vi_fde = []

for p in data_paths:
    print(20*"#")
    print(p)
    print("M2P3")
    #best_60_80.pth",60,80)
    #best_15_20.pth",15,20)
    m2p3 = PathPredictor("./_out/weights/new_2000_256_all_seed_0_m2p3_best_60_80.pth",60,20)
    mse, fde = m2p3.test(True,p)
    m2p3_mse.append(mse)
    m2p3_fde.append(fde)
    print()


    print("P3VI")
    p3vi = P3VIWrapper("./_out/weights/new_2000_256_all_seed_0_p3vi_best_15_20.pth",60,20)
    mse, fde = p3vi.test(True,p)
    p3vi_mse.append(mse)
    p3vi_fde.append(fde)
    print(20*"#","\n")

string = ""
for mse,fde in zip(m2p3_mse, m2p3_fde):
    string += "M2P3 " + str(mse) +"/"+str(fde)
print(string)

string = ""
for mse,fde in zip(p3vi_mse, p3vi_fde):
    string += "P3VI" + str(mse) +"/"+str(fde)
print(string)

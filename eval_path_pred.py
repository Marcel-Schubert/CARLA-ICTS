from P3VI.train import P3VIWrapper
from ped_path_predictor.m2p3 import PathPredictor
import os
import signal

data_paths = [
    "paths to data for pp eval"

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
    m2p3 = PathPredictor("your path to m2p3",60,80)
    mse, fde = m2p3.test(True,p)
    m2p3_mse.append(mse)
    m2p3_fde.append(fde)
    print()


    print("P3VI")
    p3vi = P3VIWrapper("your path to p3vi",60,80)
    mse, fde = p3vi.test(True,p)
    p3vi_mse.append(mse)
    p3vi_fde.append(fde)
    print(20*"#","\n")

string = ""
for mse,fde in zip(m2p3_mse, m2p3_fde):
    string += " " + str(mse) +"/"+str(fde) + " &"
print(string)

string = ""
for mse,fde in zip(p3vi_mse, p3vi_fde):
    string += " \\textbf{" + str(mse) +"}/"+"\\textbf{"+str(fde) + "} &"
print(string)

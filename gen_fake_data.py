import os
import numpy as np


def run():

    file = f"./ped_path_predictor/data/fake01.npy"
    car_file = f"./ped_path_predictor/data/fake01_car.npy"
    if not os.path.exists("./ped_path_predictor/data"):
        os.mkdir("./ped_path_predictor/data")

    print(file)
    stand = np.linspace([85,250,1,1], [85,250,1,1], num=60, endpoint=False)
    cross = np.linspace([85,250,1,1], [95,240,1,1], num=188, endpoint=False)
    # walk = np.linspace([240,95,1,1], [232,95,1,1], num=150-60-64)
    walk = np.linspace([95,240,1,1], [95,214,1,1], num=500-60-188)
    ped = np.concatenate((stand, cross, walk))[np.newaxis,:,:]

    # car = np.linspace([300,92], [230,92], num=150)[np.newaxis,:,:]
    car = np.linspace([92,400], [92,200], num=500)[np.newaxis,:,:]
    
    np.save(file, ped, allow_pickle=True)
    np.save(car_file, car, allow_pickle=True)
    print("Fake data gen done!")

if __name__ == '__main__':
    run()

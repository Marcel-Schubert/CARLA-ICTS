import os
import numpy as np

from benchmark.environment import GIDASBenchmark
from config import Config
import time


def run():

    pre_safe_scenarios = [
        # "01_int",
        # "02_int",
        # "03_int",
        # "04_int",
        # "05_int",
        # "06_int",
        # "01_non_int",
        # "02_non_int",
        # "03_non_int",
        # "04_non_int",
        # "05_non_int",
        # "06_non_int",
        "walking"
    ]

    for scenario in pre_safe_scenarios:
        Config.scenarios = [scenario]
        print(Config.scenarios)

        file = f"./P3VI/data/{Config.scenarios[0]}.npy"
        car_file = f"./P3VI/data/{Config.scenarios[0]}_car.npy"
        if not os.path.exists("./P3VI/data"):
            os.mkdir("./P3VI/data")

        print(file)
        # Create environments.
        env = GIDASBenchmark(port=Config.port)
        #agent = SAC(env.world, env.map, env.scene)
        #env.reset_agent(agent)
        #test_env = GIDASBenchmark(port=Config.port + 100, setting="special")
        env.world.random = False
        env.world.dummy_car = True

        env.extract = True
        data = []
        data_car = []
        start_time = time.time()
        # if args.int:
        #     iterations = 2 * len(env.episodes)
        # else:
        #     iterations = len(env.episodes)
        iterations = len(env.episodes) + len(env.test_episodes) + len(env.val_episodes)

        print(iterations)
        for i in range(iterations):
            state = env.reset_extract()
            episode_length = 0

            ep_data = []
            ep_data_car = []

            while episode_length < Config.max_episode_length:

                x,y,icr,son = env.extract_step()
                ep_data.append((x,y,icr,son))

                x_c, y_c = env.extract_car_pos()
                ep_data_car.append((x_c, y_c))
                episode_length+=1

            ep_data = np.array(ep_data)
            ep_data_car = np.array(ep_data_car)
            data.append(ep_data)
            data_car.append(ep_data_car)
            if i % 10 == 0:
                print("Episode:", i)
                print("time taken sofar: ", time.time()-start_time)
            if i % 50 == 0 or i == iterations - 1:
                save_data = np.array(data)
                save_data_car = np.array(data_car)
                np.save(file, save_data, allow_pickle=True)
                np.save(car_file, save_data_car, allow_pickle=True)
                print("Saved",i)

        with open(file,'rb') as f:
            arr = np.load(f, allow_pickle=True)
            print(arr[0])
            print(len(arr))
        with open(car_file,'rb') as f:
            arr = np.load(f, allow_pickle=True)
            print(arr[0])
            print(len(arr))
        env.close()


if __name__ == '__main__':
    run()
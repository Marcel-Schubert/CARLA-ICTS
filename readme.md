# HyREAL: On Hybrid Learning and Reasoning for Explainable and Safe Navigation of Autonomous Cars in Interactive POMDP

The repo contains the code and benchmark CARLA-ICTS accompanying the master thesis, 
*HyREAL: On Hybrid Learning and Reasoning for Explainable and Safe Navigation of Autonomous Cars in Interactive POMDP*.
The code is based on the repository: https://github.com/dikshant2210/Carla-CTS02

## Installation 
#### Install CARLA

See here - https://carla.readthedocs.io/en/0.9.13/start_quickstart/
**Important:** The code is only compatible with CARLA version 0.9.13. 

#### Install python 3.7 (newer version may break the code) and anaconda
The required python packages can be install with:
    `conda create --name <envname> --file req.yaml`.
Alternatively, to build an environmemt from scratch, the code is mostly based on tensorflow, pytorch,
pandas, matplotlib and scipy.

#### IS-DESPOT-p and HyLEAP

1. Install a c++ compiler >=7.3.0, g++ is recommended, and make

    `sudo apt-get install build-essential `
    `sudo apt-get install make`
2. Navigate to the following directories (in that order) and run `make`:
    - `./ISDESPOT/isdespot-ped-pred/is-despot/`
    - `./ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car`
    - `./HyLEAP/smart-car-sim-master/is-despot/`
    - `./HyLEAP/smart-car-sim-master/is-despot/problems/hybridVisual_car`

## Run experiments

### HyREAL-lite
#### Training
To train HyREAL-lite from scratch  on the whole benchmark execute the following line,
>`python train_hyreal_a2c.py --server --cuda --port=<port> --config=<config>`.

The argument `--server` specifies to use the carla config for the server,
while `--cuda` specifies that the RL-agent is run on GPU. The carla port is configured
using `--port` and the configuration for the RL-agent is passed via `--config`.

#### Testing

To test HyREAL-lite for a specific scenario `scenario` $\in$ `[01_int, 02_int, 03_int, 01_non_int, 02_non_int, 03_non_int]`, run the following command,
> `python eval_learner.py --server --port=<port> --test=scenario --agent=hyreal`

In order to execute all scenarios at once use th following command,
> `python eval_learner.py --server --port=<port> --test=all --agent=hyreal`


### A2C-CADRL
#### Training
To train A2C-CADRL from scratch  on the whole benchmark execute the following line,
>`python train_c2a.py --server --cuda --port=<port> --config=<config>`.

The argument `--server` specifies to use the carla config for the server,
while `--cuda` specifies that the RL-agent is run on GPU. The carla port is configured
using `--port` and the configuration for the RL-agent is passed via `--config`.

#### Testing

To test A2C-CADRL for a specific scenario `scenario` $\in$ `[01_int, 02_int, 03_int, 01_non_int, 02_non_int, 03_non_int]`, run the following command,
> `python eval_learner.py --server --port=<port> --test=scenario --agent=a2c`

In order to execute all scenarios at once use th following command,
> `python eval_learner.py --server --port=<port> --test=all --agent=a2c`

### NavSAC 
#### Training
To train NavSAC from scratch run,
> `python3 train_sac.py --server --cuda --port=<port> config=<config>`.

#### Testing

To test NavSAC for a specific scenario `scenario` $\in$ `[01_int, 02_int, 03_int, 01_non_int, 02_non_int, 03_non_int]`, run the following command,
> `python eval_sac.py --server --port=<port> --test=scenario`

In order to execute all scenarios at once use th following command,
> `python eval_sac.py --server --port=<port> --test=all`

### HyLEAP
### Training
HyLEAP can be trained by running,
> `python train_hyleap.py --port=<port>`.

The parameters can be adjusted in `hyleap/hybrid_visual.py`
### Testing
To test HyLEAP for a specific scenario `scenario` $\in$ `[01_int, 02_int, 03_int, 01_non_int, 02_non_int, 03_non_int]`, run the following command,
> `python eval_hyleap.py  --port=<port> --despot_port=<despot_port> --test=scenario`

The argument `--desport_port` specificies the port over which the RL-agent communicates with the planner.
To run the evaluation for all scenarios the following command can be used,
> `python eval_hyleap.py  --port=<port> --despot_port=<despot_port> --test=all`

### IS-DESPOT(*)
For IS-DESPOT(*) no training is required. The evaluation for `scenario` $\in$ `[01_int, 02_int, 03_int, 01_non_int, 02_non_int, 03_non_int]` can be run via
> `python eval_isdespot.py --server --port=<port> --despot_port=<despot_port> --test=scenario`
> `python eval_isdespot_star.py --server --port=<port> --despot_port=<despot_port> --test=scenario`

To run, the evaluation for all scenarios run:
> `python eval_isdespot.py --server --port=<port> --despot_port=<despot_port> --test=all`
> `python eval_isdespot_star.py --server --port=<port> --despot_port=<despot_port> --test=all`

### P3VI (path predictor)
In order to train P3VI it is required to first extract pedestrian paths from the simulation.
For this the script `exctract_pp_data.py` can be used. It is advised to extract the interactive and
non-interactive scenarios seperately to reduce the runtime. E.g. the command, 
> `python extract_pp_data.py --server --port=<port>` ,

extracts the non-interactive scenarios and the command,
> `python extract_pp_data.py --server --port=<port> --int` ,

extracts the interactive scenarios. Next, got to the directory `P3VI` and adjust the paths in the file `train.py`.
The path predictor can be trained with the command,
> `python train.py`.
## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

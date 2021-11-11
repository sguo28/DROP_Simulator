## Codes for "DROP: Deep relocating option policy for optimal ride-hailing vehicle repositioning"
We are happy to help if you have any questions. If you used any part of the code, please cite the following paper:

@article{qian2021drop,
  title={DROP: Deep relocating option policy for optimal ride-hailing vehicle repositioning},
  author={Qian, Xinwu and Guo, Shuocheng and Aggarwal, Vaneet},
  journal={arXiv preprint arXiv:2109.04149},
  year={2021}
}

## 1 preliminaries
1.1. INSTALL:
        conda install skimage
        conda install -c anaconda sqlalchemy
        conda install -c conda-forge polyline


## 2 folder descriptions
2.0. root folder
        - experiment: initalize vehicle location, populate vehicles, enter market, match, dispatch, and update.
        - main: run the simulation and DQN learning(if enabled), record metrics.
        - parse_results: preliminary result processing and visualization

2.1. central agent: MATCH vehciles and request, calculate cost per request

2.2. common: get and solve spatial infomation, get current time

2.3. config: spatial settings and time/time step settings.

2.4. data: preprocessed data

2.5. db: path for database

2.6. dqn agent: vehicles that learning DISPATCH policy with DQN

2.7. dummy agent: vehicles that follows fixed DISPATCH and MATCH policy.

2.8. logger: a directory for paths that saves results

2.9. logs: <!--save results for training and testing.-->

2.10. novelties: sets of codes to present various types of agents, vehicle category, vehicle status, customer perferences.

2.11. osrm/osrm-backend: for OSRM Engine deployment

2.12. simulator
- models: customer and vehicles
- service: demand generation, routing, sending requests to OSRM engine
- simulator: key processes of vehcile-customer interaction
- settings: config for simulation

2. 13. tools:  driver for saving files, dot dictionary to save parameters.


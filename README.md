## Code implementation for "DROP: Deep relocating option policy for optimal ride-hailing vehicle repositioning"
This repo is contributed by Prof. Xinwu Qian (U Alabama), Shuocheng Guo (U Alabama), and Vaneet Aggarwal (Purdue).

This research work is accepted by [Transportation Research Part C: Emerging Technologies](https://www.sciencedirect.com/science/article/pii/S0968090X22003369), and the preprint (initial version) can be found at [ArXiv](https://arxiv.org/pdf/2109.04149.pdf).

We are happy to help if you have any questions. If you used any part of the code, please cite the following paper:

@article{qian2022drop,
  title={DROP: Deep relocating option policy for optimal ride-hailing vehicle repositioning},
  author={Qian, Xinwu and Guo, Shuocheng and Aggarwal, Vaneet},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={145},
  pages={103923},
  year={2022},
  publisher={Elsevier}
}


### Data Inputs 
The preprocessed large files as data inputs can be fecthed via [OneDrive](https://bama365-my.sharepoint.com/personal/sguo18_crimson_ua_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fsguo18%5Fcrimson%5Fua%5Fedu%2FDocuments%2FDROPSimLargeFiles&ga=1).

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

2.13. tools:  driver for saving files, dot dictionary to save parameters.


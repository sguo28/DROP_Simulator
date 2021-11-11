import os
from novelties import status_codes
from config.hex_setting import DEFAULT_LOG_DIR
from tools.map import Map

FLAGS = Map()

FLAGS.offduty_threshold = -float('inf')  # 'q value off duty threshold')
FLAGS.offduty_probability= float(0.1) #, 'probability to automatically become off duty')
FLAGS.alpha = float(0.0) #, 'entropy coefficient')

FLAGS.save_memory_dir = os.path.join(DEFAULT_LOG_DIR, 'memory')  # 'replay memory storage')
FLAGS.save_network_dir = os.path.join(DEFAULT_LOG_DIR, 'networks')  # 'network model directory')
FLAGS.save_summary_dir = os.path.join(DEFAULT_LOG_DIR, 'summary')  # 'training summary directory')
FLAGS.load_network = ""  #load saved dqn_agent network.")
FLAGS.load_memory = ""  #load saved replay memory.")

FLAGS.train = bool(False)  #run training dqn_agent network.") #1
FLAGS.verbose = bool(False)  #print log verbosely.")

FLAGS.enable_pooling = bool(False)  #Enable RideSharing/CarPooling") #1
FLAGS.enable_pricing = bool(False)  #Enable Pricing Novelty") #1

FLAGS.vehicles = int(3000)  #number of vehicles")
FLAGS.dummy_vehicles = int(1) #number of vehicles using dummy agent")
FLAGS.dqn_vehicles = int(2999)  #number of vehicles using dqn agent")

FLAGS.pretrain = int(0)  #run N pretraining steps using pickled experience memory.")
FLAGS.start_time = int(1464753600 + 3600 * 5)  #simulation start datetime (unixtime)")
FLAGS.start_offset = int(0)  #simulation start datetime offset (days)")

FLAGS.days = int(7)  #simulation days")

FLAGS.n_diffusions = int(3)  #number of diffusion convolution")
FLAGS.batch_size = int(128)  #number of samples in a batch for SGD")
FLAGS.tag = str('test') # = "tag used to identify logs")
FLAGS.log_vehicle = bool(False)  #whether to log vehicle states")
FLAGS.use_osrm = bool(False)  #whether to use OSRM")
FLAGS.average = bool(False)  #whether to use diffusion filter or average filter")
FLAGS.trip_diffusion = bool(False)  #whether to use trip diffusion")
FLAGS.charging_threshold = float(0.20)

GAMMA = 0.98    # Discount Factor
MAX_MOVE = 5
NUM_SUPPLY_DEMAND_MAPS = 5
NUM_FEATURES = 7 + NUM_SUPPLY_DEMAND_MAPS * (1 + (FLAGS.n_diffusions + 1) * 2) + FLAGS.n_diffusions * 2 \
               + FLAGS.trip_diffusion * 4

# training hyper parameters
WORKING_COST = 0.2
DRIVING_COST = 0.2
STATE_REWARD_TABLE = {
    status_codes.V_IDLE : -WORKING_COST,
    status_codes.V_CRUISING : -(WORKING_COST + DRIVING_COST),
    status_codes.V_ASSIGNED : -(WORKING_COST + DRIVING_COST),
    status_codes.V_OCCUPIED : -(WORKING_COST + DRIVING_COST),
    status_codes.V_OFF_DUTY : 0.0
}
GLOBAL_STATE_UPDATE_CYCLE = 60 * 5
WAIT_ACTION_PROBABILITY = 0.70  # wait action probability in epsilon-greedy
EXPLORATION_STEPS = 3000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.01  # Final value of epsilon in epsilon-greedy
INITIAL_MEMORY_SIZE = 100  # Number of steps to populate the replay memory before training starts
NUM_SUPPLY_DEMAND_HISTORY = 7 * 24 * 3600 / GLOBAL_STATE_UPDATE_CYCLE + 1 # = 1 week
MAX_MEMORY_SIZE = 10000000  # Number of replay memory the dummy_agent uses for training
SAVE_INTERVAL = 1000  # The frequency with which the network is saved
TARGET_UPDATE_INTERVAL = 50  # The frequency with which the target network is updated
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update

SUPERCHARGING_TIME = 30 # min
SUPERCHARGING_PRICE = 0.30 # USD per min
SIM_ACCELERATOR = float(2) # accelerate charging speed

NUM_NEAREST_CS = 5
PENALTY_CHARGING_TIME = 45
MIN_PER_HOUR =60

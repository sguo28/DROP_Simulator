# config settings for DQN 
learning_rate = 1e-3
gamma = 0.99
replay_buffer_size = 1e5
batch_size = 128
relocation_dim = 7 # 1+6
charging_dim = 5
action_space = [i for i in range(relocation_dim+charging_dim)] # 7 relocation hex candidates, 5 nearest charging stations
input_dim = 4 # num of state fature lon/lat to hex_id
output_dim = relocation_dim+charging_dim
epsilon = 0.01

#weights for reward calculation
beta_earning =1 
beta_cost = 1
SOC_penalty =1
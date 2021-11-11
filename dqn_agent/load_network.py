import os
from simulator.settings import FLAGS
from config.settings import DEFAULT_LOG_DIR

# from dqn_agent.dqn_policy import DQNDispatchPolicy, DQNDispatchPolicyLearner
from dummy_agent.dispatch_policy import Dummy_DispatchPolicy

def load():
    '''
    import trained dispatching network if exists, otherwise create a new one.
    '''
    # setup_base_log_dir(FLAGS.tag) # find the path that saves NN.

    dispatch_policy = Dummy_DispatchPolicy()
    
    # if FLAGS.train:
    #     print("Set training mode")
    #     # print(tf.__version__)
    #     dispatch_policy = DQNDispatchPolicyLearner()
    #     dispatch_policy.build_q_network(load_network=FLAGS.load_network)

    #     if FLAGS.load_memory:
    #         # print(FLAGS.load_memory)
    #         dispatch_policy.load_experience_memory(FLAGS.load_memory)

    #     if FLAGS.pretrain > 0:
    #         for i in range(FLAGS.pretrain):
    #             average_loss, average_q_max = dispatch_policy.train_network(FLAGS.batch_size)
    #             # print("iterations : {}, average_loss : {:.3f}, average_q_max : {:.3f}".format(
    #             #     i, average_loss, average_q_max), flush=True)
    #             dispatch_policy.q_network.write_summary(average_loss, average_q_max)

    # else:
    #     dispatch_policy = DQNDispatchPolicy()
    # print(FLAGS.load_network)
    # if FLAGS.load_network:
    #     print("load network")
    #     dispatch_policy.build_q_network(load_network=FLAGS.load_network)

    return dispatch_policy


def setup_base_log_dir(base_log_dir):
    '''
    1. base_log_dir is the string: "test".
    2. if not esist, create new file to record simulation
    3. if training, create new files to record networks, summary, memory. (check when use memory)
    4. to comment out TRAIN mode, make sure there exists files in logs folder 
    '''
    print("Setup")
    base_log_path = "./logs/{}".format(base_log_dir)
    # print(base_log_path)
    if not os.path.exists(base_log_path):
        os.makedirs(base_log_path)

    for dirname in ["sim"]:
        p = os.path.join(base_log_path, dirname)
        if not os.path.exists(p):
            os.makedirs(p)

    if FLAGS.train:
        for dirname in ["networks", "summary", "memory"]:
            p = os.path.join(base_log_path, dirname)
            # print(p)
            if not os.path.exists(p):
                os.makedirs(p)
    # print(DEFAULT_LOG_DIR)

    if os.path.exists(DEFAULT_LOG_DIR):
        os.unlink(DEFAULT_LOG_DIR)
    os.symlink(base_log_dir, DEFAULT_LOG_DIR)
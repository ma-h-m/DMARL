import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *
from communication.socket_handler import  *
# from gradient_manager import *
import server.gradient_manager as gradient_manager
from envs.env_wrapper import SimpleAdversaryWrapper as EnvWrapper
from evaluater import start_evaluation_loop, evaluate_from_metadata
import time
# from server.reset import reset_server_state
reset = True
if reset:
    reset_server_state('server/policies', 'server/optimizers','server/policy_metadata.json' , 'server/gradients')
# create_optimizers_for_policies('server/policies', 'server/optimizers')
env = EnvWrapper()
env.reset()
# evaluate_from_metadata(env_wrapper= env)

# time.sleep(20)

# time.sleep(20)
# reset_server_state('server/policies', 'server/optimizers','server/policy_metadata.json' , 'server/gradients')
server = Server('127.0.0.1', 9998)

start_evaluation_loop(env, 10, 10)
gradient_manager.start_gradient_worker(
    gradients_path="server/gradients",
    policies_dir="server/policies",
    optimizer_dir="server/optimizers"
)

server.run()

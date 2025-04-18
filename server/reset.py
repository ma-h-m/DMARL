import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *
from communication.socket_handler import  *
# from gradient_manager import *
import server.gradient_manager as gradient_manager
from envs.env_wrapper import SimpleAdversaryWrapper as EnvWrapper
from evaluater import start_evaluation_loop, evaluate_from_metadata
reset_server_state('server/policies', 'server/optimizers','server/policy_metadata.json' , 'server/gradients')
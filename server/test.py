import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *
from communication.socket_handler import  *
# from gradient_manager import *
import server.gradient_manager as gradient_manager



# initialize_policy_metadata()
reset_server_state('server/policies', 'server/optimizers', 'server/gradients')
# create_optimizers_for_policies('server/policies', 'server/optimizers')
server = Server('127.0.0.1', 9999)
gradient_manager.start_gradient_worker(
    gradients_path="server/gradients",
    policies_dir="server/policies",
    optimizer_dir="server/optimizers"
)

server.run()

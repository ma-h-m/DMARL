from train import *
from utils import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from communication.socket_handler import  *

# client = Client("127.0.0.1", 9999, thread_id='1',policy_path = "client/temp_files/thread_1")
# client.connect()
# # client.register()
# client.request_policy("policy_1", "client/new_policy/policy_1")
# client.request_policy("policy_2", "client/new_policy/policy_2")
# client.request_policy("policy_0", "client/new_policy/policy_0")
if __name__ == "__main__":
    client = Client("127.0.0.1", 9999, thread_id='1',policy_path = "client/temp_files/thread_1")

    policies_path = setup_temp_policies(1, "client")
    client.connect()
    client.register()
    # client.request_random_policy('agent_1', os.path.join(policies_path, "policy_1"))
    # client.request_policy("policy_1", os.path.join(policies_path, "policy_1"))
    # client.request_policy("policy_2", os.path.join(policies_path, "policy_2"))
    # client.request_policy("policy_0", os.path.join(policies_path, "policy_0"))
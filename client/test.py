import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from communication.socket_handler import  *

client = Client("127.0.0.1", 9999, thread_id='1',policy_path = "client/temp_files/thread_1")
client.connect()
# client.register()
client.request_policy("policy_1", "client/new_policy/policy_1")
client.request_policy("policy_2", "client/new_policy/policy_2")
client.request_policy("policy_0", "client/new_policy/policy_0")
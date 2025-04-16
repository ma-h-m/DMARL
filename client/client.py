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
    client.request_policy("policy_1", os.path.join(policies_path, "policy_1"))
    client.request_policy("policy_2", os.path.join(policies_path, "policy_2"))
    client.request_policy("policy_0", os.path.join(policies_path, "policy_0"))

    thread_path = os.path.dirname(policies_path)
    agent_info_list = generate_agent_params(policies_path)


    # 现在可以访问每个策略实例：
    for agent_info in agent_info_list:
        policy_instance = agent_info["policy_instance"]
        print(f"Policy {agent_info['policy_id']} created with {policy_instance}")



    gradients = train(agent_info_list=agent_info_list, epochs=5)
    for agent_info in agent_info_list:
        policy_id = agent_info["policy_id"]
        gradient = gradients[agent_info['agent_id']]
        gradient_file_path = os.path.join(thread_path,"tmp",  f"gradient_{policy_id}.pt")
        torch.save(gradient, gradient_file_path)
        client.send_gradient(gradient_file_path, policy_id)
        os.remove(gradient_file_path)
    # gradients_path = save_all_gradients(gradients, save_dir=thread_path)
    
    # for agent_info in agent_info_list:
    #     agent_info["policy_instance"].save(os.path.join(agent_info["path"], "model.pt"), os.path.join(agent_info["path"], "optimizer.pt"))





    # print(gradients)
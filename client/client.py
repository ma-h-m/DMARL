from train import *
from utils import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.env_wrapper import SimpleAdversaryWrapper as EnvWrapper
from communication.socket_handler import  *


if __name__ == "__main__":
    env = EnvWrapper()
    env.reset()
    agent_ids = env.get_agent_ids()
    thread_id = 1
    client = Client("127.0.0.1", 9999, thread_id='1',policy_path = "client/temp_files/thread_1")

    policies_path = setup_temp_policies(1, "client")
    client.connect()
    client.register()
    client.close()
    training_steps_counter = 0
    while True:
        client.connect()
        for agent_id in agent_ids:
            policy_id = client.request_random_policy(agent_id, policies_path)
        client.close()
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
            client.connect()
            client.send_gradient(gradient_file_path, policy_id)
            client.close()
            os.remove(gradient_file_path)

        remove_temp_policies(1, "client")
        training_steps_counter += 1
        print(f"Training steps: {training_steps_counter}")
    # gradients_path = save_all_gradients(gradients, save_dir=thread_path)
    
    # for agent_info in agent_info_list:
    #     agent_info["policy_instance"].save(os.path.join(agent_info["path"], "model.pt"), os.path.join(agent_info["path"], "optimizer.pt"))





    # print(gradients)
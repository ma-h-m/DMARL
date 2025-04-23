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
    # client = Client("127.0.0.1", 9999, thread_id=str(thread_id), policy_path="client/temp_files/thread_1")

    training_steps_counter = 0

    while True:
        # ===== 每轮训练都重新连接一次 =====
        # client.connect()
        # client.register()

        # 请求策略
        policies_path = setup_temp_policies(thread_id, "client", False)
        # for agent_id in agent_ids:
        #     # client.request_random_policy(agent_id, policies_path)

        # 构造 agent_info
        agent_info_list = generate_agent_params(policies_path)

        # 本地训练
        gradients = train(agent_info_list=agent_info_list, epochs=1, batch_size=4096)

        # 发送所有梯度
        thread_path = os.path.dirname(policies_path)
        for agent_info in agent_info_list:
            policy_id = agent_info["policy_id"]
            gradient = gradients[agent_info["agent_id"]]
            gradient_file_path = os.path.join(thread_path, "tmp", f"gradient_{policy_id}.pt")
            os.makedirs(os.path.dirname(gradient_file_path), exist_ok=True)
            torch.save(gradient, gradient_file_path)
            # client.send_gradient(gradient_file_path, policy_id)
            os.remove(gradient_file_path)

        # client.close()
        # remove_temp_policies(thread_id, "client")

        training_steps_counter += 1
        print(f"Training steps: {training_steps_counter}")
        # break
    # gradients_path = save_all_gradients(gradients, save_dir=thread_path)
    
    # for agent_info in agent_info_list:
    #     agent_info["policy_instance"].save(os.path.join(agent_info["path"], "model.pt"), os.path.join(agent_info["path"], "optimizer.pt"))





    # print(gradients)
from train import *

if __name__ == "__main__":
    policies_path = setup_temp_policies(1, "client")
    agent_info_list = generate_agent_params(policies_path)


    # 现在可以访问每个策略实例：
    for agent_info in agent_info_list:
        policy_instance = agent_info["policy_instance"]
        print(f"Policy {agent_info['policy_id']} created with {policy_instance}")



    train(agent_info_list=agent_info_list, epochs=100)
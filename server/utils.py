import os
import json
import torch
import importlib.util
import shutil
# from communication.env_wrapper import EnvWrapper

def create_optimizers_for_policies(policies_path: str, optimizer_save_dir: str):
    """
    遍历所有 policy 文件夹，读取 config，实例化模型和优化器，并保存优化器参数。
    """
    if  os.path.exists(optimizer_save_dir):

        print(f"Removing existing optimizers directory {optimizer_save_dir}")
        shutil.rmtree(optimizer_save_dir)
    os.makedirs(optimizer_save_dir)
    for policy_name in os.listdir(policies_path):
        policy_path = os.path.join(policies_path, policy_name)
        config_path = os.path.join(policy_path, "agent.config")
        model_file = os.path.join(policy_path, "model.py")

        if not (os.path.isdir(policy_path) and os.path.exists(config_path) and os.path.exists(model_file)):
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)

        input_dim = config.get("observation_shape", 10)
        action_dim = config.get("action_shape", 5)
        lr = config.get("optimizer_config", {}).get("lr", 1e-3)
        agent_id = config.get("agent_id")

        # Load Model
        spec_model = importlib.util.spec_from_file_location("Model", model_file)
        model_module = importlib.util.module_from_spec(spec_model)
        spec_model.loader.exec_module(model_module)
        Model = model_module.Model
        model_instance = Model(input_dim, action_dim)

        # Create optimizer
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr)

        # Save optimizer state_dict
        save_path = os.path.join(optimizer_save_dir, f"{policy_name}_optimizer.pt")
        torch.save(optimizer.state_dict(), save_path)
        print(f"Saved optimizer for {policy_name} to {save_path}")



def read_agent_config(policy_dir):
    config_path = os.path.join(policy_dir, 'agent.config')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def initialize_policy_metadata(policies_root='server/policies', metadata_path='server/policy_metadata.json'):
    metadata = {}
    if os.path.exists(policies_root):
        for policy_name in os.listdir(policies_root):
            policy_dir = os.path.join(policies_root, policy_name)
            if os.path.isdir(policy_dir):
                config = read_agent_config(policy_dir)
                if config:
                    metadata[policy_name] = {
                        **config,
                        "train_steps": 0
                    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[INIT] Policy metadata initialized with {len(metadata)} policies.")





def reset_parameters_for_policies(policies_path: str):
    """
    重置每个 policy 的模型和优化器参数。
    删除 model.pt 和 optimizer.pt 文件，重新初始化模型并保存。
    """
    for policy_name in os.listdir(policies_path):
        policy_path = os.path.join(policies_path, policy_name)
        config_path = os.path.join(policy_path, "agent.config")
        model_file = os.path.join(policy_path, "model.py")

        if not (os.path.isdir(policy_path) and os.path.exists(config_path) and os.path.exists(model_file)):
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)

        input_dim = config.get("observation_shape", 10)
        action_dim = config.get("action_shape", 5)
        lr = config.get("optimizer_config", {}).get("lr", 1e-3)

        # Load Model
        spec_model = importlib.util.spec_from_file_location("Model", model_file)
        model_module = importlib.util.module_from_spec(spec_model)
        spec_model.loader.exec_module(model_module)
        Model = model_module.Model
        model_instance = Model(input_dim, action_dim)

        # Create optimizer
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=lr)

        # Define file paths
        model_path = os.path.join(policy_path, "model.pt")
        optimizer_path = os.path.join(policy_path, "optimizer.pt")

        # Remove old files if they exist
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Removed existing model at {model_path}")
        if os.path.exists(optimizer_path):
            os.remove(optimizer_path)
            print(f"Removed existing optimizer at {optimizer_path}")

        # Save fresh parameters
        torch.save(model_instance.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        print(f"Reset and saved new model and optimizer for {policy_name}")
def remove_all_gradients(gradients_path: str):
    """
    删除所有梯度文件。
    """
    if os.path.exists(gradients_path):
        for file in os.listdir(gradients_path):
            file_path = os.path.join(gradients_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed gradient file {file_path}")
        print(f"All gradient files removed from {gradients_path}")
    else:
        print(f"No gradients found at {gradients_path}")

def reset_server_state(policies_path: str = 'server/policies', optimizer_save_dir: str = 'server/optimizers', metadata_path: str = 'server/policy_metadata.json', gradients_path: str = 'server/gradients'):
    """
    重置服务器状态，包括模型和优化器参数。
    """
    reset_parameters_for_policies(policies_path)
    create_optimizers_for_policies(policies_path, optimizer_save_dir)
    print(f"Server state reset. Policies in {policies_path} and optimizers in {optimizer_save_dir}")
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        print(f"Removed existing metadata at {metadata_path}")
    initialize_policy_metadata(policies_root=policies_path, metadata_path=metadata_path)
    print(f"Server state reset. Metadata initialized at {metadata_path}")
    remove_all_gradients(gradients_path)
    print(f"Server state reset. All gradients removed from {gradients_path}")
    
# def apply_gradient_update(policy_id: str, gradients_path: str, policies_dir: str, optimizer_dir: str, remove_after_applied: bool = True):
#     """
#     根据 policy_id 找到对应的 policy 目录，加载模型和 optimizer，读取梯度，更新模型参数。
#     """
#     policy_path = os.path.join(policies_dir, policy_id)
#     config_path = os.path.join(policy_path, "agent.config")
#     model_file = os.path.join(policy_path, "model.py")
#     gradient_file = os.path.join(gradients_path, f"{policy_id}.pt")
#     optimizer_file = os.path.join(optimizer_dir, f"{policy_id}_optimizer.pt")

#     if not os.path.exists(gradient_file):
#         print(f"Gradient file {gradient_file} does not exist")
#         return

#     with open(config_path, 'r') as f:
#         config = json.load(f)

#     input_dim = config.get("observation_shape", 10)
#     action_dim = config.get("action_shape", 5)
#     lr = config.get("optimizer_config", {}).get("lr", 1e-3)

#     # Load Model
#     spec_model = importlib.util.spec_from_file_location("Model", model_file)
#     model_module = importlib.util.module_from_spec(spec_model)
#     spec_model.loader.exec_module(model_module)
#     Model = model_module.Model
#     model = Model(input_dim, action_dim)

#     # Load Optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     if os.path.exists(optimizer_file):
#         optimizer.load_state_dict(torch.load(optimizer_file))

#     # Load gradients
#     gradients = torch.load(gradient_file)  # dict[str, Tensor]
#     model_named_params = dict(model.named_parameters())  # dict[str, Parameter]

#     for name, param in model_named_params.items():
#         if name in gradients and gradients[name] is not None:
#             param.grad = gradients[name]

#     optimizer.step()
#     optimizer.zero_grad()
#     torch.save(optimizer.state_dict(), optimizer_file)
#     print(f"Applied gradient update for {policy_id} and saved optimizer to {optimizer_file}")
#     if remove_after_applied:
#         os.remove(gradient_file)
#         print(f"Removed gradient file {gradient_file}")



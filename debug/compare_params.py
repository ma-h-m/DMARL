import os
import importlib.util
import torch

def load_model_from_policy(policy_path):
    model_path = os.path.join(policy_path, "model.py")
    config_path = os.path.join(policy_path, "agent.config")
    model_pth = os.path.join(policy_path, "model.pt")

    if not os.path.exists(model_path) or not os.path.exists(model_pth):
        raise FileNotFoundError(f"Missing model.py or model.pt in {policy_path}")

    # 动态导入 Model 类
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    with open(config_path, 'r') as f:
        import json
        config = json.load(f)
        input_dim = config.get("observation_shape")
        action_dim = config.get("action_shape")

    model = model_module.Model(input_dim, action_dim)
    model.load_state_dict(torch.load(model_pth, map_location="cpu"))
    return model

def compare_model_params(model1, model2):
    diff_report = {}
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            diff_report[name1] = "Name mismatch"
            continue
        if not torch.allclose(param1, param2, atol=1e-5):
            diff_report[name1] = torch.norm(param1 - param2).item()
    return diff_report

def compare_policies(path1, path2):
    print(f"Comparing models in:\n- {path1}\n- {path2}\n")
    model1 = load_model_from_policy(path1)
    model2 = load_model_from_policy(path2)

    differences = compare_model_params(model1, model2)
    if not differences:
        print("✅ No differences found. The models are identical (within tolerance).")
    else:
        print("❌ Differences found in the following parameters:")
        for name, diff in differences.items():
            print(f"  - {name}: {diff if isinstance(diff, float) else diff}")

compare_policies("debug/policy_0_copy", "debug/policy_0")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("path1", help="Path to first policy folder")
#     parser.add_argument("path2", help="Path to second policy folder")
#     args = parser.parse_args()

#     compare_policies(args.path1, args.path2)

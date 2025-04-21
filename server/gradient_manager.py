# server/gradient_manager.py

import os
import torch
import threading
import queue
import json
import importlib.util
from typing import Dict

from utils import save_policy_checkpoint

# === Lock pool for per-policy gradient access ===
_policy_locks: Dict[str, threading.Lock] = {}
_policy_locks_lock = threading.Lock()

policy_metadata_lock = threading.Lock()
policy_metadata_path = 'server/policy_metadata.json'

def get_policy_lock(policy_id: str) -> threading.Lock:
    with _policy_locks_lock:
        if policy_id not in _policy_locks:
            _policy_locks[policy_id] = threading.Lock()
        return _policy_locks[policy_id]
    

_gradient_locks: Dict[str, threading.Lock] = {}
_gradient_locks_lock = threading.Lock()
def get_gradient_lock(policy_id: str) -> threading.Lock:
    with _gradient_locks_lock:
        if policy_id not in _gradient_locks:
            _gradient_locks[policy_id] = threading.Lock()
        return _gradient_locks[policy_id]

# === Gradient Update Queue ===
gradient_update_queue = queue.Queue()

def enqueue_gradient_update(policy_id: str):
    print(f"[Debug] Queue_id for enqueue_gradient_update: ", id(gradient_update_queue))
    print(f"[GradientManager] Enqueueing {policy_id}")
    gradient_update_queue.put(policy_id)
    print(f"[GradientManager] Queue size is now: {gradient_update_queue.qsize()}")

# === Main Gradient Update Logic ===
def apply_gradient_update(policy_id: str, gradients_path: str, policies_dir: str, optimizer_dir: str, remove_after_applied: bool = True, gradient_clip : float = 1.0):
    policy_path = os.path.join(policies_dir, policy_id)
    config_path = os.path.join(policy_path, "agent.config")
    model_file = os.path.join(policy_path, "model.py")
    gradient_file = os.path.join(gradients_path, f"{policy_id}.pt")
    optimizer_file = os.path.join(optimizer_dir, f"{policy_id}_optimizer.pt")

    if not os.path.exists(gradient_file):
        print(f"[GradientManager] Gradient file {gradient_file} does not exist")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    input_dim = config.get("observation_shape", 10)
    action_dim = config.get("action_shape", 5)
    lr = config.get("optimizer_config", {}).get("lr", 1e-3)

    # Load Model class dynamically
    spec_model = importlib.util.spec_from_file_location("Model", model_file)
    model_module = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(model_module)
    Model = model_module.Model
    model = Model(input_dim, action_dim)

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if os.path.exists(optimizer_file):
        optimizer.load_state_dict(torch.load(optimizer_file))

    # Load gradients
    gradients = torch.load(gradient_file)  # dict[str, Tensor]
    named_params = dict(model.named_parameters())
    for name, param in named_params.items():
        if name in gradients and gradients[name] is not None:
            param.grad = gradients[name]

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
    optimizer.step()
    optimizer.zero_grad()
    torch.save(optimizer.state_dict(), optimizer_file)
    torch.save(model.state_dict(), os.path.join(policy_path, "model.pt"))
    print(f"[GradientManager] Updated parameters for {policy_id}")

    if remove_after_applied:
        os.remove(gradient_file)
        print(f"[GradientManager] Removed gradient file {gradient_file}")
    
    # Update "train_steps" in policy metadata
    with policy_metadata_lock:
        if os.path.exists(policy_metadata_path):
            with open(policy_metadata_path, 'r') as f:
                policy_metadata = json.load(f)
        else:
            policy_metadata = {}

        if policy_id not in policy_metadata:
            policy_metadata[policy_id] = {}

        # Increment the "train_steps" value
        train_steps = policy_metadata[policy_id].get("train_steps", 0)
        policy_metadata[policy_id]["train_steps"] = train_steps + 1

        with open(policy_metadata_path, 'w') as f:
            json.dump(policy_metadata, f, indent=4)
        
        # Save a checkpoint of the policy
        if train_steps % 100 == 0:  # Save every 10 steps
            checkpoint_path = os.path.join(policies_dir, f"checkpoints")
            save_policy_checkpoint(policy_id, policy_path, train_steps, checkpoint_root=checkpoint_path)



# === Worker Thread ===
def gradient_worker(gradients_path: str, policies_dir: str, optimizer_dir: str):
    print("[GradientManager] Worker is running and listening for gradient updates...")
    print("[Debug] Queue_id for gradient_worker: ", id(gradient_update_queue))

    while True:
        policy_id = gradient_update_queue.get()
        print(f"[GradientManager] Got policy_id from queue: {repr(policy_id)}")

        if policy_id is None:
            print("[GradientManager] Received shutdown signal. Exiting worker.")
            break

        try:
            policy_lock = get_policy_lock(policy_id)
            print(f"[GradientManager] Waiting for lock on policy {policy_id}")
            gradient_lock = get_gradient_lock(policy_id)
            with policy_lock , gradient_lock:
                print(f"[GradientManager] Applying gradient update for {policy_id}")
                apply_gradient_update(policy_id, gradients_path, policies_dir, optimizer_dir)
                print(f"[GradientManager] Gradient update applied for {policy_id}")
        except Exception as e:
            print(f"[GradientManager] Exception during update of {policy_id}: {e}")
        finally:
            gradient_update_queue.task_done()

def start_gradient_worker(gradients_path: str, policies_dir: str, optimizer_dir: str):
    thread = threading.Thread(
        target=gradient_worker,
        args=(gradients_path, policies_dir, optimizer_dir),
        daemon=True
    )
    thread.start()
    print("[GradientManager] Gradient worker started.")
    return thread

def merge_gradients(existing_path: str, new_path: str):
    if not os.path.exists(existing_path):
        os.rename(new_path, existing_path)
        return

    existing = torch.load(existing_path)
    new = torch.load(new_path)

    if existing.keys() != new.keys():
        raise ValueError("Gradient files have mismatched keys")

    merged = {}
    for k in existing:
        merged[k] = existing[k] + new[k]

    torch.save(merged, existing_path)
    os.remove(new_path)
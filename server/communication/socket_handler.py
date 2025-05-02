import os
import socket
import json
import threading
import zipfile
from pathlib import Path
import shutil
import server.gradient_manager as gradient_manager

# from server.gradient_manager import get_policy_lock, enqueue_gradient_update

from typing import Dict
import time
def remove_later(path, delay=1.0):
    def _remove():
        time.sleep(delay)
        try:
            # if 
            shutil.rmtree(path)
            # os.remove(path)
        except FileNotFoundError:
            pass
    threading.Thread(target=_remove).start()
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"
import random
import uuid
# ========== UTILS ==========
def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=rel_path)

def unzip_folder(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)
    os.remove(zip_path)

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

# ========== SERVER SIDE ==========
class Server:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients_info_path = 'server/clients.json'
        self.policies_info_path = 'server/policy_metadata.json'
        self.tmp_path = 'server/tmp'
        self.clients = {}
        self.policies = {}
        self.load_metadata()

    def load_metadata(self):

        if os.path.exists(self.clients_info_path):
            with open(self.clients_info_path, 'r') as f:
                self.clients = json.load(f)

        with gradient_manager.policy_metadata_lock:
            if os.path.exists(self.policies_info_path):
                with open(self.policies_info_path, 'r') as f:
                    self.policies = json.load(f)

    def save_metadata(self):
        with open(self.clients_info_path, 'w') as f:
            json.dump(self.clients, f, indent=2)
        # with open(self.policies_info_path, 'w') as f:
        #     json.dump(self.policies, f, indent=2)

    def recv_exact_file(self, conn, target_path, file_size):
        with open(target_path, 'wb') as f:
            received = 0
            while received < file_size:
                chunk = conn.recv(min(BUFFER_SIZE, file_size - received))
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)

    def recv_until_separator(conn, sep=b'<SEPARATOR>'):
        data = b""
        while sep not in data:
            chunk = conn.recv(1)
            if not chunk:
                break
            data += chunk
        return data.decode()

    def handle_client(self, conn, addr):
        print(f"[SERVER] Connected by {addr}")
        client_id = None

        while True:
            try:
                msg = conn.recv(BUFFER_SIZE).decode()
                # msg = self.recv_until_separator(conn)
                if not msg:
                    break

                if msg.startswith("REGISTER"):
                    _, thread_id = msg.split(SEPARATOR)
                    client_id = f"{addr[0]}_{thread_id}"
                    self.clients[client_id] = {"ip": addr[0], "thread_id": thread_id}
                    self.save_metadata()
                    conn.sendall(f"REGISTERED{SEPARATOR}{client_id}".encode())

                elif msg.startswith("SEND_POLICY"):
                    _, policy_name, file_size_str = msg.split(SEPARATOR)
                    file_size = int(file_size_str)
                    zip_file_path = f"server/received/{policy_name}.zip"
                    self.recv_exact_file(conn, zip_file_path, file_size)
                    unzip_folder(zip_file_path, f"server/policies/{policy_name}")

                    agent_config = read_agent_config(f"server/policies/{policy_name}")
                    if agent_config:
                        self.policies[policy_name] = agent_config
                        self.save_metadata()

                    conn.sendall(f"RECEIVED{SEPARATOR}{policy_name}".encode())

                elif msg.startswith("REQUEST_POLICY"):
                    _, policy_name = msg.split(SEPARATOR)
                    folder_path = f"server/policies/{policy_name}"
                    zip_path = f"{self.tmp_path}/{policy_name}.zip"
                    os.makedirs(self.tmp_path, exist_ok=True)
                    zip_folder(folder_path, zip_path)
                    file_size = os.path.getsize(zip_path)
                    conn.sendall(f"{file_size}".encode() + b"\n")
                    with open(zip_path, 'rb') as f:
                        while chunk := f.read(BUFFER_SIZE):
                            conn.sendall(chunk)
                
                # elif msg.startswith("REQUEST_RANDOM_POLICY"):
                #     _, agent_id = msg.split(SEPARATOR)
                #     print(f"[SERVER] Client requested random policy for agent_id: {agent_id}")

                #     # Load policy metadata
                #     with gradient_manager.policy_metadata_lock:
                #         with open(self.policies_info_path, 'r') as f:
                #             all_policies = json.load(f)

                #     # Filter by agent_id
                #     matched_policies = [pid for pid, info in all_policies.items() if info.get("agent_id") == agent_id]

                #     if not matched_policies:
                #         error_msg = f"ERROR{SEPARATOR}No policy found for agent_id {agent_id}"
                #         conn.sendall(error_msg.encode())
                #         print(f"[SERVER] {error_msg}")
                #         return

                #     # Randomly select a matching policy
                #     selected_policy_id = random.choice(matched_policies)
                #     print(f"[SERVER] Selected policy: {selected_policy_id}")

                #     folder_path = f"server/policies/{selected_policy_id}"
                #     zip_path = f"{self.tmp_path}/{selected_policy_id}.zip"
                #     os.makedirs(self.tmp_path, exist_ok=True)

                #     policy_lock = gradient_manager.get_policy_lock(selected_policy_id)
                #     with policy_lock:
                #     # Zip the folder
                #         zip_folder(folder_path, zip_path)
                #         file_size = os.path.getsize(zip_path)

                #         # === Send protocol ===
                #         # 1. Send policy_id first (terminated with SEPARATOR)
                #         conn.sendall(f"{selected_policy_id}{SEPARATOR}".encode())

                #         # 2. Send zip file size
                #         conn.sendall(f"{file_size}\n".encode())

                #         # 3. Send zip file content
                #         with open(zip_path, 'rb') as f:
                #             while chunk := f.read(BUFFER_SIZE):
                #                 conn.sendall(chunk)

                #         print(f"[SERVER] Sent random policy {selected_policy_id} to client (size: {file_size} bytes)")
                #         # Clean up
                #         os.remove(zip_path)
                elif msg.startswith("REQUEST_RANDOM_POLICY"):
                    _, agent_id = msg.split(SEPARATOR)
                    print(f"[SERVER] Client requested random policy for agent_id: {agent_id}")

                    # Load policy metadata
                    with gradient_manager.policy_metadata_lock:
                        with open(self.policies_info_path, 'r') as f:
                            all_policies = json.load(f)

                    # Filter by agent_id
                    matched_policies = [pid for pid, info in all_policies.items() if info.get("agent_id") == agent_id]

                    if not matched_policies:
                        error_msg = f"ERROR{SEPARATOR}No policy found for agent_id {agent_id}"
                        conn.sendall(error_msg.encode())
                        print(f"[SERVER] {error_msg}")
                        return

                    # Randomly select a matching policy
                    selected_policy_id = random.choice(matched_policies)
                    print(f"[SERVER] Selected policy: {selected_policy_id}")

                    folder_path = f"server/policies/{selected_policy_id}"

                    # ==== 创建唯一临时目录 ====
                    unique_tmp_dir = os.path.join(self.tmp_path, str(uuid.uuid4()))
                    os.makedirs(unique_tmp_dir, exist_ok=True)
                    zip_path = os.path.join(unique_tmp_dir, f"{selected_policy_id}.zip")

                    policy_lock = gradient_manager.get_policy_lock(selected_policy_id)
                    with policy_lock:
                        zip_folder(folder_path, zip_path)
                        file_size = os.path.getsize(zip_path)

                        # === Send protocol ===
                        conn.sendall(f"{selected_policy_id}{SEPARATOR}".encode())
                        conn.sendall(f"{file_size}\n".encode())

                        with open(zip_path, 'rb') as f:
                            while chunk := f.read(BUFFER_SIZE):
                                conn.sendall(chunk)

                        print(f"[SERVER] Sent random policy {selected_policy_id} to client (size: {file_size} bytes)")

                    # ==== 清理 zip 文件和临时目录 ====
                    try:
                        # remove_later(zip_path)
                        # os.remove(zip_path)
                        remove_later(unique_tmp_dir)
                        # os.rmdir(unique_tmp_dir)  # 删除目录本身
                    except Exception as e:
                        print(f"[SERVER WARNING] Failed to clean temp files: {e}")

                elif msg.startswith("SEND_GRADIENT"):
                    _, policy_id, file_size_str = msg.split(SEPARATOR)
                    file_size = int(file_size_str)

                    os.makedirs("server/gradients/incoming", exist_ok=True)
                    temp_path = f"server/gradients/incoming/{policy_id}.pt"
                    final_path = f"server/gradients/{policy_id}.pt"

                    gradient_lock = gradient_manager.get_gradient_lock(policy_id)
                    with gradient_lock:
                        self.recv_exact_file(conn, temp_path, file_size)

                        if os.path.exists(final_path):
                            # print(f"[SERVER] Merging gradient for {policy_id}")
                            gradient_manager.merge_gradients(final_path, temp_path)
                        else:
                            os.rename(temp_path, final_path)
                            # print(f"[SERVER] Stored initial gradient for {policy_id}")


                    conn.sendall(f"RECEIVED_GRADIENT{SEPARATOR}{policy_id}".encode())
                    gradient_manager.enqueue_gradient_update(policy_id)
                    print(f"[SERVER] Gradient for {policy_id} enqueued for update.")

                elif msg.startswith("EXIT"):
                    break
            except Exception as e:
                print(f"[SERVER ERROR] {e}")
                break

        conn.close()
        print(f"[SERVER] Connection with {addr} closed")

    def run(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"[SERVER] Listening on {self.host}:{self.port}")

        while True:
            conn, addr = self.server_socket.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()


# ========== CLIENT SIDE ==========
class Client:
    def __init__(self, server_host='127.0.0.1', server_port=9999, thread_id='1', policy_path='tmp'):
        self.server_host = server_host
        self.server_port = server_port
        self.thread_id = thread_id
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.path = policy_path

    def connect(self):
        self.client_socket.connect((self.server_host, self.server_port))
        self.register()

    def register(self):
        msg = f"REGISTER{SEPARATOR}{self.thread_id}"
        self.client_socket.sendall(msg.encode())
        response = self.client_socket.recv(BUFFER_SIZE).decode()
        print(f"[CLIENT] Server response: {response}")

    def send_policy_folder(self, policy_path: str):
        policy_name = Path(policy_path).name
        zip_path = f"{self.path}/tmp/{policy_name}.zip"
        zip_folder(policy_path, zip_path)
        file_size = os.path.getsize(zip_path)
        msg = f"SEND_POLICY{SEPARATOR}{policy_name}{SEPARATOR}{file_size}"
        self.client_socket.sendall(msg.encode())

        with open(zip_path, 'rb') as f:
            while chunk := f.read(BUFFER_SIZE):
                self.client_socket.sendall(chunk)

        response = self.client_socket.recv(BUFFER_SIZE).decode()
        print(f"[CLIENT] Server response: {response}")

    def request_policy(self, policy_name: str, save_to: str):
        msg = f"REQUEST_POLICY{SEPARATOR}{policy_name}"
        self.client_socket.sendall(msg.encode())
        size_line = b""
        while not size_line.endswith(b"\n"):
            size_line += self.client_socket.recv(1)
        total_size = int(size_line.decode().strip())
        zip_path = f"{self.path}/tmp/{policy_name}.zip"
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        with open(zip_path, 'wb') as f:
            received = 0
            while received < total_size:
                chunk = self.client_socket.recv(min(BUFFER_SIZE, total_size - received))
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
        unzip_folder(zip_path, save_to)

    def send_gradient(self, gradient_file_path: str, policy_id: str):
        file_size = os.path.getsize(gradient_file_path)
        msg = f"SEND_GRADIENT{SEPARATOR}{policy_id}{SEPARATOR}{file_size}"
        self.client_socket.sendall(msg.encode())

        with open(gradient_file_path, 'rb') as f:
            while chunk := f.read(BUFFER_SIZE):
                self.client_socket.sendall(chunk)

        response = self.client_socket.recv(BUFFER_SIZE).decode()
        print(f"[CLIENT] Server response: {response}")

    def close(self):
        self.client_socket.sendall("EXIT".encode())
        self.client_socket.close()
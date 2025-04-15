import os
import socket
import json
import threading
import zipfile
from pathlib import Path

BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

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

def read_agent_config(policy_dir):
    config_path = os.path.join(policy_dir, 'agent.config')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

# ========== SERVER SIDE ==========
class PolicyServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients_info_path = 'server/clients.json'
        self.policies_info_path = 'server/policy_metadata.json'
        self.clients = {}
        self.policies = {}
        self.load_metadata()

    def load_metadata(self):
        if os.path.exists(self.clients_info_path):
            with open(self.clients_info_path, 'r') as f:
                self.clients = json.load(f)
        if os.path.exists(self.policies_info_path):
            with open(self.policies_info_path, 'r') as f:
                self.policies = json.load(f)

    def save_metadata(self):
        with open(self.clients_info_path, 'w') as f:
            json.dump(self.clients, f, indent=2)
        with open(self.policies_info_path, 'w') as f:
            json.dump(self.policies, f, indent=2)

    def handle_client(self, conn, addr):
        print(f"[SERVER] Connected by {addr}")
        client_id = None

        while True:
            try:
                msg = conn.recv(BUFFER_SIZE).decode()
                if not msg:
                    break

                if msg.startswith("REGISTER"):
                    _, thread_id = msg.split(SEPARATOR)
                    client_id = f"{addr[0]}_{thread_id}"
                    self.clients[client_id] = {"ip": addr[0], "thread_id": thread_id}
                    self.save_metadata()
                    conn.sendall(f"REGISTERED{SEPARATOR}{client_id}".encode())

                elif msg.startswith("SEND_POLICY"):
                    _, policy_name = msg.split(SEPARATOR)
                    zip_file_path = f"server/received/{policy_name}.zip"
                    with open(zip_file_path, "wb") as f:
                        while True:
                            data = conn.recv(BUFFER_SIZE)
                            if data == b"<END>":
                                break
                            f.write(data)
                    unzip_folder(zip_file_path, f"server/policies/{policy_name}")

                    agent_config = read_agent_config(f"server/policies/{policy_name}")
                    if agent_config:
                        self.policies[policy_name] = agent_config
                        self.save_metadata()

                    conn.sendall(f"RECEIVED{SEPARATOR}{policy_name}".encode())

                elif msg.startswith("REQUEST_POLICY"):
                    _, policy_name = msg.split(SEPARATOR)
                    folder_path = f"server/policies/{policy_name}"
                    zip_path = f"/tmp/{policy_name}.zip"
                    zip_folder(folder_path, zip_path)

                    with open(zip_path, 'rb') as f:
                        while chunk := f.read(BUFFER_SIZE):
                            conn.sendall(chunk)
                    conn.sendall(b"<END>")

                elif msg.startswith("SEND_GRADIENT"):
                    _, agent_id = msg.split(SEPARATOR)
                    file_path = f"server/gradients/{agent_id}.pt"
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'wb') as f:
                        while True:
                            data = conn.recv(BUFFER_SIZE)
                            if data == b"<END>":
                                break
                            f.write(data)
                    conn.sendall(f"RECEIVED_GRADIENT{SEPARATOR}{agent_id}".encode())

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
class PolicyClient:
    def __init__(self, server_host='127.0.0.1', server_port=9999, thread_id='1'):
        self.server_host = server_host
        self.server_port = server_port
        self.thread_id = thread_id
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        

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
        zip_path = f"/tmp/{policy_name}.zip"
        zip_folder(policy_path, zip_path)

        msg = f"SEND_POLICY{SEPARATOR}{policy_name}"
        self.client_socket.sendall(msg.encode())

        with open(zip_path, 'rb') as f:
            while chunk := f.read(BUFFER_SIZE):
                self.client_socket.sendall(chunk)
        self.client_socket.sendall(b"<END>")

        response = self.client_socket.recv(BUFFER_SIZE).decode()
        print(f"[CLIENT] Server response: {response}")

    def request_policy(self, policy_name: str, save_to: str):
        msg = f"REQUEST_POLICY{SEPARATOR}{policy_name}"
        self.client_socket.sendall(msg.encode())

        zip_path = f"/tmp/{policy_name}_download.zip"
        with open(zip_path, 'wb') as f:
            while True:
                data = self.client_socket.recv(BUFFER_SIZE)
                if data == b"<END>":
                    break
                f.write(data)
        unzip_folder(zip_path, save_to)

    def send_gradient(self, gradient_file_path: str, agent_id: str):
        msg = f"SEND_GRADIENT{SEPARATOR}{agent_id}"
        self.client_socket.sendall(msg.encode())

        with open(gradient_file_path, 'rb') as f:
            while chunk := f.read(BUFFER_SIZE):
                self.client_socket.sendall(chunk)
        self.client_socket.sendall(b"<END>")

        response = self.client_socket.recv(BUFFER_SIZE).decode()
        print(f"[CLIENT] Server response: {response}")

    def close(self):
        self.client_socket.sendall("EXIT".encode())
        self.client_socket.close()


# ========== Example Execution ==========
# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--server', action='store_true', help='Run as server')
#     parser.add_argument('--client', action='store_true', help='Run as client')
#     parser.add_argument('--thread_id', type=str, default='1', help='Client thread id')
#     parser.add_argument('--policy_path', type=str, help='Path to policy folder to send')
#     parser.add_argument('--request_policy', type=str, help='Policy name to pull from server')
#     parser.add_argument('--gradient_path', type=str, help='Path to gradient file to send')
#     parser.add_argument('--agent_id', type=str, help='Agent ID for gradient')
#     args = parser.parse_args()

#     if args.server:
#         server = PolicyServer()
#         server.run()

#     elif args.client:
#         client = PolicyClient(thread_id=args.thread_id)
#         client.connect()
#         if args.policy_path:
#             client.send_policy_folder(args.policy_path)
#         if args.request_policy:
#             client.request_policy(args.request_policy, save_to='client/pulled_policies')
#         if args.gradient_path and args.agent_id:
#             client.send_gradient(args.gradient_path, args.agent_id)
#         client.close()
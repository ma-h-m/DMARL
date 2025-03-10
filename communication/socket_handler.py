import socket
import json
import pickle
import threading

BUFFER_SIZE = 4096

class SocketServer:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clients = {}

    def start(self):
        """启动服务器，等待客户端连接"""
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server started at {self.host}:{self.port}")
        
        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"Client connected: {addr}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def handle_client(self, client_socket):
        """处理客户端请求"""
        try:
            data = client_socket.recv(BUFFER_SIZE).decode()
            request = json.loads(data)

            if request["type"] == "upload_policy":
                self.receive_policy(client_socket, request["policy_name"])
            elif request["type"] == "download_policy":
                self.send_policy(client_socket, request["policy_name"])

        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()

    def receive_policy(self, client_socket, policy_name):
        """接收客户端上传的策略"""
        policy_data = client_socket.recv(BUFFER_SIZE)
        with open(f"server/policies/{policy_name}.pkl", "wb") as f:
            f.write(policy_data)
        print(f"Received policy: {policy_name}")

    def send_policy(self, client_socket, policy_name):
        """发送策略给客户端"""
        try:
            with open(f"server/policies/{policy_name}.pkl", "rb") as f:
                policy_data = f.read()
            client_socket.send(policy_data)
            print(f"Sent policy: {policy_name}")
        except FileNotFoundError:
            client_socket.send(b"ERROR: Policy not found")


class SocketClient:
    def __init__(self, server_host="127.0.0.1", server_port=5000):
        self.server_host = server_host
        self.server_port = server_port

    def send_request(self, request):
        """向服务器发送请求"""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.server_host, self.server_port))
        client_socket.send(json.dumps(request).encode())

        if request["type"] == "download_policy":
            self.receive_policy(client_socket, request["policy_name"])
        client_socket.close()

    def upload_policy(self, policy_name, policy_object):
        """上传策略到服务器"""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.server_host, self.server_port))
        
        request = {"type": "upload_policy", "policy_name": policy_name}
        client_socket.send(json.dumps(request).encode())

        policy_data = pickle.dumps(policy_object)
        client_socket.send(policy_data)
        print(f"Uploaded policy: {policy_name}")

        client_socket.close()

    def receive_policy(self, client_socket, policy_name):
        """接收从服务器下载的策略"""
        policy_data = client_socket.recv(BUFFER_SIZE)
        with open(f"client/{policy_name}.pkl", "wb") as f:
            f.write(policy_data)
        print(f"Downloaded policy: {policy_name}")
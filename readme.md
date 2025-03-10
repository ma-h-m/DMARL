#### File Structure
Initial design:

marl_framework/
│── config/                         # Configuration files
│   ├── default.yaml                 # Training and communication parameters
│
│── server/                          # Server-side code
│   ├── server.py                     # Main server program
│   ├── policy_pool.py                # Manages policy storage and retrieval
|   |-- evaluation.py                 # Evaluate polices in pool
│   ├── policies/                     # Stores policy code and parameters
│   │   ├── policy_1/
│   │   │   ├── model.py              # Policy implementation
│   │   │   ├── model.pth             # Trained model parameters
│   │   ├── policy_2/
│   │   │   ├── rule_based.py         # Rule-based policy implementation
│   │   │   ├── config.json           # Configuration for rule-based policies
|   |   |-- ...
│   ├── metadata.json                 # Records policy information (name, file paths)
│
│── client/                          # Client-side code
│   ├── client.py                     # Main client program
│   ├── train.py                      # Training logic for policies
│   ├── policy.py                     # Policy wrapper
│   ├── new_policy/                   # Folder for new policies created on client side
│   │   ├── policy_1/                 # New policy 1
│   │   │   ├── model.py              # Model code for policy 1
│   │   │   ├── agent.config          # Configuration file for policy 1 (includes role and idx)
│   │   │   ├── policy.py             # Policy logic for policy 1 (training and inference logic)
│   │   ├── policy_2/                 # New policy 2
│   │   │   ├── model.py              # Model code for policy 2
│   │   │   ├── agent.config          # Configuration file for policy 2 (includes role and idx)
│   │   │   ├── policy.py             # Policy logic for policy 2 (training and inference logic)
│   ├── temp_files/                   # Temporary files for each client thread
│   │   ├── thread_1/                 # Folder for thread 1
│   │   │   ├── policies/             # Stores multiple policies (training & environment)
│   │   │   │   ├── policy_1/
│   │   │   │   │   ├── model.py      # Model code for policy 1
│   │   │   │   │   ├── model.pth     # Trained model for policy 1
|   |   |   |   |   |── agent.config  
|   |   |   |   |   |── policy.py
|   |   |   |   |   |── optimizer.pth
│   │   │   │   ├── policy_2/
│   │   │   │   │   ├── model.py      # Model code for policy 2
│   │   │   │   │   ├── model.pth     # Trained model for policy 2
|   |   |   |   |   |── agent.config  
|   |   |   |   |   |── policy.py
|   |   |   |   |   |── optimizer.pth
│   │   │   ├── agent_params.json     # Stores agent-specific parameters (e.g., paths, roles)
│   │   ├── thread_2/                 # Folder for thread 2
│   │   │   ├── policies/
│   │   │   │   ├── policy_1/
│   │   │   │   │   ├── model.py      # Model code for policy 1
│   │   │   │   │   ├── model.pth     # Trained model for policy 1
|   |   |   |   |   |── agent.config  
|   |   |   |   |   |── policy.py
|   |   |   |   |   |── optimizer.pth
│   │   │   │   ├── policy_2/
│   │   │   │   │   ├── model.py      # Model code for policy 2
│   │   │   │   │   ├── model.pth     # Trained model for policy 2
|   |   |   |   |   |── agent.config  
|   |   |   |   |   |── policy.py
|   |   |   |   |   |── optimizer.pth
│   │   │   ├── agent_params.json     
│   ├── utils/                        # Utility functions for client
│   │   ├── file_manager.py           # Handles file and directory operations
│
│── communication/                    # Networking and communication module
│   ├── socket_handler.py              # Handles server-client communication via sockets
│
│── scripts/                          # Scripts for launching and testing
│   ├── start_server.sh                # Shell script to start the server
│   ├── start_client.sh                # Shell script to start a client
│
│── tests/                            # Unit tests for the framework
│   ├── test_server.py                 # Tests for server components
│   ├── test_client.py                 # Tests for client behavior
│
│── utils/                            # Utility functions
│   ├── logger.py                      # Logging utilities
│   ├── config_loader.py               # Loads configurations
|
│── envs/                             # Environment adapters and custom environments
│   ├── env_wrapper.py                # Wrapper to integrate with environments
│
│── README.md                         # Project documentation
#!/bin/bash

# 用法: ./start_clients.sh <num_clients>
# 例如: ./start_clients.sh 4 会启动 thread_id 为 1~4 的客户端

NUM_CLIENTS=$1

if [ -z "$NUM_CLIENTS" ]; then
  echo "Usage: $0 <num_clients>"
  exit 1
fi

for (( i=1; i<=NUM_CLIENTS; i++ ))
do
  echo "Starting client thread $i..."
  python3 client/client.py --thread_id $i > logs/client_$i.log 2>&1 &
done

echo "All clients started in background. Check logs/ for output."

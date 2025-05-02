#!/bin/bash

# 启动 Server
echo "Launching server..."
bash script/lauch_server.sh
sleep 2  # 给 server 一点时间启动

# 启动多个 Clients
NUM_CLIENTS=4  # 修改这里以调整客户端数量
echo "Launching $NUM_CLIENTS clients..."

mkdir -p logs

for i in $(seq 1 $NUM_CLIENTS); do
    echo "Starting client $i..."
    python3 client/client.py --thread_id "$i" > "logs/client_${i}.log" 2>&1 &
done

echo "All components launched. Check logs/ for output."

#!/bin/bash

# 启动 server 并将日志写入 logs/server.log
mkdir -p logs

echo "Starting server..."
python3 server/test.py > logs/server.log 2>&1 &

echo "Server started in background. Log: logs/server.log"

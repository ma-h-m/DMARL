#!/bin/bash

echo "Stopping all server and client processes..."

# 1. 杀掉包含 server/test.py 的所有 python 进程
ps aux | grep '[p]ython3.*server/test.py' | awk '{print $2}' | xargs -r kill -9

# 2. 杀掉包含 client.py 的所有 python 进程
ps aux | grep '[p]ython3.*client.py' | awk '{print $2}' | xargs -r kill -9

echo "All processes terminated."
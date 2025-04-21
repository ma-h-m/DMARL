tell application "Terminal"
    # 启动 server
    do script "/usr/local/bin/python3 server/test.py"

    # 启动 client（新建一个窗口）
    delay 1
    do script "/usr/local/bin/python3 client/client.py" in (make new window)
end tell
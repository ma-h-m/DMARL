
from pettingzoo.mpe import simple_adversary_v3

def test_env():
    env = simple_adversary_v3.parallel_env(render_mode=None)
    print("env created OK")

import threading

threading.Thread(target=test_env).start()
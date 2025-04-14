from pettingzoo.mpe import simple_adversary_v3

env = simple_adversary_v3.env(render_mode="human")
env.reset(seed=42)
cnt = 0
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
    if agent == "agent_0":
        print(cnt)
        
        cnt += 1
    env.step(action)
env.close()
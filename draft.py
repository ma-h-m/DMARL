from pettingzoo.mpe import simple_reference_v3

env = simple_reference_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
    print(rewards)
    print(terminations)
    print(truncations)
    print(infos)
    break
env.close()

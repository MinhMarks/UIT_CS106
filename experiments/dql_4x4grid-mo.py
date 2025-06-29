import os
import sys
import torch
import numpy as np
import pandas as pd



if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import dqn_agent
from sumo_rl.agents.dqn_agent import DQNAgent
from sumo_rl.agents.dqn_agentplusreal import DQNAgentPlus


if __name__ == "__main__":
    start_epoch = 240
    epochs = 1
    start_episode = 1
    start_run = 1 
    episodes = 100
    save_model_path = "experiments/dqn_modelreal"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui= True,
        num_seconds=8000,
        reward_fn=["diff-waiting-time", "average-speed"],
        reward_weights=[1, 0.1],
        enforce_max_green=True,
        min_green=5,
        delta_time=5,
    )

    for epoch in range(start_epoch, start_epoch+100):
        initial_states = env.reset()
        if epoch == 1:
            agents = {
                ts: DQNAgentPlus(
                    state_size=len(env.encode(initial_states[ts], ts)),
                    action_size=env.action_spaces(ts).n,
                    use_cuda=True, 
                )
                for ts in env.ts_ids
            }
        elif start_epoch == epoch:
            save_epoch_episode_path = save_model_path + f"/epoch {epoch-1}/" 
        
            agents = {
                ts: DQNAgentPlus(
                    state_size=len(env.encode(initial_states[ts], ts)),
                    action_size=env.action_spaces(ts).n,
                    path = save_epoch_episode_path + f"/agent{ts}.pth",
                    use_cuda=True,
                )
                for ts in env.ts_ids
            }

        states = env.reset()
        encoded_states = {ts: env.encode(states[ts], ts) for ts in states}
        done = {"__all__": False}

        while not done["__all__"]:
            actions = {
                ts: agents[ts].act(encoded_states[ts]) for ts in agents
            }

            next_states, rewards, done, _ = env.step(actions)
            next_encoded = {
                ts: env.encode(next_states[ts], ts) for ts in next_states
            }

            for ts in agents:
                agents[ts].remember(
                    encoded_states[ts],
                    actions[ts],
                    rewards[ts],
                    next_encoded[ts],
                    done[ts],
                )
                agents[ts].replay()

            encoded_states = next_encoded

        this_epoch_path = save_model_path + f"/epoch {epoch}"
        if not os.path.exists(this_epoch_path):
            os.makedirs(this_epoch_path)
        

        for ts in agents:
            agents[ts].replay()
            agents[ts].save_model(
                path=this_epoch_path + f"/agent{ts}.pth",
            )

        env.save_csv(f"outputs/4x4real/dqn-4x4grid_epoch{epoch}", "" )

    env.close()

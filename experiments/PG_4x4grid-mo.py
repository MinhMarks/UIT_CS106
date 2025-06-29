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
from sumo_rl.agents import pg_agent
from sumo_rl.agents.pg_agent import ActorCriticAgent

if __name__ == "__main__":
    start_run = 1
    runs = 1
    start_episode = 261
    episodes = 280
    save_model_path = "experiments/model"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=True,
        num_seconds=8000,
        min_green=5,
        delta_time=5,
    )

    for run in range(start_run, runs + 1):
        initial_states = env.reset()
        if start_episode == 1:
            agents = {
                ts: ActorCriticAgent(
                    state_size=len(env.encode(initial_states[ts], ts)),
                    action_size=env.action_spaces(ts).n,
                )
                for ts in env.ts_ids
            }
        else:
            save_run_episode_path = save_model_path + f"/run {start_run}/" + f"/episode {start_episode-1}"
        
            agents = {
                ts: ActorCriticAgent(
                    state_size=len(env.encode(initial_states[ts], ts)),
                    action_size=env.action_spaces(ts).n,
                    actor_path = save_run_episode_path + f"/actor_ts{ts}.pth",
                    critic_path = save_run_episode_path + f"/critic_ts{ts}.pth",
                )
                for ts in env.ts_ids
            }

        for episode in range(start_episode, episodes + 1):
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
                        rewards[ts]
                    )
                
                encoded_states = next_encoded

            this_run_path = save_model_path + f"/run {run}"
            if not os.path.exists(this_run_path):
                os.makedirs(this_run_path)
            
            this_run_episode_path = this_run_path + f"/episode {episode}"
            if not os.path.exists(this_run_episode_path):
                os.makedirs(this_run_episode_path)

            for ts in agents:
                agents[ts].replay()
                agents[ts].save_model(
                    actor_path=this_run_episode_path + f"/actor_ts{ts}.pth",
                    critic_path=this_run_episode_path + f"/critic_ts{ts}.pth",
                )

            env.save_csv(f"outputs/4x4/PG-4x4grid_run{run}", episode)

    env.close()

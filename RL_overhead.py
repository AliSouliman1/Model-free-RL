import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np
import tyro
from dataclasses import dataclass

import gymnasium as gym
import wandb as wandb
import os
import stable_baselines3 as sb3

from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import random
import time
from typing import Callable

@dataclass
class Args:
    exp_name:str=os.path.basename(__file__)[:-len(".py")]

    seed:int=1

    cuda:bool=True

    wandb_project_name:str="RL_testing"

    wandb_entity:str="alisouliman"

    track:bool=True

    upload_model:bool=False
    
    capture_video:bool=False

    save_model:bool=True
    #Algorith hyperparameters
    env_name:str="CartPole-v1"

    env_num: int = 1


    total_timesteps:int=1000000

    learning_rate:float=2.5e-4

    gamma:int=0.99

    buffer_size:int=10000

    tau:float=0.9

    target_network_frequency:int=500

    batch_size:int=248

    start_epsilon:float=1

    end_epsilon:float=0.05

    steps_for_epsilon:int=500

    learning_start:int=10000

    train_frequency:int=10


def make_env(env_name,seed,env_number,capture_video,run_name):
    def thunk():
        if capture_video & env_number==0:
            env=gym.make(env_name,render_mode="rgb_array")
            env=gym.wrappers.RecordVideo(env,f"video/{run_name}")
        else:
            env=gym.make(env_name)
        env=gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env
    
    return thunk

class Q_network(nn.Module):
    def __init__(self, envs) -> None:
        super().__init__()
        
        self.networl=nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(),120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,envs.single_action_space.n)
        )
    def forward(self,x):

        return self.networl(x)
    

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
        

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: T.nn.Module,
    device: T.device = T.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(T.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(T.Tensor(obs).to(device))
            actions = T.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns




if __name__=="__main__":



    args=tyro.cli(Args)
    print(args.exp_name)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )
    

    write=SummaryWriter(f"runs/{run_name}")


    random.seed(args.seed)
    np.random.seed(args.seed)
    T.manual_seed(args.seed)
    
    device=("cuda" if T.cuda.is_available() and args.cuda else "cpu")

    env=gym.vector.SyncVectorEnv([make_env(args.env_name,args.seed+i,i,args.capture_video,f"some run/{i}") for i in range(args.env_num)])

    Q_value=Q_network(env).to(device)
    optimizer=optim.Adam(Q_value.parameters(),lr=args.learning_rate)
    Q_target=Q_network(env).to(device)
    Q_target.load_state_dict(Q_value.state_dict())

    rb=ReplayBuffer(args.buffer_size,env.single_observation_space,env.single_action_space,device,handle_timeout_termination=False)


    ###################################################################################################################################
                            # Setup is Complete # what follows is the algorithms
    ###################################################################################################################################


    obs,_=env.reset(seed=args.seed)
    for step in range(args.total_timesteps):
        if step%200==0:
            print("step=",step)
        epsilon=linear_schedule(args.start_epsilon,args.end_epsilon,0.5*args.total_timesteps,step)
        if random.random()<epsilon:
            actions=np.array([env.single_action_space.sample() for _ in range(env.num_envs)])

        
        else:
            q_values=Q_value(T.Tensor(obs).to(device))
            actions=T.argmax(q_values,dim=1).cpu().numpy()
            #print(actions)

        
        next_obs,reward,terminated,truncated,info=env.step(actions)

        #####
        real_next_obs=next_obs.copy()
        #print(real_next_obs.shape)
        for idx,trunc in enumerate(truncated):
            if trunc:
                real_next_obs[idx]=info["final_observation"][idx]

            
        ####

        rb.add(obs,real_next_obs,actions,reward,terminated,info)

        obs=next_obs

        if step > args.learning_start:
            if step % args.train_frequency == 0:
                data=rb.sample(args.batch_size)
                with T.no_grad():
                    target_max,_=Q_target(data.next_observations).max(dim=1)### print
                    td_target=data.rewards.flatten()+args.gamma*(1-data.dones.flatten())*target_max

                    #print (target_max.type) ### to kknow  numpy flatten or torch flatten
                    #print(data.rewards.type)


                values_old_states=Q_value(data.observations).gather(1,data.actions).squeeze()
                loss=F.mse_loss(td_target,values_old_states)
                #print(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step%args.target_network_frequency==0:
                for target_network_param, q_network_param in zip(Q_target.parameters(), Q_value.parameters()):
                    target_network_param.data.copy_(
                    args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
        


    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        T.save(Q_value.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_name,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Q_network,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            write.add_scalar("eval/episodic_return", episodic_return, idx)                


                    





    env.close()
    write.close()

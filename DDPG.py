import torch as torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import wandb
import time
import numpy as np
import random
import gymnasium as gym

from dataclasses import dataclass
import tyro
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer

from torch.distributions import Normal
import os





@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]

    env_name:str="Hopper-v4"

    env_number:int=1

    record_video:bool=True

    use_wandb:bool=True

    wandb_project:str="DDPG"


    torch_deterministic: bool = True

    wandb_entity:str="alisouliman"

    ### algorith hyper parameters
    total_steps:int=1000000

    buffer_size:int=100000

    update_frequency:int=2

    tau:float=0.005

    batch_size:int=256
    learning_start:int=25000

    seed:int = 1
    

    exploration_noise:float=0.1



    learning_rate:float=3e-4


    gamma:float=0.99
    

def make_env(env_name,seed,idx,capture_video):
    def thunk():
        if capture_video :
            env=gym.make(env_name,render_mode="rgb_array")
            env=gym.wrappers.RecordVideo(env,f"DDPG{args.wandb_project}")
        else:
            env=gym.make(env_name)
        env.action_space.seed(seed)

        env=gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


class critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

        
                

if __name__=="__main__":

    args=tyro.cli(Args)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


    envs=gym.vector.SyncVectorEnv([make_env(args.env_name,args.seed+i,i,args.record_video) for i in range(args.env_number)])
    envs.single_observation_space.dtype = np.float32

    device=("cuda" if torch.cuda.is_available() else "cpu")
    rb=ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False
    )


    Q_network=critic(envs).to(device)
    Q_target=critic(envs).to(device)
    Q_target.load_state_dict(Q_network.state_dict())
    Q_optimizer=optim.Adam(list(Q_network.parameters()),lr=args.learning_rate)

    policy=actor(envs).to(device)
    policy_target=actor(envs).to(device)
    policy_target.load_state_dict(policy.state_dict())
    policy_optimizer=optim.Adam(list(policy.parameters()),lr=args.learning_rate)


    count=0
    obs,_=envs.reset(seed=args.seed)
    for step in range (args.total_steps):
        if step<args.learning_start:
            action=np.array([envs.single_action_space.sample() for _ in range(args.env_number)])
        else:
            with torch.no_grad():
                """action=policy(torch.Tensor(obs).to(device)) ###to device
                noise=torch.normal(0,policy.action_scale * args.exploration_noise)
                action=action+noise
                action=action.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)"""
                action = policy(torch.Tensor(obs).to(device))
                action += torch.normal(0, policy.action_scale * args.exploration_noise)
                action = action.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
        next_obs,reward,terminations,truncations,info=envs.step(action)

        if "final_info" in info:
            for inff in info["final_info"]:
                print(f"global_step={step}, episodic_return={inff['episode']['r']}")
                writer.add_scalar("charts/episodic_return", inff["episode"]["r"], step)
                writer.add_scalar("charts/episodic_length", inff["episode"]["l"], step)
                break
        """
        for idx,trunc in enumerate(truncations):
            if trunc:
                next_obs[idx]=info["final_observation"][idx]
        rb.add(obs,next_obs,action,reward,terminations,info)
        """
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:

                print("truncation      ",truncations)
                print("idx     ",idx)
                print("trunc     ",trunc)
                #print(real_next_obs)
                count=count+1
                print(count)
                #if "final_observation" in info:

                real_next_obs[idx] = info["final_observation"][idx]
        rb.add(obs, real_next_obs, action, reward, terminations, info)


        obs=next_obs

        ###########################################
        if step> args.learning_start:

            data=rb.sample(args.batch_size)
            with torch.no_grad():
                actions=policy_target(data.next_observations)
                Q_values_target=Q_target(data.next_observations,actions)
                data_R=data.rewards.flatten()
                #print("data_R shape",data_R.shape,"data_R size",data_R.size(),"data_R dim",data_R.dim())

                target=data.rewards.flatten()+(1-data.dones.flatten())*args.gamma*(Q_values_target).view(-1)
                
                #print("Q_values_target shape",Q_values_target.shape,"Q_values_target size",Q_values_target.size(),"Q_values_target dim",Q_values_target.dim())

            #print("target shape",target.shape,"target size",target.size(),"target dim",target.dim())
            Q_values=Q_network(data.observations,data.actions).view(-1)
            #print("Q_values shape",Q_values.shape,"Q_values size",Q_values.size(),"Q_values dim",Q_values.dim())
            Q_values=Q_values.view(-1)
            #print("with View",Q_values.shape)
            loss=F.mse_loss(target,Q_values)

            #print("Loss shape",loss.shape,"Loss size",loss.size(),"Loss dim",loss.dim(),"loss",loss)
            
            Q_optimizer.zero_grad()
            loss.backward()
            Q_optimizer.step()

            if step%args.update_frequency==0:
                policy_loss=-Q_network(data.observations,policy(data.observations)).mean()
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()


                for param, target_param in zip(policy.parameters(), policy_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                
                for param, target_param in zip(Q_network.parameters(),Q_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)



    envs.close()
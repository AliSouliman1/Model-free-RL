import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np

import wandb
import argparse
import os
import time
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import tyro



class args():
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_number_steps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration: float = 0.1
    """the scale of exploration noise"""
    min_number_samples: int = 25e3
    """timestep to start learning"""
    target_networks_update: int = 2
    """the frequency of training policy (delayed)"""






def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class QNetWork(nn.Module):
    def __init__(self,env):
        super(QNetWork,self).__init__()

        self.l1=nn.Linear(np.array(env.single_observation_space.shape).prod(),400)
        self.ln1=nn.LayerNorm(400)#layer norm
        self.l2=nn.Linear(400+np.prod(env.single_action_space.shape),300)
        self.ln2=nn.LayerNorm(300)#layer norm
        self.out=nn.Linear(300,1)


    def forward(self,obs,action):
        out=F.relu(self.l1(obs))
        out=self.ln1(out)
        out=F.relu(self.l2(torch.cat([out,action],1)))
        out=self.ln2(out)
        out=self.out(out)
        return out


class Actor(nn.Module):
    def __init__(self,env):
        super(Actor,self).__init__()
        self.l1=nn.Linear(np.array(env.single_observation_space.shape).prod(),400)
        self.l2=nn.Linear(400,300)
        self.out=nn.Linear(300,np.prod(env.single_action_space.shape))
        #self.scale=torch.tensor((env.action_space.high - env.action_space.low))
        self.register_buffer(
            "scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )


    def forward(self,obs):
        out=F.relu(self.l1(obs))
        out=F.relu(self.l2(out))
        out=torch.tanh(self.out(out))

        return out*self.scale + self.action_bias


if __name__=="__main__":
    args = tyro.cli(args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )
    writer = SummaryWriter(f"runs/{run_name}")
    device=("cuda" if torch.cuda.is_available() else "cpu")

    #create the env
    env=gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(env.single_action_space, gym.spaces.Box)


    actor=Actor(env).to(device)
    target_actor=Actor(env).to(device)
    Ql=QNetWork(env).to(device)
    target_Q=QNetWork(env).to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_Q.load_state_dict(Ql.state_dict())

    actor_optim=optim.Adam(params=actor.parameters(),lr=0.0001)
    Q_optim=optim.Adam(params=Ql.parameters(),lr=0.001)

    rb=ReplayBuffer(1000000,
                    env.single_observation_space,
                    env.single_action_space,
                    device,handle_timeout_termination=False)
    start_time=time.time()
    env.single_observation_space.dtype = np.float32

    # start interacting and learning
    obs, _ = env.reset(seed=args.seed)

    #outer loop for each 
    for step in range(args.total_number_steps):
        if step<args.min_number_samples:
            action=np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
        
        else:
            action=actor(torch.Tensor(obs).to(device))
            exploration=torch.normal(0,actor.scale*args.exploration).to(device)
            #print("exploration",exploration.device)
            action=action+exploration
            action = action.cpu().detach().numpy().clip(env.single_action_space.low, env.single_action_space.high)
            #print("action dtype",action.dtype)
        
        
        next_obs,reward,terminated,truncated,info=env.step(action)


        if "final_info" in info:
            for infos in info["final_info"]:
                print(f"global_step={step}, episodic_return={infos['episode']['r']}")
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], step)
                break
        ##handeling termination and final states
        ##adding data to the replay buffer, 
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncated):
            if trunc:
                real_next_obs[idx] = info["final_observation"][idx]
                print("truncated and ",terminated,truncated)

        rb.add(obs, real_next_obs, action, reward, terminated, info)
        obs=next_obs

        if step>args.min_number_samples:

            data=rb.sample(args.batch_size)
            with torch.no_grad():
                #print("type",data.next_observations.dtype)
                actions=target_actor(data.next_observations.float())
                q_target_values=data.rewards.flatten()+(1-data.dones.flatten())*args.gamma*target_Q(data.next_observations.float(),actions).view(-1)#Q(s)=r+gamma*Q(s+1,actor(s+1))

            q=Ql(data.observations.float(),data.actions).view(-1)
            q_error=F.mse_loss(q,q_target_values)
            
            Q_optim.zero_grad()
            q_error.backward()
            Q_optim.step()

            #update actor
            actor_error=-Ql(data.observations.float(),actor(data.observations.float())).mean()
            actor_optim.zero_grad()
            actor_error.backward()
            actor_optim.step()

            if step%(args.target_networks_update)==0:
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(Ql.parameters(), target_Q.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            if step%1000==0:
                print(step)
    env.close()

    writer.close()





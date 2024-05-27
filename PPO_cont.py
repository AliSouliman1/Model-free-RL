import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

import gymnasium as gym

from dataclasses import dataclass

import numpy as np
import wandb
import tyro
import stable_baselines3 as sb3
from stable_baselines3.common.buffers import ReplayBuffer
import time
import os
import random


@dataclass
class Args:
    ###env
    env_name:str="HalfCheetah-v4"

    env_number:int=1

    seed:int=4

    wandb_project:str="Rl_Test"

    wandb_entity:str="alisouliman"

    capture_video:bool=False

    wandb_track:bool=True

    ###algorithm

    batch_size:int=256

    buffer_size:int=100000

    epsioln:float=0.2

    gamma:float=0.99

    learning_rate:float=3e-4

    num_step:int=2048

    total_num_step:int=1000000

    num_minibatches:int=32

    minibatch_size:int=0

    num_iterations:int=0

    clip_coef: float = 0.2

    norm_adv:bool=True

    lambd:float=0.95

    clip_vloss:bool=True

    num_epoch:int=10


    entropy_coef:float=0.0
    value_function_coef:float=0.5
    max_grad_norm:float=0.5


def make_env(env_name,idx,run_name,capture_video,gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_name)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk
    

def layer_init(layer,std=np.sqrt(2),bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer
    
class Agent(nn.Module):
    def __init__(self,envs):
        super().__init__()

        self.critic=nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1),std=1.0),

            )
        
        self.actor_mean=nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,np.array(envs.single_action_space.shape).prod()),std=0.01)
            
        )
        self.actor_logstd=nn.Parameter(torch.zeros(1,np.array(envs.single_action_space.shape).prod()))

    def get_value(self,x):
        return self.critic(x)

    def get_action_and_value(self,x,action=None):
        mean=self.actor_mean(x)
        actor_std=torch.exp(self.actor_logstd)
        dis=torch.distributions.Normal(mean,actor_std)
        if action==None:
            action=dis.sample()
        prob=dis.log_prob(action)
        
        return action,prob.sum(1),dis.entropy().sum(1),self.get_value(x)
        
#run_name=os.join(os.path,f"{time.time()}")

if __name__=="__main__":

    args=tyro.cli(Args)

    run_name = f"{args.env_name}__{args.env_name}__{args.seed}__{int(time.time())}"
    args.batch_size=int(args.num_step*args.env_number)
    args.minibatch_size=int(args.batch_size//args.num_minibatches)
    args.num_iterations=args.total_num_step//args.batch_size
    device=("cuda" if torch.cuda.is_available() else "cpu")
    envs=gym.vector.SyncVectorEnv([make_env(args.env_name,i,run_name,args.capture_video,args.gamma ) for i in range(args.env_number)])


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


    agent=Agent(envs).to(device)
    optimizer=optim.Adam(agent.parameters(),args.learning_rate,eps=1e-5)
    global_Step=0
    obs=torch.zeros((args.num_step,args.env_number)+envs.single_observation_space.shape,device=device)
    actions=torch.zeros((args.num_step,args.env_number)+envs.single_action_space.shape,device=device)
    log_probs=torch.zeros((args.num_step,args.env_number),device=device)
    rewards=torch.zeros((args.num_step,args.env_number),device=device)
    dones=torch.zeros((args.num_step,args.env_number),device=device)
    values=torch.zeros((args.num_step,args.env_number),device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.env_number).to(device)
    for iteration in range(1,args.num_iterations+1):
        for step in range(args.num_step):
            global_step+=args.env_number

            obs[step]=next_obs
            dones[step]=next_done

            with torch.no_grad():
                action,log_prob,_,value=agent.get_action_and_value(next_obs)
                values[step]=value.flatten()
            log_probs[step]=log_prob
            actions[step]=action


            next_obs,reward,termination,truncation,infos=envs.step(action.cpu().numpy())
            rewards[step]=torch.Tensor(reward).view(-1)
            next_done=np.logical_or(termination,truncation)
            print("termination     ", termination)
            print("truncation      ", truncation)
            next_obs=torch.Tensor(next_obs).to(device) 
            next_done=torch.Tensor(next_done).to(device)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)


        with torch.no_grad():
            next_value=agent.get_value(next_obs).reshape(1,-1)
            advantage=torch.zeros_like(rewards).to(device)
            gae=0
            for t in reversed(range(len(rewards))):
                
                if t==args.num_step-1:
                    nextisnonterminal=1.0-dones[t]
                    nextvalues=next_value
                else:
                    nextisnonterminal=1.0-dones[t]
                    nextvalues=values[t+1]
                delta=rewards[t]+args.gamma*nextvalues*nextisnonterminal-values[t]
                advantage[t]=gae=delta+args.lambd*args.gamma*nextisnonterminal*gae
            returns=advantage+values




        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantage.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        b_index=np.arange(args.batch_size)
        
        clipfracs=[]
        
        for epoch in range(args.num_epoch):
            np.random.shuffle(b_index)
            for start in range(0,args.batch_size,args.minibatch_size): #we will take a gradient step in ezch mini batch 
                end=start+args.minibatch_size
                mb_indexes=b_index[start:end]
                
                #print(b_obs[mb_indexes])
                _,newlogprob,entropy,newvalue=agent.get_action_and_value(b_obs[mb_indexes],b_actions[mb_indexes])
                #print (entropy,"     ",newlogprob,newvalue)
                log_ratio   =(newlogprob - b_logprobs[mb_indexes])
                ratio=torch.exp(log_ratio)
########################################################################################################################################
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_indexes]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8) 
########################################################################################################################################

                pg_loss1=-mb_advantages*ratio
                pg_loss2=-mb_advantages*torch.clamp(ratio,1-args.clip_coef,1+args.clip_coef)
                pg_loss_clip=torch.max(pg_loss1,pg_loss2).mean()


                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_indexes]) ** 2
                    v_clipped = b_values[mb_indexes] + torch.clamp(
                        newvalue - b_values[mb_indexes],### clip the change , we want to clip the change not the values themselves
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_indexes]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_indexes]) ** 2).mean()

                entropy_loss=entropy.mean()
                loss=pg_loss_clip+args.value_function_coef*v_loss-args.entropy_coef*entropy_loss

                #print(loss,"      ",loss.shape)
                optimizer.zero_grad()
                #entropy_loss.backward()
                #pg_loss.backward()
                #v_loss.backward()
                loss.backward()

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
    writer.close()


                
            



        









        






import argparse
import os
from distutils.util import strtobool
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def layer_init(layer,std=np.sqrt(2),bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self,envs):
        super(Agent,self).__init__()
        self.critic=nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1),std=1.),
        )
        self.actor=nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,envs.single_action_space.n),std=0.01),
        )

    def get_value(self,x):
        return self.critic(x)

    def get_action_and_value(self,x,action=None):
        logits=self.actor(x)
        probs=Categorical(logits=logits)
        if action is None:
            action=probs.sample()
        return action,probs.log_prob(action),probs.entropy(),self.critic(x)




def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exp-name',type=str,default=os.path.basename(__file__).rstrip(",py"),
        help='the name of this experemnt ')
    parser.add_argument('--gym-id',type=str,default="CartPole-v1",
        help='the id of the gym invironment ')
    parser.add_argument('--learning-rate',type=float,default=2.5e-4,
        help='the learning rate of the optimizer ')
    parser.add_argument('--total-timesteps',type=int,default=25000,
        help='total tiesteps of the experiment')
    parser.add_argument('--seed',type=int,default=1,
        help='the seed of the experiment ')
    parser.add_argument('--torch-deterministic',type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
        help='if toggled  torch.backends.cudnn.deterministic=False ')
    parser.add_argument('--cuda',type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--capture-video',type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True,
        help='capture and sae videos to ''videos'' folder ')
    parser.add_argument('--anneal-lr',type=lambda x:bool(strtobool(x)),default=True,
        help='change lr so it gets to zero when doing the last update')
    
    

    parser.add_argument('--num-envs',type=int,default=4,
        help='numbee of parallel game envirnments')
    parser.add_argument('--num-steps',type=int,default=128,
        help='number of steps in each environment per policy rollout')
    parser.add_argument('--gae',type=lambda x:bool(strtobool(x)),default=True,
        help='calculate Advantage using GAE')
    parser.add_argument('--gamma',type=float,default=0.99,
        help='discount factor gamma')
    parser.add_argument('--gae-lambda',type=float,default=0.95,
        help='lambda for gae') 
    parser.add_argument('--num-minibatches',type=int,default=4,
        help='numbee of minibatches')   
    
    parser.add_argument('--update-epoches',type=int,default=4,
        help='numbee of minibatches')
    parser.add_argument('--norm-adv',type=lambda x:bool(strtobool(x)),default=True,
        help='Normalize the Advantage')    
    parser.add_argument('--clip-coef',type=float,default=0.2,
        help='clipping coef') 
    parser.add_argument('--clip-vloss',type=lambda x:bool(strtobool(x)),default=True,
        help='if TRue clip loss')    
    parser.add_argument('--ent-coef',type=float,default=0.01,
        help='entropy coef')   
    parser.add_argument('--vs-coef',type=float,default=0.5,
        help='value loss coef')   
    parser.add_argument('--max-grad-norm',type=float,default=0.5,
        help='max norm for gradient clipping')
    
    parser.add_argument('--target-kl',type=float,default=None,
        help='KL threshold')   

    args=parser.parse_args()
    args.batch_size=int(args.num_envs*args.num_steps)
    args.minibatch_size=int(args.batch_size//args.num_minibatches)
    return args
    
if __name__=="__main__":
    args=parse_args()
    run_name=f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer=SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key,value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=args.torch_deterministic

    device=torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu") 

    print("device    ",device)

    envs=gym.vector.SyncVectorEnv(
        [make_env(args.gym_id,args.seed+i,i,args.capture_video,run_name)
    for i in range(args.num_envs)])

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "not discrete action space! only Discrete action space"


    agent=Agent(envs).to(device)

    optimizer=optim.Adam(agent.parameters(), lr=args.learning_rate,eps=1e-5)



    obs=torch.zeros((args.num_steps,args.num_envs)+envs.single_observation_space.shape).to(device)
    actions=torch.zeros((args.num_steps,args.num_envs)+envs.single_action_space.shape).to(device)
    logprobs=torch.zeros((args.num_steps,args.num_envs)).to(device)
    rewards=torch.zeros((args.num_steps,args.num_envs)).to(device)
    dones=torch.zeros((args.num_steps,args.num_envs)).to(device)
    values=torch.zeros((args.num_steps,args.num_envs)).to(device)
    
    global_step=0
    start_time=time.time()
    next_obs=torch.Tensor(envs.reset()).to(device)
    next_done= torch.zeros(args.num_envs).to(device)
    num_updates=args.total_timesteps//args.batch_size

    for update in range(1,num_updates+1):
        print("update   ",update)
        #Annealing le
        if args.anneal_lr:
            frac=1.0-(update-1.0)/num_updates
            lrnow=frac*args.learning_rate
            optimizer.param_groups[0]["lr"]=lrnow
        

        for step in range(0,args.num_steps):
            global_step+=1*args.num_envs
            obs[step]=next_obs
            dones[step]=next_done

            with torch.no_grad():
                action,logprob,_,value=agent.get_action_and_value(next_obs)
                values[step]=value.flatten()
            actions[step]=action
            logprobs[step]=logprob


            next_obs,reward,done,info=envs.step(action.cpu().numpy())
            rewards[step]=torch.tensor(reward).to(device).view(-1)
            next_obs,next_done=torch.Tensor(next_obs).to(device),torch.Tensor(done).to(device)
            """
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break"""
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
                # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        b_inds=np.arange(args.batch_size)
        clipfracs=[]
        for epoch in range(args.update_epoches):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]


                _, newlogprob,entropy,new_value=agent.get_action_and_value(
                    b_obs[mb_inds],b_actions.long()[mb_inds]
                )
                logratio=newlogprob-b_logprobs[mb_inds]
                ration=logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ration - 1) - logratio).mean()
                    clipfracs += [((ration - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages=b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages=(mb_advantages-mb_advantages.mean())/(mb_advantages.std()+1e-8)
            
            

                pg_loss1=-mb_advantages*ration
                pg_loss2=-mb_advantages*torch.clamp(ration,1-args.clip_coef,1+args.clip_coef)
                pg_loss=torch.max(pg_loss1,pg_loss2).mean()
                #LOSS
                new_value=new_value.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped=(new_value-b_returns[mb_inds])**2
                    v_clipped=b_values[mb_inds]+torch.clamp(
                        new_value-b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped=(v_clipped-b_returns[mb_inds])**2
                    v_loss_max=torch.max(v_loss_unclipped,v_loss_clipped)
                    v_loss=0.5*v_loss_max.mean()
                else:
                    v_loss=0.5*((new_value-b_returns[mb_inds])**2).mean()
                entropy_loss=entropy.mean()
                loss=pg_loss-args.ent_coef*entropy_loss+v_loss*args.vs_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            """if args.target_kl is not None:
                if approx_kl>args.target_kl:
                    break"""
        y_pred,y_true=b_values.cpu().numpy(),b_returns.cpu().numpy()
        var_y=np.var(y_true)
        explained_var=np.nan if var_y==0 else 1-np.var(y_true-y_pred)/var_y


    envs.close()
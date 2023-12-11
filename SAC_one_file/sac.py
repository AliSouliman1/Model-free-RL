import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from distutils.util import strtobool

import gymnasium as gym
import wandb

import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, 
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    args = parser.parse_args()
    # fmt: on
    return args

class Network(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.fc1=nn.Linear(np.array((env.single_observation_space.shape)).prod()+np.prod(env.single_action_space.shape),256)
        self.fc2=nn.Linear(256,256)
        self.fc3=nn.Linear(256,1)



    def forward(self,obs,action):
        inn=torch.cat([obs,action],1)
        out=self.fc1(inn)
        out=F.relu(out)
        out=self.fc2(out)
        out=F.relu(out)
        out=self.fc3(out)
        return out
    
def make_env(env_id,seed,idx,record_video,run_name):
    def fun():
        if record_video and idx==0:
            env=make_env(env_id,render_mode="rge_array")
            env=gym.wrappers.RecordVideo(env)
        else:
            env=gym.make(env_id)
        env=gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return fun

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self,env):
        super().__init__()
        self.fc1=nn.Linear(np.array(env.single_observation_space.shape).prod(),256)
        self.fc2=nn.Linear(256,256)
        self.mu=nn.Linear(256,np.prod(env.single_action_space.shape))
        self.log_sigma=nn.Linear(256,np.prod(env.single_action_space.shape))
        self.env=env
        self.max_action=torch.tensor(self.env.action_space.high,dtype=torch.float32).to(device)
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )


 
    def forward(self,obs):
        out=self.fc1(obs)
        out=F.relu(out)
        out=self.fc2(out)
        out=F.relu(out)
        mu=self.mu(out)
        log_sigma=self.log_sigma(out)
        log_sigma = torch.tanh(log_sigma)
        log_sigma = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sigma + 1)
        return mu,log_sigma 
    
    def get_action(self,observation):
        mu,log_sigma= self(observation)
        sigma=torch.exp(log_sigma)
        #print("mu",mu,"   sigma",sigma)
        des=Normal(mu,sigma)
        action=des.rsample()

        squashed_action=torch.tanh(action)
        ############action_rescaled=squashed_action*self.max_action#############
        action_rescaled=squashed_action*self.action_scale + self.action_bias

        log_prob_action=des.log_prob(action)
        log_prob_action -= torch.log(self.action_scale * (1 - squashed_action.pow(2)) + 1e-6)
        log_prob_action = log_prob_action.sum(1, keepdim=True)
        return action_rescaled, log_prob_action


    

if __name__ == "__main__":

    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:

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
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #print("device  ",device)
    #env=gym.vector.SyncVectorEnv([make_env(args.env_id,args.seed,idx,args.record_video,args.run_name) for idx in range(args.num_envs)])

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    actor=Actor(envs).to(device)
    q1=Network(envs).to(device)
    q2=Network(envs).to(device)
    q1_target=Network(envs).to(device)
    q2_target=Network(envs).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    q_optimizer=optim.Adam(list(q1.parameters())+list(q2.parameters()), lr=args.q_lr)
    actor_optimizer=optim.Adam(actor.parameters(),lr=args.policy_lr)
    global_step=0
#######
    envs.single_observation_space.dtype = np.float32
######
#replayBuffer
    replay_buffer=ReplayBuffer(args.buffer_size,
                            envs.single_observation_space,
                            envs.single_action_space,
                            device,
                            handle_timeout_termination=False
                           
                           )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        action,_=actor.get_action(torch.Tensor(obs).to(device))
        action=action.detach().cpu().numpy()
        obs_,reward,terminated,truncated,info=envs.step(action)
        next_obs=obs_.copy()
        for ind,trunc in enumerate(truncated):
            if trunc:
                next_obs[ind]=info("final_observation"[ind])
        

        replay_buffer.add(obs,next_obs,action,reward,terminated,info)
        obs=next_obs
        if global_step>=args.batch_size:
            with torch.no_grad():
                data=replay_buffer.sample(args.batch_size)
                next_action,next_log_prob=actor.get_action(data.next_observations)
                #print("action on  ",next_action.device)
                #print("data.nxt.ob on  ",data.next_observations.device)
                q1_value_next=q1_target(data.next_observations,next_action)
                q2_value_next=q2_target(data.next_observations,next_action)
                q_value_next=torch.min(q1_value_next,q2_value_next)
                q_hat=data.rewards+args.gamma*(q_value_next-next_log_prob)
            q1_value=q1(data.observations,data.actions)
            q2_value=q2(data.observations,data.actions)

            q1_loss=F.mse_loss(q1_value,q_hat)
            q2_loss=F.mse_loss(q2_value,q_hat)
            q_loss=q1_loss+q2_loss

            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()
            

            pi, log_pi= actor.get_action(data.observations)
            qf1_pi = q1(data.observations, pi)
            qf2_pi = q2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((args.alpha * log_pi) - min_qf_pi).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        if global_step%100==0:
                print("SPS:", int(global_step / (time.time() - start_time)))


    envs.close()
    writer.close()
                
                

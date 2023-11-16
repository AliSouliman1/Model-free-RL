import os
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self,beta, input_shape,n_actions,fc1=256,fc2=256,
                 name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork,self).__init__()

        self.input_shape=input_shape
        self.n_actions=n_actions
        self.fc1=fc1
        self.fc2=fc2
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.critic=nn.Sequential(
            nn.Linear(input_shape[0]+n_actions,fc1),
            nn.ReLU(),
            nn.Linear(self.fc1,self.fc2),
            nn.ReLU(),
            nn.Linear(self.fc2,1)
        )



        self.optimizer=optim.Adam(self.parameters(),lr=beta)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self,state,action):
        #print(f"state: {state.shape}, action: {action.shape}")
        return self.critic(torch.cat([state,action],dim=2))
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    

class ValueNetwork(nn.Module):
    def __init__(self,beta, input_shape,fc1=256,fc2=256,
                 name="Value",chkpt_dir='tmp/sac'):
        super(ValueNetwork,self).__init__()
        self.input_shape=input_shape
        self.fc1=fc1
        self.fc2=fc2
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.Value=self.Value=nn.Sequential(
            nn.Linear(*input_shape,fc1),
            nn.ReLU(),
            nn.Linear(self.fc1,self.fc2),
            nn.ReLU(),
            nn.Linear(self.fc2,1)
        )


        self.optimizer=optim.Adam(self.parameters(),lr=beta)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self,state):
        return self.Value(state)


    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))






class ActorNetwork(nn.Module):
    def __init__(self,alpha,input_shape,n_actions,max_action,fc1=256,fc2=256,
                 name="actor", chkpt_dir='tmp/sac'):
        super(ActorNetwork,self).__init__()
        self.input_shape=input_shape
        self.n_actions=n_actions
        self.max_action=max_action
        self.fc1=fc1
        self.fc2=fc2
        self.reparam_noise = 1e-6
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.actor=nn.Sequential(
            nn.Linear(*input_shape,fc1),
            nn.ReLU(),
            nn.Linear(self.fc1,self.fc2),
            nn.ReLU()
        )

        self.mu=nn.Linear(self.fc2,n_actions)
        self.sigma=nn.Linear(self.fc2,n_actions)


        self.optimizer=optim.Adam(self.parameters(),lr=alpha)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self,state):
        mu=self.mu(self.actor(state))
        sigma=self.sigma(self.actor(state))
        #print("mu_Before= ",mu)
        #print("sigma_Before= ", sigma)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu,sigma
    
    def sample_action(self,state,reparameterize=True):
        mu,sigma=self.forward(state)
        #print("mu_after=",mu,"  ","sigma_After=", sigma)
        des=Normal(mu,sigma)

        if reparameterize:
            actions=des.rsample()
        else:
            actions=des.sample()
        

        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs=des.log_prob(actions)
        #print("in log  ",(1-action.pow(2)+self.reparam_noise))        
        #log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action,log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


        

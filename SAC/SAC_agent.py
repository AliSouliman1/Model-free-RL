import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional
import numpy as np
from SAC_buffer import buffer
from SAC_networks import CriticNetwork,ValueNetwork,ActorNetwork

class Agent(nn.Module):
    def __init__(self, alpha=0.0003, beta=0.0003, input_shape=[8],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        super(Agent,self).__init__()
        self.alpha=alpha
        self.beta=beta
        self.input_shape=input_shape
        self.env=env
        self.gamma=gamma
        self.n_actions=n_actions
        self.batch_size=batch_size
        self.tau=tau
        self.memory=buffer(1000000,input_shape,n_actions)
        self.actor=ActorNetwork(alpha,input_shape,n_actions,max_action=env.action_space.high,
                                name="actor")
        self.Val=ValueNetwork(beta,input_shape,name="Value")
        self.Val_target=ValueNetwork(beta,input_shape,name="Value_target")

        self.Q1=CriticNetwork(beta,input_shape,n_actions,name="critic1")
        self.Q2=CriticNetwork(beta,input_shape,n_actions,name="critic2")



        self.scale = reward_scale
        self.update_network_parameters(tau=1)


    def choose_action(self,state):
        state=torch.Tensor([state]).to(self.actor.device)
        action,_=self.actor.sample_action(state)
        return action.cpu().detach().numpy()[0]
    
    def save_to_buffer(self,state,new_state,action,reward,done):
        self.memory.save_transition(state,new_state,action,reward,done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.Val_target.named_parameters()
        value_params = self.Val.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.Val_target.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.Val.save_checkpoint()
        self.Val_target.save_checkpoint()
        self.Q1.save_checkpoint()
        self.Q2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.Val.load_checkpoint()
        self.Val_target.load_checkpoint()
        self.Q1.load_checkpoint()
        self.Q2.load_checkpoint()

    def learn(self):
        if self.batch_size>self.memory.counter:
            return
        
        states,new_states,actions,rewards,dones=self.memory.sample(self.batch_size)

        states=torch.Tensor([states]).to(self.actor.device)
        new_states=torch.Tensor([new_states]).to(self.actor.device)
        actions=torch.Tensor([actions]).to(self.actor.device)
        rewards=torch.Tensor([rewards]).to(self.actor.device)
        dones=torch.Tensor([dones]).to(self.actor.device)


        ##Value function networks training
        value=self.Val(states) #this is used to train the value network but not the value target network
        target_value=self.Val_target(states).view(-1)# this is used to train the critic network, and to calculate the actual values of states
        
        action_from_policy,log_prob=self.actor.sample_action(states,reparameterize=False)
        q1_new_policy=self.Q1(states,action_from_policy)
        q2_new_policy=self.Q2(states,action_from_policy)
        Q_value=torch.min(q1_new_policy,q2_new_policy)
        Q_value=Q_value.view(-1)
        self.Val.optimizer.zero_grad()
        value_for_loss=Q_value-log_prob
        value = value.permute(0, 2, 1)
        value_loss=0.5*nn.functional.mse_loss(value,value_for_loss)
        value_loss.backward(retain_graph=True)
        self.Val.optimizer.step()


    ##Actor Network Training
    #
        action_from_policy,log_prob=self.actor.sample_action(states,reparameterize=True)
        q1_new_policy=self.Q1(states,action_from_policy)
        q2_new_policy=self.Q2(states,action_from_policy)
        q_value=torch.min(q1_new_policy,q2_new_policy)
        q_value=q_value.view(-1)


        actor_loss=torch.mean(log_prob-q_value)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()




        self.Q1.optimizer.zero_grad()
        self.Q2.optimizer.zero_grad()
        q_hat = self.scale*rewards + self.gamma*target_value
        q1_old_policy = self.Q1.forward(states, actions).view(-1)
        q2_old_policy = self.Q2.forward(states, actions).view(-1)
        q1_old_policy = q1_old_policy.unsqueeze(0)  # Add a dimension, size will be [1, 256]
        q2_old_policy = q2_old_policy.unsqueeze(0)  # Add a dimension, size will be [1, 256]

        critic_1_loss = 0.5 * nn.functional.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * nn.functional.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.Q1.optimizer.step()
        self.Q2.optimizer.step()

        self.update_network_parameters()


    
    
        


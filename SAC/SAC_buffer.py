import numpy as np



class buffer():
    def __init__(self,max_size,input_shape,n_actions,batch_size=256):
        self.mem_size=max_size
        self.counter=0
        self.states=np.zeros((max_size,*input_shape))
        self.actions=np.zeros((max_size,n_actions))
        self.new_states=np.zeros((max_size,*input_shape))
        self.dones=np.zeros (max_size,dtype=bool)
        self.rewards=np.zeros(max_size)
        self.batch_size=batch_size


    def save_transition(self,state,state_,action,reward,done):
        index=self.counter%self.mem_size

        self.states[index]=state
        self.new_states[index]=state_
        self.actions[index]=action
        self.rewards[index]=reward
        self.dones[index]=done

        self.counter+=1


    def sample(self,batch_size):
        if self.counter<self.batch_size:
            return
        size=min(self.mem_size,self.counter)
        indexes=np.random.choice(size,batch_size)
        states=self.states[indexes]
        new_states=self.new_states[indexes]
        actions=self.actions[indexes]
        rewards=self.rewards[indexes]
        dones=self.dones[indexes]

        return states,new_states,actions,rewards,dones


    
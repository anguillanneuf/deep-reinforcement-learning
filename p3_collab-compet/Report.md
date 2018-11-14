# Project Report: Collaboration and Competition

![alt text][collab-compet-gif]

Here is a [video](https://youtu.be/YkIKTJ8ZUU0) showing my agent in the reacher environment (the second version with 20 arms). 

## Learning Algorithm

The learning algorithm used in the project is adapted from the lab [`MADDPG`](https://github.com/anguillanneuf/deep-reinforcement-learning/tree/master/MADDPG), which stands for Multi-Agent Deep Deterministic Policy Gradient method. It is an extension of the DDPG method. Instead of training one actor and critic network for a single agent, MADDPG trains pairs of actor and critic networks for multiple agents. Each agent decides which actions to take and predicts its own score in an episode. In order to decide the actions, the agent only uses the states returned from the Unity environment. But in order to predict scores, the agent makes use of both the states returned from the environment and the actions they select. 

### Hyperparameters and Pseudocode

```
M                  = 40000
T                  = 1000
BUFFER_SIZE        = 100000
BATCH_SIZE         = 256
NOISE              = 1.0
NOISE_REDUCTION    = 1.0
GAMMA              = 0.95 
TAU                = 0.02
LR_CRITIC          = 1e-3
LR_ACTOR           = 1e-3

Initialize replay buffer R of BUFFER_SIZE
Initialize two agents, for each agent 
    Initialize local critic network Q with random weights θ
    Initialize local actor network μ with random weights φ   
    Copy θ to target critic network Q' with random weights θ' 
    Copy φ to target actor network μ' with random weights φ'

For episode=1, M do
    For t=1, T do
        Reset noise for both agents (to encourage exploration)
        Select a set of a_t using both local actor networks with some added noise η
        Execute the set of (a_t + η) and observe a set of (r_t, s_(t+1))
        Store (s_t, (a_t + η), r_t, s_(t+1), done) in R, where each item has a length of 2

        If the size of R is greater than BUFFER_SIZE
        Sample a random minibatch of BATCH_SIZE from R
        
        Obtain a set of a_(t+1)' from both target actor networks 
        Concatenate them with the set of s_(t+1)
        Feed the concatenated input to the set of target critic network and obtain a set of r_(t+1)' 
        Calculate a set of y_t = r_t + GAMMA * (r_(t+1)')
        Concatenate the set of (a_t + η) with the set of s_(t+1)
        Feed the concatenated input to the set of local rewards and obtain a set of local rewards 
        Calculate a set of smooth L1 losses between of y_t and local reward
        Perform gradient descent on both local networks with learning rate LR_CRITIC and update φ 
        
        Obtain a set of a_t from the set of local actor networks and s_t
        Concatenate them with the set of s_t
        Feed the concatenated input to the set of local crtic networks to obtain a set of local rewards
        Calculate a set of actor losses L_a = - local rewards
        Perform gradient descent using L_a with learning rate LR_ACTOR and update θ         
        
        Update the local actor and critic weights 10 times every episode, then
        Reset Q'= Q using a soft update of TAU
        Reset μ'= μ using a soft update of TAU
    END FOR
END FOR
```

### Actor and Critic Networks

As shown in [`networkforall.py`](networkforall.py), both the actor and critic networks use ReLU for all the hidden layers. They have almost identical architectures except for the input and output layers. The input layer to the actor network has a shape of (batch_size, 24) whereas the input layer to the critic network has a shape of (batch_size, 24+24+2+2). The output of the actor network is a `tanh` layer, which is to bound the actions between -1 and 1. In contrast, the output of the critic layer is a single scalar.  Both networks have two hidden layers with 256 and 128 nodes respectively, and the same learning rate of 1e-3. The final layer weights and biases are initialized from a uniform distribution [-3e-3, -3e-3], as used in the DDPG model introduced in the 2016 Google Deepmind paper ["Continuous Control With Deep Reinforcement Learning"](https://arxiv.org/abs/1509.02971). 

## Plot of Rewards

I successfully solved the environment in 2919 episodes by achieving an average score above 0.5. The pink line in the plot below shows the average score of a window of 100 episodes. The light blue line shows the maximum score for each episode. 

![alt text][rewards] 

## Future Work

I didn't get `torch.nn.BatchNorm1d` to work in `networkforall.py`. I can see that the values returned from the environment are not bounded between -1 and 1. By scaling the input to the critic network, I believe that training should be improved significantly. This is something that I will try. 

Something else I want to try is setting up the Unity environment in parallel so that I can fill up the replay buffer faster, and train the agents to learn different strategies instead of getting comfortable and stuck in one strategy. The code sample for parallel training in the lab looks like nothing I recognize at the moment, but I can see how parallel training could be key to solving the Soccer game, which I really want to do.  

[collab-compet-gif]: https://github.com/anguillanneuf/deep-reinforcement-learning/blob/master/p3_collab-compet/plots/cc.gif "MADDPG"
[rewards]: https://github.com/anguillanneuf/deep-reinforcement-learning/blob/master/p3_collab-compet/plots/scores.png "Rewards"

# Project Report: Continuous Control

Here is a [video](https://youtu.be/iK_oNxBwWEA) showing my agent in the reacher environment (the second version with 20 arms). 

## Learning Algorithm

The learning algorithm used in the project is adapted from [`ddpg-bipedal`](https://github.com/anguillanneuf/deep-reinforcement-learning/tree/master/ddpg-bipedal), and it is modeled closely after the Deep Deterministic Policy Gradient (DDPG) method outlined in the 2016 Google Deepmind paper ["Continuous Control With Deep Reinforcement Learning"](https://arxiv.org/abs/1509.02971). Making use of both a value-based approach (the critic) and a policy-based (the actor) approach in one model, DDPG improves performance by a reinforcement learning agent by refining the low-bias and high-variance solution driven by *the critic* using the high-biased but low-variance solution driven by *the actor*, to achieve a low-bias and low-variance solution in the end. 

### Hyperparameters and Pseudocode

```
BUFFER_SIZE = 1e6       # replay buffer size
BATCH_SIZE = 1024       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 3e-4         # learning rate of the actor
LR_CRITIC = 6e-4        # learning rate of the critic

Initialize replay buffer R of BUFFER_SIZE
Initialize local critic network Q with random weights θ
Initialize local actor network μ with random weights φ   
Copy θ to target critic network Q' with random weights θ' 
Copy φ to target actor network μ' with random weights φ'

For episode=1, M do
    Initialize a random process η for action exploration
    For t=1, T do
        Select a_t using local actor network μ with some added noise η
        Execute (a_t + η) and observe r_t, s_(t+1)
        Store (s_t, (a_t + η), r_t, s_(t+1), done) in R

        If the size of R is greater than BUFFER_SIZE
        Sample a random minibatch of BATCH_SIZE in R
        
        Obtain A_t from target actor network μ' and s_(t+1)
        Obtain r_(t+1)' from target critic network Q', A_t, and s_(t+1)
        Set projected reward y_t = r_t + GAMMA * (r_(t+1)')
        Obtain expected local reward from local critic network Q, (a_t + η), and s_t
        Calculate critic loss Lc using mean squared error of y_t and local reward
        Perform gradient descent using L_c with learning rate LR_CRITIC and update φ 
        
        Obtain a_t from local actor network μ and s_t
        Obtain local reward from local crtic network Q, a_t, and s_t
        Set actor loss L_a = - local reward
        Perform gradient descent using L_a with learning rate LR_ACTOR and update θ         
        
        Reset Q'= Q using a soft update of TAU
        Reset μ'= μ using a soft update of TAU
    END FOR
END FOR
```

### Actor and Critic Networks

As shown in [`model.py`](model.py), both the actor and critic networks ReLU for all the hidden layers. The final output of the actor network is a `tanh` layer. Both networks have two hidden layers with 256 and 128 nodes respectively. The final layer weights and biases are initialized from a uniform distribution [-3e-3, -3e-3], which "ensures the initial outputs of the policy and value estimates [are] near zero". Additionally, to scale the 33 features so they all exist in the same range (maybe not entirely necessary), batch normalization is used. Lastly, to ensure that the model continues to explore, an Ornstein-Uhlenbeck noise is added to actions obtained from the local actor network.  

## Plot of Rewards

I successfully trained two sets of solutions that passed the targeted score of 30 on Google Cloud Platform. The plot below is trained using all the hyperparameters outlined in the previous section except for `LR_ACTOR = 5e-4` and `LR_ACTOR = 9e-4`. This plot also exists in [`Continuous_Control_GCP.ipynb`](Continuous_Control_GCP.ipynb) where it is original from. The number steps taken to solve in this case is `154`, with an average score over of `30.00`. 

![alt text][rewards]

My second set of solution is trained using the exact same hyperparameters in the previous section, in a headless fashion, as shown in [`main.py`](main.py), and are saved in `checkpoint_actor.pth` and `checkpoint_critic.pth`. The number of steps taken to solve is `106`. Unfortunately, I have failed to record the average score there, and I didn't save the scores. Thus no plot. 

## Future Work

Some students have reported great results using Proximal Policy Optimization (PPO) instead of DDPG. I have sent a request to Hemanta Gupta to see his implementation in the Slack channel. Hope to learn from my classmates! 


[rewards]: https://github.com/anguillanneuf/deep-reinforcement-learning/blob/master/p2_continuous-control/plot.png "Rewards"

# Report 

In this project, I have used Double DQN to train an agent to navigate in a square arena and collect yellow bananas.

[![Banana](https://github.com/anguillanneuf/deep-reinforcement-learning/blob/master/p1_navigation/Screen%20Shot%202018-08-27%20at%2011.38.41%20AM.png)](https://youtu.be/mG1daIuP2CU) 

The environment is set up so that the agent gets a reward of +1 for collecting a yellow banana, and a reward of -1 for collecting a blue banana. The goal is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions. They contain information such as the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent learns how to best select actions from four discrete actions: forward, backward, left, and right. 

The task is episodic. The environment is considered solved when the agent gets an average score of +13 over 100 consecutive episodes.

## Learning Alogrithm  

Here is some pseudocode outlining the Double DQN implementation detail I used for the project, along with the hyperparameters I used. This is based on the [paper](https://arxiv.org/abs/1509.06461) "Deep Reinforcement Learning with Double Q-learning" by Hado van Hasselt, Arthur Guez, David Silver, introduced in the Nanodegree Program. 

```
BUFFER_SIZE = 1e5
M           = 1000
T           = 2000
BATCH_SIZE  = 64
GAMMA       = 0.99
LR          = 5e-4
C           = 4
TAU         = 1e-3

Initialize replay memory D of BUFFER_SIZE
Initialize online action-value function Q with random weights θ
Initialize target action-value function Q' with random weights θ'
For episode=1, M do
    For t=1, T do
        With probability ε select a random action a0
        Otherwise select a0=argmax Q(s0, a; θ)
        Execute a0 and observe r0, s1
        Store (s0, a0, r0, s1, done) in D

        If the size of D is greater than BATCH_SIZE
        Sample a random minibatch of BATCH_SIZE in D
        For each batch (s_i, a_i, r_i, s_j, done)
        Caculate TD_target
            if s_j is the terminating episode
                TD_target = r_i
            else
                TD_target = r_i + GAMMA * Q'(s_j, argmax Q(s0, a; θ); θ')
        Calculate TD_diff = TD_target - Q(s_i, a_i; θ)
        Perform a gradient descent step with learning rate LR and update θ
        Every C steps reset Q'= Q using a soft update of TAU
    END FOR
END FOR
```

In terms of my online and target action-value function Q and Q', they share the same neural network architecture as shown in `model.py`. The neural network is a simple feed-forward neural network. It has an input layer of size 37, two hidden layers of size 64, and an output layer of 4, which correspond to the values associated with each valid action. The model makes use of a rectified linear unit (ReLU) as the activation function between the fully connected layers. 

## Plot of Rewards 

As presented in `Navigation.ipynb`, the environment was solved in 386 episodes with	an average score of 13.02 calculated from a score window of size 100. The plot below shows the improvement of the average scores during training. 

![Plot](https://github.com/anguillanneuf/deep-reinforcement-learning/blob/master/p1_navigation/PlotOfReward.png)

## Ideas for Future Work

I have done minimum work to adapt the DQN we implemented for the Lunar Lander project to a double DQN architecture for this project. I spent majority of my time understanding Double DQN. I am aware of other techniques that could further improve performance, such as prioritized experience replay and dueling DQN. Those are the ones I want to implement and compare with vanilla DQN and double DQN.  

I am also really interested in rewriting the code using TensorFlow estimators, in order to train it on Google Cloud Machine Learning Engine (CMLE), and perhaps use the hyperparameter tuning feature in CMLE for finding the best hyperparameters, instead of tuning it by hand. Stay tuned!

# Use Google Cloud Platform 

I selected a Compute Engine instance with `1 x NVIDIA Tesla P100` in `us-east1-b`. 

I loaded it with the image below available in Compute Engine/Images. 

```
c1-deeplearning-pytorch-0-4-cu92-20181009

Description
    Google, Deep Learning Image: 
        PyTorch 0.4.1
        m9 CUDA 9.2
        A Debian based image with PyTorch (CUDA 9.2) and Intel® optimized NumPy, SciPy, and scikit-learn.
Labels
    None
Creation time
    Oct 9, 2018, 8:40:16 PM
Family
    pytorch-0-4-gpu
Encryption type
    Google managed
```

Here are the commands I needed to set up my Compute Engine on Google Cloud Platform for training the network. 

```
# Generate SSH public key for my GitHub account. 
cd ~/.ssh/
ssh-keygen
cat ~/.ssh/id_rsa.pub
# Copy key in https://github.com/settings/keys


# Start virtualenv
virtualenv --python python3 py3
source py3/bin/activate


# Get gym 
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[box2d]'
pip install -e '.[atari]'
pip install -e '.[classic_control]'


# Get the project. 
git clone git@github.com:anguillanneuf/deep-reinforcement-learning.git
cd python/
pip install .


# Get the headless reacher environment. 
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip
unzip Reacher_Linux_NoVis.zip
mv Reacher_Linux_NoVis/ deep-reinforcement-learning/p2_continuous-control/

```

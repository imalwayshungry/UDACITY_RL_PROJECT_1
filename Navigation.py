#MSA
#Clean up and ready to SUBMIT

#!/usr/bin/env python
# coding: utf-8

# # Navigation
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
# 
# ### 1. Start the Environment
# 
# We begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:


from unityagents import UnityEnvironment
import numpy as np


# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Banana.app"`
# - **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
# - **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
# - **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
# - **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
# - **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
# - **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Banana.app")
# ```

# In[2]:


#env = UnityEnvironment(file_name="...")
env = UnityEnvironment(file_name="Banana.app")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
# - `0` - walk forward 
# - `1` - walk backward
# - `2` - turn left
# - `3` - turn right
# 
# The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Once this cell is executed, you will watch the agent's performance, if it selects an action (uniformly) at random with each time step.  A window should pop up that allows you to observe the agent, as it moves through the environment.  
# 
# Of course, as part of the project, you'll have to change the code so that the agent is able to use its experience to gradually choose better actions when interacting with the environment!

# In[ ]:

from dqn_agent import ReplayBuffer
from dqn_agent import Agent


LOAD_PREV_MODEL = False
SET_EPS = False
MODEL_FILE_NAME = "MODEL_CHECKPOINT.34994.model"

max_games = 1000
max_steps = 1000

my_agent = Agent(37, action_size, 0)

if LOAD_PREV_MODEL:
    my_agent.load_model(MODEL_FILE_NAME, 37, action_size, 0)

#my_buffer = ReplayBuffer(4, 1000, 10, 0)
epsilon = 0.0
min_eps = 0.1
all_scores = []

#env_info = env.reset(train_mode=False)[brain_name] # reset the environment
#state = env_info.vector_observations[0]            # get the current state
#score = 0           # initialize the score

for idx_games in range(1, max_games):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    print("GAME NUMBER: " + str(idx_games))
    if idx_games % 100 == 0:
        print("WRITING MODEL.")
        my_agent.save_model()
    if SET_EPS:
        if idx_games % 100 == 0 and epsilon >= min_eps:
            print("DECAYING EPS: " + str(epsilon))
            epsilon = epsilon - 0.1
    for steps_idx in range(0, max_steps):
        #action = np.random.randint(action_size)        # select an action
        action = my_agent.act(state, epsilon)  #***simply returns actions, from local policy network.
        #print("ACTION: " + str(action))
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        if reward == -1:
            reward = -50
        if reward == 1:
            reward = 50
        my_agent.step(state, action, reward, next_state, done)
        state = next_state
        if done:   # exit loop if episode finished
            all_scores.append(score)
            break
    print("Score: {}".format(score))

print("Saving model")
my_agent.save_model()
from matplotlib import pyplot as plt
plt.plot(all_scores)
plt.show()
# When finished, you can close the environment.

# In[ ]:


env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```

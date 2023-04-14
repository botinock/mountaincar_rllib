# mountaincar_rllib
Train RL agent using Ray RLLib to solve mountain car with discrete action space problem.

## Problem statement
Mountain car https://gymnasium.farama.org/environments/classic_control/mountain_car/  
MDP with discrete action space. So Model-free off/on policy algorithms are ideal for this task.  
I used DQN and PPO models as stable off-policy and on-policy baselines.

## Reward function
Base reward is a simple penalty -1 each step. With maximum lenght of episode agent will receive -200 return if it does not reach the goal.  
Since a random agent reaching the goal probability is very close to zero, we can say that this is hard to exploration task. So an ordinary algo as a DQN with epsilon greedy strategy will never train and solve the task.  
There are many approaches to solve hard to exploration tasks:  
* Manual intristic reward
* Intristinc reward models
* Prerecorded human trajectories
Model-based algorithms are also can be used but with above approaches it will be more effective.
I used itristic reward model called ICM (intristic curiosity model) but it did not work for me. But manual reward from observations worked fine.

### Intristic rewards
I used two similar rewards.
- Absolute value of velocity
- Absolute value differential of y coordinate, calculated as $dy = \sqrt{v^2 - dx^2}$

## Results
### Trained agent demo
![](https://github.com/botinock/mountaincar_rllib/blob/master/demo/trained_demo.gif)

### Metrics
![](https://github.com/botinock/mountaincar_rllib/blob/master/plots/mean_reward.png)
![](https://github.com/botinock/mountaincar_rllib/blob/master/plots/min_reward.png)
![](https://github.com/botinock/mountaincar_rllib/blob/master/plots/max_reward.png)
![](https://github.com/botinock/mountaincar_rllib/blob/master/plots/episode_lenghts.png)

We can see agent started reaching goal at about 200000 step.
DQN works better than PPO and velocity based reward works better than y-based reward for this task with parameres you can see in yaml files.

From the certain moment DQNs started to degradate from overfitting.

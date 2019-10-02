(there is a great article by karpathy btw here http://karpathy.github.io/2016/05/31/rl/, about playing pong with RL)
lets have a look at the files
### dqn.py


QLearner
DONE (1) __init__
- do I need to build tf model here?
it is called with q_func, what is this? the approximation function for q values

Bellman error:
what we are minimising in the fitted Q-iteration
full fitted Q-iterationn algorithm:
1. collect trajectories
- 2. set y(i) <= r(s(i), a(i)) + gamma * max(action) (Q(s(i+1), action))
this y(i) is the target value
- 3. minimise Bellman Error: E = Q(s(i), a(i)) - y(i)

stopping_criterion_met

DONE (2) step_env
this part is interesting since is completely uncorrelated from the learning that happens completely asyncronosuly
also interesting to see how to use tensorflow,with run session and self.best_action

(3) update_model
log_progress

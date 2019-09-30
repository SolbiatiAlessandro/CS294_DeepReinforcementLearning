Q LEARNING
====
1. Only critic, not actor
2. Extract policy from value function
3. Q-learning

omit policy gradient completeliy, just learn Value Function.
do the argmax of the Advantage function
fit Advantage function, sample new policy using argmax over Advantage function

instead of learning Value function you learn Q function, more complicated but you need to do a argmax over value function (non feasble if you can only sample trajectories)

off policy
model free

Fitted Q-iteration
1. collect dataset using SOME policy
2. compute target value
3. train neural network to regress on target values

Collection Policy:
policy from point 1 does not necessarely need to be the one learned with 2 and 3. So how does collection policy depends on learned Q? that's a delicate questio

exploration: 
epsilon greedy
boltzman exploration (temperature of the actions): you don't explore super bad trajectories

Fitted Q-value iteration does not converge!
Connverges in the tabular case, because is a restriction. But ot the fitted versino.

ADVANCED Q-LEARNING
========
two problems

1. not gradient descent (target value dependent on Q)
Solution: target network
you decouple target from parameter being learned by not using Q in the target but a old copy of Q (with old parameter). This is a gradient descent (real regressionn)

2. samples temporally correlated
sequential states are correlated. 
Solution: replay buffer. Training data are random sampled from buffer that is populated periodically running the latest policy, not correlated.

this leads to
CLASSIC DQN ALGORITHM
1.  take action, observe (s,a,s+1,r) put it in buffer
2.  sample mini batch from buffer
3.  compute target value y = r + gamma * max * Q(target network)
4.  regress to target value 
5.  update parameter of target network (every N steps)

Another problem: overestimating value (Q-function overestimate value of states), solution: double-DQN
Target Value
Qa <- r + gamma  * Qb ( s+1, argmax Qa (s+1, a+1) )
Qb <- r + gamma  * Qa ( s+1, argmax Qb (s+1, a+1) )
Don't use same network to choose action and evaluate value
you can use current network for Qa and target nentwork for Qb

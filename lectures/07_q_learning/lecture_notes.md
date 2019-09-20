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



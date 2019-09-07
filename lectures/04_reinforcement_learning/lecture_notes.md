Goes through basic SAR notation for RL, goes through basic properties of MDP.

Stationarity of a MDP:
when we say that the 'model converge', it means that the RL agent converge to a stationary distribution. 
A stationary distribution of the transition probabilities. Thus trajectories are stationary.
There are some guarantees over the stationarity, based on 'ergodicity'. (You can't go back to the same state)

Observation on Expectation
the reward function is always non smooth and non differentiable. (+1, -1) reward
This is a bad optimization problem, because you can't compute any gradient.
That's why what you deal with is expectation. The expectation of the reward is smooth and differentiable, and you 
can compute gradient descent on it.

ALGORITHMS:
==========
Always broken down into three parts
1. generate samples (run policy)
2. fit a model (estimate returns)
3. improve policy -> repeat

Policy gradient (model free)
try different trajectories, and make one less or more likely based on how good they are. 
1. run policy(theta)
2. compute J(theta) (objective)
3. compute gradient and update theta

(model based method)
2. learn F(theta) such that s(t+1) = F(state, action)
3. backprop through F and reward to train policty(state) = action
RL by backprop
1. collect data (usually real data  this is expensive part)
2. update model f
3. update policy with backprop
can only deal with discrete policy.

CONDITIONAL EXPECTATION (Q-function)
Q(state, action) 
we can write  expectation in a recursive way
E = r + E(r + E(r + E(...)))
if you now Q = r + E(...)
improving policy can become more straitforward
Q-function for a policy pi

Q(state, action) = sum of expected reward from now to the end executing policy pi

VALUE FUNCTION
V(state, action)
expectation of reward if you start from state
equivalent to say
V(State) = Expected(action according to policy, state)

and the expectation of the value function is the RL objective!


observations on Q-functions and value funcitons:
- if we now Q(s,a) we can improve the policy
taking the argmax of Q values
- we can compute  the gradient to increase probability of good actions:
modify pi(action, state) to increase probability that Q(state, action)  > V(states)

TYPES OF RL ALGORITHMS
=======
(Always maximise expected reward)
- *policy gradient* (directly differentiate policy, sampled based estimation of trajectoris)
apply gradient  of compute  function

- *value-function based*: estimate value function with NN without explicit policy
value is the measure of the expectation (from a sample)

- *actor-critic*: value function to improve policy
value functions + policy gradient

- *model-based RL*: estimate the transitino model
1. used the model to plan
2. backpropagate gradients into the policy
3. use the model to learn a value function

TRADEOFF
======
why there  is not just one thing?
differente tradeoff

- sample efficiency (how fast they converge)
concerend where you don't access a similuator.
how many samples we need to get a good policy?
most important question: is it off policy?
*off policy* => can use sample to improve the policy that were generated not using that policy (more efficient), DQN, model based deep RL, model based shallow RL. why would you use a less efficient algorithm? isn't efficieny always better? "wall clock time is not  the same as sample  efficieny".
*on policy* => all time  policy is changed we need to generate new sample (less effificnet)

- stability an ease of  use 
does the algorithm converge?
what does it converge?
does it converge every time?
why is this even a question, why we care about algorithm that  don't  converege or we don't know what they converge toknow what they converge to. In supervised learning we don't ask this question. Many RL  algorithm are not gradient descent. 
DQN learning is not a gradient descent, is doing  'fixed point iteration' that converge under simple function classes, but is not guaranteed  to  converge under complex function as NN.  Value functino fitting: at best minimises error of fit. at worst does not optimise anything. Value fitting algorithm are not guarnateed to converge, but there are euristic that help you to converge.
Model-based RL is  not  optimised for reward, but optimised  for a more accurate  model. Is  not  guaranteed that a more accurate  model will  yield better reward. model minise error of fit, will converge. but the convergence of the model doesn't give reward covergence.
Policy gradient  is a gradient descent, but is often the least efficient, even if   it convergence.


different assumptions:
- policy stochastic or deterministc?
- states space/actions space continous discrete?
common assumptions #1: full observability
- episodic of infinite horizon MDPs
common assumptions #2: episodic learning
(not true for stock trading)
common assumption #3: smoothness


different things are easy or hard in different settings
- easy to represent a policy 
- easy to represent a model

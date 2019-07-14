## Policy Gradient Algorithm -> Model Free Reinforcement Learing Algorithm
(new homework!)

Plan
1. Policy Gradient 
2. What does it do
3. Variance reduction: Causality
4. Variance reduction: Baselines
5. Examples

Goals:
- understand policy gradient
- unnderstand practical considerations

The goal of RL:
maximise total reward, 
you have a state conditional policy -> function approximator with some parameters, 
policy take a state s and output an action probability distribution
part of a sequential decision process, with an environment (unknown function) that produce the next state given an action

goal of the policy:maximise the reward

(finite horizon case)

### How do we optimize objective?

Objective: expectection of the reward over a really complex multidimensional trajectory (of state and action tuples)

sum of reward <-> trajectory
J(theta) -> expectation over trajectory : objective
tau ~ sequence of (state, action), is a trajectory


Theta-start = arg(max-theta) Expected-over-time[ Sum-time r(s-time, action-time) ]

We need to evaluate objective, can't do it -> need to aproximate J(theta) -> obtain sample based estimate. You generate samples running policy
How do you improve it?
Computing gradient and take steps over the gradient, (we can climb the gradient to improve our policy)

Gradient(J(theta))  .. derivation ..

I can evaluate objective gradient using samples

Once I evaluated the gradient I can do: Theta <- Theta + alpha(Gradient(J(theta)))
where alpha is learing rate

### REINFORCE ALGORITHM:
1. sample trajectories running policy
2. compute gradient for the objective
3. update the policy (parameters) with gradient

Comparison between policy gradient and supervised learning gradient, 
policy gradient is the same but is multiplied by the reward, so is somehow weighted

this can be seen as a formalizatio of the notion of "trial and error"

### Problem with policy gradiennt
Variance, if you keep estimating all the time you get a different estimate since you are using samples.
Why high variance? Because adding a costant (when you multiply by reward) can change the gradient.



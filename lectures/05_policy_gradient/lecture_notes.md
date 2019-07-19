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

## VARIANCE REDUCTION

### CAUSALITY
reduce variance using **causality** -> policy at time t2 can not affect reward at time t where t < t2
"reward to go" -> entirety of the consequence in my decision
why causality reduces variance? -> you are summing less number, if you multiply by lower nunmber gives you lower variance
(policies are time invariant)

### BASELINE
reduce variance using **baseline**
now you are multiplying by rewards, they might be all positive! so you can subtract to every gradient a baseline, you can use the average of all the return for the trajectory.  It is not the best baseline but is pretty good.
is this unbiased in expectation? yes (proof is that expected value of the baseline zero)
How to find best baseline? let's minmize variance of the baseline
Var[x] = E[x^2] - E[x]^2
The best baseline is the expected reward weighted by gradient magnitude

###on policy/off policy
**On Policy**: each time you generate a new policy you generate new sample and you disregard old samples
Policy gradient is on-policy
On-Policy version is really inefficient since you need to regenerate sample all the time you change your policy by a tiny bit

**Off Policy**: importance sampling
let's say we don't have sample but we have sample from a different distribution: I can use importance sampling (computing the ration between the two distributions and then multiplyit by the reward inside the expecation)
if we do it off policy we can save regenerating samples all the times




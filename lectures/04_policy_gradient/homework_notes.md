# Homework 02 - Notes

## Review 

### Formula (1) : objective function

(1) This is the objective function, `tau` is a trajectory.
`tau ~ pisubtetha` means that the trajectories are distributed as the probability distribution `pi` that comes out of the current policy `p`
`pisubtheta(tau)` is the probability that the given trajectory tau will happen
the reward of a trajectory `r(tau)` is just the sum of all the rewards of the states on the trajectory
More formally, `π(θ)` is a probability distribution over the action space, conditioned on the state. 
In the agent-environment loop, the agent samples an action at from πθ(·|st) and the environment responds with a reward r(st,at).

### Formula (2) : gradient definition

We simply take the gradient of the objective function and we apply gradient descent.
How do we compute it? Using a batch of trajectories

### Formula (4): gradient computation

The computation of the gradient in batches is just the gradient in the classical supervised learning problem, just weighted by the reward.
Not super clear yet how to compute since we have `pisubtheta(tau)` where tau is a trajectory. 

### Formula (5): gradient computation with exploded trajectories

Keep in mind that is batches of trajectories `tau`. So here we explode the trajectories so we can understand better what `pisubtheta` means.
This formula is important for the computation so we might as well spend a bit more time on it. Let's examine the single parts.
 - `∇θ log πθ(ait|sit)`, here we are taking the gradient of the logarithm of `pi`. This is because `pisubtheta` *is not a number but is a probability distribution*. The english language meaning of this expression is that the gradient in theta of `pi` evaluated in a specific tuple (action, state) tell us how much that probability would change if we change a bit the action.
- 􏰇`r(sit, ait)`, here we sum all the reward and we get the score of the given trajectory.
What we are doing here is summing the gradients of all the steps of the trajectory and then multiplying by the `total reward of the trajectory.`
What does this mean? Well in supervised learning (when you compute the gradient of the log loss), that gradient tells you where to move towards your truth label, that's because the log loss is computed for the truth labels. Here the policy `pi` is not the truth label, so our `pi` gradient **does not tell us a priori that the gradient direction is the improvement direction**. To know if it is the correct direction we need to multiply by the total trajectory reward: that's how we know the improvement direction.

### Formula (6) : reward-to-go

Looks like now we are moving the sum of rewards inside the sum of gradients. The idea here is that I am *multiplying step by step the current gradient by the reward to go*.
This reduces the variance, you can imagine you are assigning your 'truth labels' in a less generic and more accurate way compared to before.

### Formula (7) and (8): discount factor

We multiply the reward at every step `t1` by a discount factor `gamma ** (t1 - t)`. This reduces the importance to future rewards since they are less influenced by the gradient at the current steps. This explains (8) whereas is not clear how this is useful in applying the discount factor to the whole trajectory like in (7).

### Formula (9): baseline subtraction

We can subtract a baseline constant with respect to `tau` (so it is constant in all the trajectory `tau`, but is not necesseraly constant in  time `t` of the simulation). 
This also reduced variance as explained in lecture. Why? This is a interesting point. Here you are multiplying by the gradient, but what happens if you reward is always positive? The model will think that what he is doing is always correct, in the sense that step by step the gradient will be multiplied by a positive return, thus it will indicate that the gradient direction is the improvement direction.
In this case we are actually giving the model a *wrong signal*, and a good work around can be subtracting a baseline. The **basline is choosen to be the average of the rewards for the trajectory**, so that your single rewards will be correctly calibrated in being positive or negative (since they we subtract the average).

### Formula (10): value function as a baseline

In the homework we use a state-dependent baseline: it's a value function as it is a function of a state (it expresses the value of a particular state). It is definied as the the sum of future rewards starting from a particular state, given the current policy. This means in practice that you run the policy and you check what is the rewards from there?

An interesting point here is that actually we don't know if this baseline is unbiased with respect to the gradient: in lecture was proven that a baseline date that is not trajectory dependent is unbiased. The homework will go thourgh this later.

### Formula (11): final policy gradient expression

This just puts everything together: we get the baseline of (6) of the sums of gradients multiplied step by step by the reward to go. There is the (8) discount factor and the (9) baseline subtraction.

Supervised Learning of Behaviours
====

What is a *supervised learning model?*

```
A supervised learning model is a parametrization of a probability distribution for an action A given an observation O.
```

Neural nets are powerful function approximator to represent this probability distribution, what is important to understand is that what the **Neural Net is doing is only parametrizing the probability distribution.** We use the notation PI_theta(action| observation), where theta is the parametrization of the probability distribution PI.


## Sequential Learning

To solve the sequential decision making problem we need to take into account time, but we use the same formalism: PI_theta(action_t | observation_t)

We also need to introduce the **concept of state**. State is the underlying reality that leads to the observation: state -> observation.
State follows the **Markov property**, where S_t fully characterize S_t+1, indipendetly of S_t-1.

## Imitation learning

Does this supervised approach works? Most of the time it doesn't. Mistake made by a supervised learned agent become cumulatively bigger..
How can we solve this?
1. **Heuristic apporach**, introduce noise, correct the action using some domain specific knowledge. Not interesting apporach
2. **Algorithmic apporach**, how can we have the Pdata(observatino) = PI_theta(observation), i.e. how can we have the labels distribution be the same as the agent behaviour generated observation? this is more interesting. We will explore this.


How to Pdata(observatino) = PI_theta(observation)?

idea: instead of trying to work on the PI_theta of the agent we can work on the Pdata of the training data.
**DAgger: Dataset Aggregation** -> add onpolicy data
goal: collect tratining data from Ptheta instead of Pdata
how? we just run Ptheta and we make them labelled by human
interesting dataset with observation from policy, action from human. We aggregate dataset and we repeat.
(Ross et al 11)

Bottlneck is human data collection. Can we make it work without more data?
Build a better model, but we need to fit the expert.

Why might we fail to fit the expert? 
1. **Non-markovian behaviour** -> your human is behaving based on the past. 
You can train the Neural Network based on the past -> RNN or LSTM
2. **Multimodial behaviour**  -> different solutions to same problem.
Mixture of Gaussian models -> output N means, N variances, and a scalar weight on each of those means and variances. (mixure density network). Doesn't work well with high dimensional actions.
Latent variable models -> the output is still unimodal gaussian. We introduce a randomness in the input, and then we train the network on change the output distribution based on that noise. (Conditional Variational Autoencoders)
Autoregressive discretization -> with discrete actions there is no problem, softmax with continuous distribution.You discretize the continuos space.


## PAPERS: 

1. A Machine Learning approach to visual perception of forest trails for mobile robots
Goal: fly drone in the forest
Training data: they have a guy with three cameras pointing front left and right, and they assign labels as go straight, turn right, turn left.

2. Learning manipulation tasks from visual perceptions with LSTM
Goal: robot manipulation task
They use LSTM for non-markovianity of the process -> LSTM output for gaussians
Gaussian Mixture model for multimodality


## What is the problem with Imitation learning?
- Data provided by humans, limited. -> Deep Learning neeeds a lot of data.
- Humans not good at providing some demonstrations
- Human can learn autonomusly, can machines do the same?

## Analysis of cost function on imitation learning

We assign a cost function where c(s, a) is 0 if action is in the expert policy, is 1 otherwise

Discrete case -> C = Expectation of all cost for every state action t  ~ (epsilon * T^2) 
ContinuousCase using Dagger -> C = Expectation of all cost for every state action t  ~ (epsilon * T)


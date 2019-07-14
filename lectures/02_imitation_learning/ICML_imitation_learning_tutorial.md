[VIDEO](https://www.youtube.com/watch?v=WjFdD7PDGw0)

# IMITATION LEARNING:
Given: demonstrations or demonstrator
goal: train policy to imitate demonstr

Components:
- Demonstrations
- Env
- Policy
- Loss Function
- Learning Algorithm

Rollout: sequentially execute policy on initial state

Use-case: imitation learning black box policy -> use for model compression

## Behavioural Cloning

Reduction to Supervised Learning
Assumptions: 
1. you want to learn perfectly
2. minimize 1-step deviation error along the expert trajectories

problem: reach new states never trained on -> behaviour undefnied

## Direct Policy Learning

Query demonstrator for new data: reduction problem. Can what I learnt be reduced to something else?
http://hunch.net/~jl/projects/reductions/reductions.html

Sequential Learning Reductions
Data Aggregation (DAgger)
-> online learning
-> follow-the-leader
Policy Aggregation (SEARN & SMILe)

## Inverse RL

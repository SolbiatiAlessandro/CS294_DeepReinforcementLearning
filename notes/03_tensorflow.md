(Greg GSI)

GOAL OF THE COURSE: train an agent to perform useful tasks

DATA -> (this lecture) train model -> AGENT -> get more data

GOAL OF THIS COURSE: how to train a model?

Theta = arg min(thet) Sum( |f-theta(x) - y| )

TENSORFLOW
====
let you define a computation graph
compute gradient

1. define the computatino graph

say you want to define 2 layers NN

you define:
x
h1 = sigma(W1x)
h2 = sigma(W2h1)
y = sigma(W3h2)

TF calculates gradient discent for you

Notebook
====
0. what is TF
1. how to input data
2. how do you perform computation
3. how to create variables
4. how to train nn for regression
5. tips and tricks




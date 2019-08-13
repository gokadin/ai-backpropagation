# Backpropagation algorithm

Backpropagation is a technique used to teach a neural network that has at least one hidden layer. 

## This is part 2 of a series of github repos on neural networks

- [part 1 - linear associative network](https://github.com/gokadin/ai-linear-associative-network)
- part 2 - backpropagation (**you are here**)

## Table of Contents

- [Theory](#theory)  
  - [Introducing the perceptron](#introducing-the-perceptron)
  - [Backpropagation](#backpropagation)
    - [Notation](#notation)
    - [The forward pass](#the-forward-pass)
    - [The backward pass](#the-backward-pass)
  - [Algorithm summary](#algorithm-summary)
- [Code example](#code-example)
- [References](#references)

## Theory

### Introducing the perceptron

A perceptron is the same as our artificial neuron from part 1 of this series, expect that it has an activation threshold. 

![perceptron](readme-images/perceptron.jpg)

#### Activation functions

If $u = \sum{\vec{x}\vec{w}}$ then typical activation functions are:

- Sigmoid

$$ y = \frac{1}{1 + e^{-u}} $$

- ReLU or rectified linear unit

$$ y = max(0, u) $$

- tanh

$$ y = tanh(u) $$

### Backpropagation

The backpropagation algorithm is used to train artificial neural networks, more specifically those with more than two layers. 

It's using a forward pass to compute the outputs of the network, calculates the error and then goes backwards towards the input layer to update each weight based on the error gradient. 

#### Notation

![notation](readme-images/backpropagation-notation.jpg)

$t$ being the current association out of $T$ associations. 

We will assign the following activation functions to each layer:

- input layer -> identity function
- hidden layer -> sigmoid function
- output layer -> identity function

#### The forward pass

During the forward pass, we feed the inputs to the input layer and get the results in the output layer. 

The input to each perceptron in the hidden layer $u_{jt}$ is the sum of all perceptron of the previous layer times their corresponding weight:

$$u_{jt} = \sum_{i = 1}^{I} w_{ij}x_{it}$$

However, since our hidden layer's activation function for each perceptron is the sigmoid, then their output will be: 

$$ z_{jt} = f_j(u_{jt}) = (1 + e^{-u_{jt}})^{-1} $$

In the same manner, the input to the output layer perceptrons are

$$ u_{kt} = \sum^{J}_{j = 1} w_{jk}z_{jt} $$

and their output is

$$ y_{kt} = f_k(u_{kt}) = u_{kt} $$

since the activation function is the identity function in this case. 

Once the inputs have been propagated through the network, we can calculate the error:

$$ E = \sum^{T}_{t = 1} E_t = \frac{1}{2} \sum^{T}_{t = 1} (y_{kt} - y\prime_{kt})^2 $$

#### The backward pass

...

### Algorithm summary

...

## Code example

under construction...

## References

- Artificial intelligence engines by James V Stone (2019)
- http://neuralnetworksanddeeplearning.com/chap2.html
# Backpropagation algorithm

Backpropagation is a technique used to teach a neural network that has at least one hidden layer. 

## This is part 2 of a series of github repos on neural networks

- [part 1 - linear associative network](https://github.com/gokadin/ai-linear-associative-network)
- part 2 - backpropagation (**you are here**)

## Table of Contents

- [Theory](#theory)  
  - [Introducing the perceptron](#introducing-the-perceptron)
- [Code example](#code-example)
- [References](#references)

## Theory

### Introducing the perceptron

A perceptron is the same as our artificial neuron from part 1 of this series, expect that it has an activation threshold. 

![perceptron](readme-images/perceptron.jpg)

#### Activation functions

If $u = \sum{\vec{x}\vec{w}}$ then typical activation functions are:

- Sigmoid

$$ y = (1 + e^{-u})^{-1} $$

- ReLU or rectified linear unit

$$ y = max(0, u) $$

- tanh

$$ y = tanh(u) $$

## Code example

under construction...

## References

- Artificial intelligence engines by James V Stone (2019)
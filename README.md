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

If <img src="/tex/22a1bca1370e6ec943222dc2ea608065.svg?invert_in_darkmode&sanitize=true" align=middle width=73.66339529999999pt height=24.657735299999988pt/> then typical activation functions are:

- Sigmoid

<p align="center"><img src="/tex/af4b64501b7d73bd2f2f5c56f5cac500.svg?invert_in_darkmode&sanitize=true" align=middle width=87.37202265pt height=34.3600389pt/></p>

- ReLU or rectified linear unit

<p align="center"><img src="/tex/26fea42205a1ecb142c6ba3cb3ba73bf.svg?invert_in_darkmode&sanitize=true" align=middle width=100.80487889999999pt height=16.438356pt/></p>

- tanh

<p align="center"><img src="/tex/4660ad0287e8a2fe5ffefc23983877df.svg?invert_in_darkmode&sanitize=true" align=middle width=86.7257853pt height=16.438356pt/></p>

### Backpropagation

The backpropagation algorithm is used to train artificial neural networks, more specifically those with more than two layers. 

It's using a forward pass to compute the outputs of the network, calculates the error and then goes backwards towards the input layer to update each weight based on the error gradient. 

#### Notation

![notation](readme-images/backpropagation-notation.jpg)

<img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> being the current association out of <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/> associations. 

We will assign the following activation functions to each layer:

- input layer -> identity function
- hidden layer -> sigmoid function
- output layer -> identity function

#### The forward pass

During the forward pass, we feed the inputs to the input layer and get the results in the output layer. 

The input to each perceptron in the hidden layer <img src="/tex/f8bbbfffa921d3289fa9fdb9a1cf47c4.svg?invert_in_darkmode&sanitize=true" align=middle width=20.48055239999999pt height=14.15524440000002pt/> is the sum of all perceptron of the previous layer times their corresponding weight:

<p align="center"><img src="/tex/634b2baa2264a3d1e63748c5bbe16883.svg?invert_in_darkmode&sanitize=true" align=middle width=112.0615452pt height=47.806078649999996pt/></p>

However, since our hidden layer's activation function for each perceptron is the sigmoid, then their output will be: 

<p align="center"><img src="/tex/5600978e3afab43af2339742b528e132.svg?invert_in_darkmode&sanitize=true" align=middle width=207.6529389pt height=18.905967299999997pt/></p>

In the same manner, the input to the output layer perceptrons are

<p align="center"><img src="/tex/089ae982e01cf7066a17c61d442e4e09.svg?invert_in_darkmode&sanitize=true" align=middle width=115.5414876pt height=50.04352485pt/></p>

and their output is the same since we assigned them the identity activation function. 

<p align="center"><img src="/tex/ed55f30fd57b7f81e3eb501fd93a0426.svg?invert_in_darkmode&sanitize=true" align=middle width=137.97585285pt height=16.438356pt/></p>

Once the inputs have been propagated through the network, we can calculate the error:

<p align="center"><img src="/tex/6057c698055c0474a4a50b2ad52f8b05.svg?invert_in_darkmode&sanitize=true" align=middle width=226.73802029999996pt height=47.60747145pt/></p>

#### The backward pass

...

### Algorithm summary

...

## Code example

under construction...

## References

- Artificial intelligence engines by James V Stone (2019)
- http://neuralnetworksanddeeplearning.com/chap2.html
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

We will assign the following activation functions to each layer perceptrons:

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

Now that we have the error, we can use it to update each weight of the network by going backwards layer by layer. 

We know from *part 1* that the change of a weight is the negative of that weight's component in the error gradient times the learning rate. For a weight between the last hidden layer and the output layer, we then have

<p align="center"><img src="/tex/d487ff384e18d4e3e131374a1b020be2.svg?invert_in_darkmode&sanitize=true" align=middle width=123.57475514999999pt height=38.5152603pt/></p>

We can find the error gradient by using the chain rule

<p align="center"><img src="/tex/bd12a05a54dd4b0c9a28f8118bf5accd.svg?invert_in_darkmode&sanitize=true" align=middle width=196.07986365pt height=38.5152603pt/></p> where <p align="center"><img src="/tex/f81547709974d10052c82a9e92bcbfbf.svg?invert_in_darkmode&sanitize=true" align=middle width=108.8831898pt height=14.611878599999999pt/></p>

Similarily, for a weight between hidden layers, in our case between the input layer and our first hidden layer, we have

<p align="center"><img src="/tex/6a7b90b9efb24cc2b6ecdfdadd20791b.svg?invert_in_darkmode&sanitize=true" align=middle width=118.34445975pt height=38.5152603pt/></p>

<p align="center"><img src="/tex/b5da48296be00475509f0eb26498996b.svg?invert_in_darkmode&sanitize=true" align=middle width=188.82323294999998pt height=38.5152603pt/></p> where <p align="center"><img src="/tex/681b2b539d0e01e943930084bd33ce48.svg?invert_in_darkmode&sanitize=true" align=middle width=196.17316455pt height=48.18280005pt/></p>

Here the calculations are *slightly* more complex. Let's analyse the delta term <img src="/tex/a3ec72e0f05115605b57d81cfab96e7d.svg?invert_in_darkmode&sanitize=true" align=middle width=18.37621829999999pt height=22.831056599999986pt/> and understand how we got there. We start by calculating the partial derivative of <img src="/tex/f8bbbfffa921d3289fa9fdb9a1cf47c4.svg?invert_in_darkmode&sanitize=true" align=middle width=20.48055239999999pt height=14.15524440000002pt/> in respect to the error by using the chain rule

<p align="center"><img src="/tex/e4d98a206a2733836d784f9065398280.svg?invert_in_darkmode&sanitize=true" align=middle width=119.78647559999999pt height=38.5152603pt/></p>

<p align="center"><img src="/tex/95f713b775424bdeda4bd528c112c12e.svg?invert_in_darkmode&sanitize=true" align=middle width=245.88211725pt height=48.18280005pt/></p> and <p align="center"><img src="/tex/901a8e88ab6651110940411393ccdc74.svg?invert_in_darkmode&sanitize=true" align=middle width=205.06454265pt height=38.5152603pt/></p>

Remember that our activation function <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is the sigmoid function and that its derivative is <img src="/tex/63905ec601ca88b13ff9a43d55aee30f.svg?invert_in_darkmode&sanitize=true" align=middle width=105.09150299999999pt height=24.65753399999998pt/>

The change of a weight for <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/> associations is the accumulation of each association

<p align="center"><img src="/tex/f8986238597442b770c458c54abaccdc.svg?invert_in_darkmode&sanitize=true" align=middle width=145.85336865pt height=47.60747145pt/></p>

### Algorithm summary

...

## Code example

under construction...

## References

- Artificial intelligence engines by James V Stone (2019)
- http://neuralnetworksanddeeplearning.com/chap2.html
- https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/
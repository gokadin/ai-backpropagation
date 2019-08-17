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
  - [Visualizing backpropagation](#visualizing-backpropagation)
    - [Forward pass](#forward-pass)
    - [Backward pass](#backward-pass)
- [Code example](#code-example)
- [References](#references)

## Theory

### Introducing the perceptron

A perceptron is the same as our artificial neuron from part 1 of this series, expect that it has an activation function $f$ that determines its output $y$. 

![perceptron](readme-images/perceptron.jpg)

#### Some activation functions

If $u = \sum{\vec{x}\vec{w}}$ then typical activation functions are:

- Sigmoid $\quad y = \frac{1}{1 + e^{-u}}$

- ReLU or rectified linear unit $\quad y = max(0, u)$

- tanh $\quad y = tanh(u)$

### Backpropagation

The backpropagation algorithm is used to train artificial neural networks, more specifically those with more than two layers. 

It's using a forward pass to compute the outputs of the network, calculates the error and then goes backwards towards the input layer to update each weight based on the error gradient. 

#### Notation

![notation](readme-images/backpropagation-notation.jpg)

$t$ being the current association out of $T$ associations. 

We will assign the following activation functions to each layer perceptrons:

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

and their output is the same since we assigned them the identity activation function. 

$$ y_{kt} = f_k(u_{kt}) = u_{kt} $$

Once the inputs have been propagated through the network, we can calculate the error:

$$ E = \sum^{T}_{t = 1} E_t = \frac{1}{2} \sum^{T}_{t = 1} (y_{kt} - y\prime_{kt})^2 $$

#### The backward pass

Now that we have the error, we can use it to update each weight of the network by going backwards layer by layer. 

We know from *part 1* that the change of a weight is the negative of that weight's component in the error gradient times the learning rate. For a weight between the last hidden layer and the output layer, we then have

$$ \Delta w_{jkt} = -\epsilon \frac{\partial E_t}{\partial w_{jk}} $$

We can find the error gradient by using the chain rule

$$ \frac{\partial E_t}{\partial w_{jk}} = \frac{\partial E_t}{\partial u_{kt}} \frac{\partial u_{kt}}{\partial w_{jk}} = \delta_{kt} z_{jt} \quad where \quad \delta_{kt} = y_{kt} - y\prime_{kt} $$

Similarly, for a weight between hidden layers, in our case between the input layer and our first hidden layer, we have

$$ \Delta w_{ijt} = -\epsilon \frac{\partial E_t}{\partial w_{ij}} $$

$$ \frac{\partial E_t}{\partial w_{ij}} = \frac{\partial E_t}{\partial u_{jt}} \frac{\partial u_{jt}}{\partial w_{ij}} = \delta_{jt} x_{it} \quad where \quad \delta_{jt} = z_{jt} (1 - z_{jt}) \sum^K_{k = 1} \delta_{kt} w_{jk} $$

Here the calculations are *slightly* more complex. Let's analyze the delta term $\delta_{jt}$ and understand how we got there. We start by calculating the partial derivative of $u_{jt}$ in respect to the error by using the chain rule

$$ \frac{\partial E_t}{\partial u_{jt}} = \frac{\partial E_t}{\partial z_{jt}} \frac{d z_{jt}}{du_{jt}} $$

$$ \frac{\partial E_t}{\partial z_{jt}} = \sum^K_{k = 1} \frac{\partial E_t}{\partial u_{kt}} \frac{\partial u_{kt}}{\partial z_{jt}} = \sum^K_{k = 1} \delta_{kt} w_{jk} \quad and \quad \frac{d z_{jt}}{du_{jt}} = f'(z_{jt}) = z_{jt}(1 - z{jt}) $$

Remember that our activation function $f$ is the sigmoid function and that its derivative is $f(x)(1 - f(x))$

The change of a weight for $T$ associations is the accumulation of each association

$$ \Delta w_{ij} = -\epsilon \sum^T_{t = 1} \delta_{jt} x_{it} $$

### Algorithm summary

First, initialize network weights to a small random value. 

Repeat the steps below until the error is about $0$

- for each association, propagate the network forward and get the outputs
  - calculate the $\delta$ term for each output layer node ($y_{kt} - y\prime_{kt}$)
  - accumulate the gradient for each output weight ($\delta_{kt} z_{jt}$)
  - calculate the $\delta$ term for each hidden layer node ($z_{jt}(1 - z_{jt})\sum^K_{k = 1}\delta_{kt} w_{jt}$)
  - accumulate the gradient for each hidden layer weight ($\delta_{jt} x_{it}$)
- update all weights and reset accumulated gradients ($w_{ij} = w_{ij} - \epsilon \sum^T_{t = 1}\delta_{jt} x_{it}$)

### Visualizing backpropagation

In this example, we'll use actual numbers to follow each step of the network. We'll feed our 2x2x1 network with inputs $[1.0, 1.0]$ and we will expect an output of $[0.5]$. To make matters simpler, we'll initialize all of our weights with the same value of $0.5$. However, keep in mind that normally weights are initialized using random numbers. We will also design the network with a sigmoid activation function for the hidden layer and the identity function for the input and output layers. 

#### Forward pass

We start by setting all of the nodes of the input layer with the input values; $x_1 = 1.0, x_2 = 1.0$. 

Since the input layer nodes have no activation function, then $y_i = x_i = 1.0$. 

![backpropagation-visual](readme-images/backprop-visual-1.jpg)

We then propagate the network forward by setting the $J$ layer node inputs ($x_j$) with the sum of all of the previous layer node outputs times their corresponding weights:

$$ x_j = \sum^I_{i = 1} x_i w_{ij} = 1.0 * 0.5 + 1.0 * 0.5 = 1.0 $$

![backpropagation-visual](readme-images/backprop-visual-2.jpg)

![backpropagation-visual](readme-images/backprop-visual-3.jpg)

We then activate the $J$ layer nodes by passing it's inputs to the sigmoid function $f(x) = \frac{1}{1 + e^{-x}}$

![backpropagation-visual](readme-images/backprop-visual-4.jpg)

And we propagate those results to the final layer $x_k = 0.731 * 0.5 + 0.731 * 0.5 = 0.731$

Since we didn't assign an activation function to our output layer node, then $y_k = x_k = 0.731$

![backpropagation-visual](readme-images/backprop-visual-5.jpg)

#### Backward pass

...

## Code example

The example teaches a 2x2x1 network the XOR operator. 

$$ x = \begin{bmatrix}
    1.0 & 1.0 & 0.0 & 0.0 \\
    1.0 & 0.0 & 1.0 & 0.0 \\ 
\end{bmatrix} \quad y\prime = \begin{bmatrix}
    0.0 & 1.0 & 1.0 & 0.0 \\
\end{bmatrix} $$

![code-example](readme-images/backpropagation-code-example.jpg)

Where $f$ is the sigmoid function for the hidden layer nodes. 

## References

- Artificial intelligence engines by James V Stone (2019)
- http://neuralnetworksanddeeplearning.com/chap2.html
- https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/
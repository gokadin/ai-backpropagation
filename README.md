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

<p align="center"><img src="/tex/b96abf43fa8539cc9ea550f3860b16fa.svg?invert_in_darkmode&sanitize=true" align=middle width=383.52490109999997pt height=38.5152603pt/></p>

Similarly, for a weight between hidden layers, in our case between the input layer and our first hidden layer, we have

<p align="center"><img src="/tex/6a7b90b9efb24cc2b6ecdfdadd20791b.svg?invert_in_darkmode&sanitize=true" align=middle width=118.34445975pt height=38.5152603pt/></p>

<p align="center"><img src="/tex/26d8e54af9b9c5f2195681156b3cdddd.svg?invert_in_darkmode&sanitize=true" align=middle width=463.55822865pt height=48.18280005pt/></p>

Here the calculations are *slightly* more complex. Let's analyze the delta term <img src="/tex/a3ec72e0f05115605b57d81cfab96e7d.svg?invert_in_darkmode&sanitize=true" align=middle width=18.37621829999999pt height=22.831056599999986pt/> and understand how we got there. We start by calculating the partial derivative of <img src="/tex/f8bbbfffa921d3289fa9fdb9a1cf47c4.svg?invert_in_darkmode&sanitize=true" align=middle width=20.48055239999999pt height=14.15524440000002pt/> in respect to the error by using the chain rule

<p align="center"><img src="/tex/e4d98a206a2733836d784f9065398280.svg?invert_in_darkmode&sanitize=true" align=middle width=119.78647559999999pt height=38.5152603pt/></p>

<p align="center"><img src="/tex/18091e78f58dcc23ad486df3e2b7b347.svg?invert_in_darkmode&sanitize=true" align=middle width=513.72992715pt height=48.18280005pt/></p>

Remember that our activation function <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is the sigmoid function and that its derivative is <img src="/tex/63905ec601ca88b13ff9a43d55aee30f.svg?invert_in_darkmode&sanitize=true" align=middle width=105.09150299999999pt height=24.65753399999998pt/>

The change of a weight for <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/> associations is the accumulation of each association

<p align="center"><img src="/tex/f8986238597442b770c458c54abaccdc.svg?invert_in_darkmode&sanitize=true" align=middle width=145.85336865pt height=47.60747145pt/></p>

### Algorithm summary

- initialize network weights to a small random value
- while error gradient is not ~<img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/> 
  - for each association, propagate the network forward and get the outputs
    - accumulate the $\delta$ term for each output layer node ($y_{kt} - y\prime_{kt}$)  
    - accumulate the gradient for each output weight ($\delta_{kt} z_{jt}$)  
    - accumulate the $\delta$ term for each hidden layer node ($z_{jt}(1 - z_{jt})\sum^K_{k = 1}\delta_{kt} w_{jt}$)  
    - accumulate the gradient for each hidden layer weight ($\delta_{jt} x_{it}$) 
  - update all weights and reset accumulated gradient and delta values (<img src="/tex/ee701a4682b8b020790a6e5d3183bd37.svg?invert_in_darkmode&sanitize=true" align=middle width=178.84515495pt height=32.256008400000006pt/>)

### Visualizing backpropagation

In this example, we'll use real numbers to follow each step of the network. We'll feed our 2x2x1 network with inputs <img src="/tex/e4f0b9bce59fcd6b7ace485069c84ced.svg?invert_in_darkmode&sanitize=true" align=middle width=58.44761669999998pt height=24.65753399999998pt/> and we will expect an output of <img src="/tex/51d89c4114d0201a214771c31c6bff9f.svg?invert_in_darkmode&sanitize=true" align=middle width=30.137091599999987pt height=24.65753399999998pt/>. To make matters simpler, we'll initialize all of our weights with the same value of <img src="/tex/cde2d598001a947a6afd044a43d15629.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=21.18721440000001pt/>. However, keep in mind that normally weights are initialized using random numbers. We will also design the network with a sigmoid activation function for the hidden layer and the identity function for the input and output layers. 

#### The forward pass

We start by setting all of the nodes of the input layer with the input values; <img src="/tex/902d9d457b5dd887bab6109f2939c439.svg?invert_in_darkmode&sanitize=true" align=middle width=126.68932649999998pt height=21.18721440000001pt/>. 

Since the input layer nodes have no activation function, then <img src="/tex/0a353c7c5dda5a3590dfdf659d666296.svg?invert_in_darkmode&sanitize=true" align=middle width=93.23991599999998pt height=21.18721440000001pt/>. 

![backpropagation-visual](readme-images/backprop-visual-1.jpg)

We then propagate the network forward by setting the <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> layer node inputs (<img src="/tex/4d8443b72a1de913b4a3995119296c90.svg?invert_in_darkmode&sanitize=true" align=middle width=15.499497749999989pt height=14.15524440000002pt/>) with the sum of all of the previous layer node outputs times their corresponding weights:

<p align="center"><img src="/tex/b8b64f33f599e5f31d0d1ee4aa8d7fd2.svg?invert_in_darkmode&sanitize=true" align=middle width=302.93619674999997pt height=47.806078649999996pt/></p>

![backpropagation-visual](readme-images/backprop-visual-2.jpg)

![backpropagation-visual](readme-images/backprop-visual-3.jpg)

We then activate the <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> layer nodes by passing it's inputs to the sigmoid function <img src="/tex/2bdf51d6b6da5953c65a2097ac846972.svg?invert_in_darkmode&sanitize=true" align=middle width=95.00189984999999pt height=27.77565449999998pt/>

![backpropagation-visual](readme-images/backprop-visual-4.jpg)

And we propagate those results to the final layer <img src="/tex/268ed1352dff47b71e61b3328c3edce3.svg?invert_in_darkmode&sanitize=true" align=middle width=266.79778949999996pt height=21.18721440000001pt/>

Since we didn't assign an activation function to our output layer node, then <img src="/tex/e1f7e2cd426e2ece1aad8a07e24f10d7.svg?invert_in_darkmode&sanitize=true" align=middle width=114.9086301pt height=21.18721440000001pt/>

![backpropagation-visual](readme-images/backprop-visual-5.jpg)

#### The backward pass

...

## Code example

The example teaches a 2x2x1 network the XOR operator. 

<p align="center"><img src="/tex/45e155848e9c53acf0d057548b843990.svg?invert_in_darkmode&sanitize=true" align=middle width=383.295594pt height=39.452455349999994pt/></p>

![code-example](readme-images/backpropagation-code-example.jpg)

Where <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is the sigmoid function for the hidden layer nodes. 

## References

- Artificial intelligence engines by James V Stone (2019)
- http://neuralnetworksanddeeplearning.com/chap2.html
- https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/
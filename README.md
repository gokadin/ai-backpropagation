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

A perceptron is a processing unit that takes an input <img src="/tex/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=14.15524440000002pt/>, transforms it using an activation function <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> and outputs the result <img src="/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>. 

Just as we saw in *part 1*, its input is the sum of the previous layer node outputs times their corresponding weight, plus the previous layer bias unit times its weight:

<p align="center"><img src="/tex/364a0eb6624fe587c128c4de95f94f58.svg?invert_in_darkmode&sanitize=true" align=middle width=158.0792961pt height=47.806078649999996pt/></p>

If we treat the bias as an additional node in a layer with a constant value of <img src="/tex/e11a8cfcf953c683196d7a48677b2277.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=21.18721440000001pt/>, then we can simplify the equation:

<p align="center"><img src="/tex/7c6f451f5ce1a1ee8880a8a4c1b64be0.svg?invert_in_darkmode&sanitize=true" align=middle width=102.11472149999999pt height=47.806078649999996pt/></p>

![perceptron](readme-images/perceptron.jpg)

#### Some activation functions

If <img src="/tex/22a1bca1370e6ec943222dc2ea608065.svg?invert_in_darkmode&sanitize=true" align=middle width=73.66339529999999pt height=24.657735299999988pt/> then typical activation functions are:

- Sigmoid <img src="/tex/d86b02d0ff42235bec29783460db8196.svg?invert_in_darkmode&sanitize=true" align=middle width=72.10045094999998pt height=27.77565449999998pt/>

- ReLU or rectified linear unit <img src="/tex/e1ffed025adfa4b6f4c2d0b31b8a7148.svg?invert_in_darkmode&sanitize=true" align=middle width=100.80487889999998pt height=24.65753399999998pt/>

- tanh <img src="/tex/4ad8e0d09581519fd366703474f136cb.svg?invert_in_darkmode&sanitize=true" align=middle width=86.72578694999999pt height=24.65753399999998pt/>

### Backpropagation

The backpropagation algorithm is used to train artificial neural networks, more specifically those with more than two layers. 

It's using a forward pass to compute the outputs of the network, calculates the error and then goes backwards towards the input layer to update each weight based on the error gradient. 

#### Notation

- <img src="/tex/beb7fb5a05eb06456ac32c429a488e7c.svg?invert_in_darkmode&sanitize=true" align=middle width=62.46195944999999pt height=14.15524440000002pt/>, are inputs to a node for layers <img src="/tex/b8e8ee904bff633d4f9b547d63ab02c3.svg?invert_in_darkmode&sanitize=true" align=middle width=47.134574849999986pt height=22.465723500000017pt/> respectively. 
- <img src="/tex/deafcef8509ceda8122d0f571d95d999.svg?invert_in_darkmode&sanitize=true" align=middle width=58.45528919999998pt height=14.15524440000002pt/>, are the outputs from a node for layers <img src="/tex/b8e8ee904bff633d4f9b547d63ab02c3.svg?invert_in_darkmode&sanitize=true" align=middle width=47.134574849999986pt height=22.465723500000017pt/> respectively. 
- <img src="/tex/2044a83864c2e5cb971dfb7289018a53.svg?invert_in_darkmode&sanitize=true" align=middle width=20.43577799999999pt height=18.264896099999987pt/> is the expected output of a node of the <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> output layer. 
- <img src="/tex/22ed3508464d792f01cafea152faf749.svg?invert_in_darkmode&sanitize=true" align=middle width=55.79069924999999pt height=14.15524440000002pt/> are weights of node connections from layer <img src="/tex/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode&sanitize=true" align=middle width=8.515988249999989pt height=22.465723500000017pt/> to <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> and from layer <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> to <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> respectively.
- <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> is the current association out of <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/> associations. 

We will assign the following activation functions to each layer perceptrons for all following examples:

- input layer -> identity function
- hidden layer -> sigmoid function
- output layer -> identity function

#### The forward pass

During the forward pass, we feed the inputs to the input layer and get the results in the output layer. 

The input to each perceptron in the hidden layer <img src="/tex/37b7efd01cadaf004426bbffacbe789e.svg?invert_in_darkmode&sanitize=true" align=middle width=20.465266799999988pt height=14.15524440000002pt/> is the sum of all perceptron of the previous layer times their corresponding weight:

<p align="center"><img src="/tex/5d70e98d4c0a50ab91845bcaf8f1c148.svg?invert_in_darkmode&sanitize=true" align=middle width=112.04626124999999pt height=47.806078649999996pt/></p>

However, since our hidden layer's activation function for each perceptron is the sigmoid, then their output will be: 

<p align="center"><img src="/tex/c8e35d5cf0e85503ce933db89a6ceb9c.svg?invert_in_darkmode&sanitize=true" align=middle width=207.7346304pt height=18.905967299999997pt/></p>

In the same manner, the input to the output layer perceptrons are

<p align="center"><img src="/tex/1ea3438c06b1a112d773b02b7b4fd402.svg?invert_in_darkmode&sanitize=true" align=middle width=115.94097899999998pt height=50.04352485pt/></p>

and their output is the same since we assigned them the identity activation function. 

<p align="center"><img src="/tex/09afeca56ff80b12106c808e9ac097fb.svg?invert_in_darkmode&sanitize=true" align=middle width=137.9452833pt height=16.438356pt/></p>

Once the inputs have been propagated through the network, we can calculate the error:

<p align="center"><img src="/tex/6057c698055c0474a4a50b2ad52f8b05.svg?invert_in_darkmode&sanitize=true" align=middle width=226.73802029999996pt height=47.60747145pt/></p>

#### The backward pass

Now that we have the error, we can use it to update each weight of the network by going backwards layer by layer. 

We know from *part 1* that the change of a weight is the negative of that weight's component in the error gradient times the learning rate. For a weight between the last hidden layer and the output layer, we then have

<p align="center"><img src="/tex/d487ff384e18d4e3e131374a1b020be2.svg?invert_in_darkmode&sanitize=true" align=middle width=123.57475514999999pt height=38.5152603pt/></p>

We can find the error gradient by using the chain rule

<p align="center"><img src="/tex/4cd82dbea08570fbcff0df63b4ed570a.svg?invert_in_darkmode&sanitize=true" align=middle width=383.9243793pt height=38.5152603pt/></p>

Similarly, for a weight between hidden layers, in our case between the input layer and our first hidden layer, we have

<p align="center"><img src="/tex/6a7b90b9efb24cc2b6ecdfdadd20791b.svg?invert_in_darkmode&sanitize=true" align=middle width=118.34445975pt height=38.5152603pt/></p>

<p align="center"><img src="/tex/a744eadb2e66bcbb33e45eedf1240f2a.svg?invert_in_darkmode&sanitize=true" align=middle width=464.3724821999999pt height=48.18280005pt/></p>

Here the calculations are *slightly* more complex. Let's analyze the delta term <img src="/tex/a3ec72e0f05115605b57d81cfab96e7d.svg?invert_in_darkmode&sanitize=true" align=middle width=18.37621829999999pt height=22.831056599999986pt/> and understand how we got there. We start by calculating the partial derivative of <img src="/tex/f8bbbfffa921d3289fa9fdb9a1cf47c4.svg?invert_in_darkmode&sanitize=true" align=middle width=20.48055239999999pt height=14.15524440000002pt/> in respect to the error by using the chain rule

<p align="center"><img src="/tex/27d8e79faef062f9f5ee979318bc8d9f.svg?invert_in_darkmode&sanitize=true" align=middle width=120.17064509999999pt height=38.5152603pt/></p>

<p align="center"><img src="/tex/45a199a55d24da3882cf776b427f046a.svg?invert_in_darkmode&sanitize=true" align=middle width=515.20994085pt height=48.18280005pt/></p>

Remember that our activation function <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is the sigmoid function and that its derivative is <img src="/tex/63905ec601ca88b13ff9a43d55aee30f.svg?invert_in_darkmode&sanitize=true" align=middle width=105.09150299999999pt height=24.65753399999998pt/>

The change of a weight for <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/> associations is the accumulation of each association

<p align="center"><img src="/tex/f8986238597442b770c458c54abaccdc.svg?invert_in_darkmode&sanitize=true" align=middle width=145.85336865pt height=47.60747145pt/></p>

### Algorithm summary

First, initialize network weights to a small random value. 

Repeat the steps below until the error is about <img src="/tex/29632a9bf827ce0200454dd32fc3be82.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=21.18721440000001pt/>

- for each association, propagate the network forward and get the outputs
  - calculate the <img src="/tex/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928075099999989pt height=22.831056599999986pt/> term for each output layer node (<img src="/tex/df9fdf72f97ac36f131b95a7e6d8fa93.svg?invert_in_darkmode&sanitize=true" align=middle width=66.60590639999998pt height=19.1781018pt/>)
  - accumulate the gradient for each output weight (<img src="/tex/c0e74ee9a32d2c89a9710dcacfab145b.svg?invert_in_darkmode&sanitize=true" align=middle width=39.48936419999999pt height=22.831056599999986pt/>)
  - calculate the <img src="/tex/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928075099999989pt height=22.831056599999986pt/> term for each hidden layer node (<img src="/tex/e53349390d41cec36f5431c3736f598f.svg?invert_in_darkmode&sanitize=true" align=middle width=171.76030905pt height=32.256008400000006pt/>)
  - accumulate the gradient for each hidden layer weight (<img src="/tex/c6030a58bf6eac6e114111d20953f5e1.svg?invert_in_darkmode&sanitize=true" align=middle width=38.209792499999985pt height=22.831056599999986pt/>)
- update all weights and reset accumulated gradients (<img src="/tex/ee701a4682b8b020790a6e5d3183bd37.svg?invert_in_darkmode&sanitize=true" align=middle width=178.84515495pt height=32.256008400000006pt/>)

### Visualizing backpropagation

In this example, we'll use actual numbers to follow each step of the network. We'll feed our 2x2x1 network with inputs <img src="/tex/e4f0b9bce59fcd6b7ace485069c84ced.svg?invert_in_darkmode&sanitize=true" align=middle width=58.44761669999998pt height=24.65753399999998pt/> and we will expect an output of <img src="/tex/51d89c4114d0201a214771c31c6bff9f.svg?invert_in_darkmode&sanitize=true" align=middle width=30.137091599999987pt height=24.65753399999998pt/>. To make matters simpler, we'll initialize all of our weights with the same value of <img src="/tex/cde2d598001a947a6afd044a43d15629.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=21.18721440000001pt/>. However, keep in mind that normally weights are initialized using random numbers. We will also design the network with a sigmoid activation function for the hidden layer and the identity function for the input and output layers. 

#### Forward pass

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

#### Backward pass

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
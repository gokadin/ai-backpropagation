# Backpropagation algorithm

Backpropagation is a technique used to teach a neural network that has at least one hidden layer. 

## This is part 2 of a series of github repos on neural networks

- [part 1 - simplest network](https://github.com/gokadin/ai-simplest-network)
- part 2 - backpropagation (**you are here**)
- [part 3 - backpropagation-continued](https://github.com/gokadin/ai-backpropagation-continued)
- [part 4 - hopfield networks](https://github.com/gokadin/ai-hopfield-networks)

## Table of Contents

- [Theory](#theory)  
  - [Introducing the perceptron](#introducing-the-perceptron)
    - [Activation functions](#activation-functions)
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

#### Activation functions

Why do we need an activation function? Without it the output of every node will be linear, making the neural network output a linear function of the inputs. Since the combination of two linear functions is also a linear function, you can't compute more interesting functions without non-linear ones. This means that the network will only be able to solve problems that can be solved with linear regression. 

If <img src="/tex/36d6e7b3abf42ac72c9766d04e081cf1.svg?invert_in_darkmode&sanitize=true" align=middle width=73.64811134999998pt height=24.657735299999988pt/> then typical activation functions are:

- Sigmoid <img src="/tex/06a4dabcd23aee6477642df56f5823d0.svg?invert_in_darkmode&sanitize=true" align=middle width=71.65327124999999pt height=27.77565449999998pt/>
- ReLU or rectified linear unit <img src="/tex/2303c524a588b86a618bd6158a13de22.svg?invert_in_darkmode&sanitize=true" align=middle width=100.78959329999998pt height=24.65753399999998pt/>
- tanh <img src="/tex/d780a3472dc758b220b61dab83747006.svg?invert_in_darkmode&sanitize=true" align=middle width=86.71050134999999pt height=24.65753399999998pt/>

### Backpropagation

The backpropagation algorithm is used to train artificial neural networks, more specifically those with more than two layers. 

It's using a forward pass to compute the outputs of the network, calculates the error and then goes backwards towards the input layer to update each weight based on the error gradient. 

#### Notation

- <img src="/tex/beb7fb5a05eb06456ac32c429a488e7c.svg?invert_in_darkmode&sanitize=true" align=middle width=62.46195944999999pt height=14.15524440000002pt/>, are inputs to a node for layers <img src="/tex/b8e8ee904bff633d4f9b547d63ab02c3.svg?invert_in_darkmode&sanitize=true" align=middle width=47.134574849999986pt height=22.465723500000017pt/> respectively. 
- <img src="/tex/deafcef8509ceda8122d0f571d95d999.svg?invert_in_darkmode&sanitize=true" align=middle width=58.45528919999998pt height=14.15524440000002pt/>, are the outputs from a node for layers <img src="/tex/b8e8ee904bff633d4f9b547d63ab02c3.svg?invert_in_darkmode&sanitize=true" align=middle width=47.134574849999986pt height=22.465723500000017pt/> respectively. 
- <img src="/tex/2044a83864c2e5cb971dfb7289018a53.svg?invert_in_darkmode&sanitize=true" align=middle width=20.43577799999999pt height=18.264896099999987pt/> is the expected output of a node of the <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> output layer. 
- <img src="/tex/22ed3508464d792f01cafea152faf749.svg?invert_in_darkmode&sanitize=true" align=middle width=55.79069924999999pt height=14.15524440000002pt/> are weights of node connections from layer <img src="/tex/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode&sanitize=true" align=middle width=8.515988249999989pt height=22.465723500000017pt/> to <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> and from layer <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> to <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> respectively.
- <img src="/tex/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode&sanitize=true" align=middle width=5.936097749999991pt height=20.221802699999984pt/> is the current association out of <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/> associations. 

We will assign the following activation functions to each layer nodes for all following examples:

- input layer -> identity function
- hidden layer -> sigmoid function
- output layer -> identity function

#### The forward pass

During the forward pass, we feed the inputs to the input layer and get the results in the output layer. 

The input to each node in the hidden layer <img src="/tex/37b7efd01cadaf004426bbffacbe789e.svg?invert_in_darkmode&sanitize=true" align=middle width=20.465266799999988pt height=14.15524440000002pt/> is the sum of the output from all nodes of the input layer times their corresponding weight:

<p align="center"><img src="/tex/07be965da2a124c3a996abde43bae6ef.svg?invert_in_darkmode&sanitize=true" align=middle width=110.7107034pt height=47.806078649999996pt/></p>

Since the hidden layer's activation function for each node is the sigmoid, then their output will be: 

<p align="center"><img src="/tex/2f47d825a24e00faa091cbb66d6ffd6f.svg?invert_in_darkmode&sanitize=true" align=middle width=180.09521145pt height=34.3600389pt/></p>

In the same manner, the input to the output layer nodes are

<p align="center"><img src="/tex/1ea3438c06b1a112d773b02b7b4fd402.svg?invert_in_darkmode&sanitize=true" align=middle width=115.94097899999998pt height=50.04352485pt/></p>

and their output is the same since we assigned them the identity activation function. 

<p align="center"><img src="/tex/09afeca56ff80b12106c808e9ac097fb.svg?invert_in_darkmode&sanitize=true" align=middle width=137.9452833pt height=16.438356pt/></p>

Once the inputs have been propagated through the network, we can calculate the error. If we have multiple associations, we simply sum the error of each association. 

<p align="center"><img src="/tex/6057c698055c0474a4a50b2ad52f8b05.svg?invert_in_darkmode&sanitize=true" align=middle width=226.73802029999996pt height=47.60747145pt/></p>

#### The backward pass

Now that we have the error, we can use it to update each weight of the network by going backwards layer by layer. 

We know from *part 1* of this series that the change of a weight is the negative of that weight's component in the error gradient times the learning rate. For a weight between the last hidden layer and the output layer, we then have

<p align="center"><img src="/tex/d487ff384e18d4e3e131374a1b020be2.svg?invert_in_darkmode&sanitize=true" align=middle width=123.57475514999999pt height=38.5152603pt/></p>

We can find the error gradient by using the chain rule

<p align="center"><img src="/tex/6cec7be8c15cfa602950e6c372b3c241.svg?invert_in_darkmode&sanitize=true" align=middle width=852.4350730499999pt height=38.5152603pt/></p>

Therefore the change in weight is <img src="/tex/d81f8b3a4cf5f4f7e0143ae5e2968935.svg?invert_in_darkmode&sanitize=true" align=middle width=125.4901989pt height=22.831056599999986pt/>

For multiple associations, then the change in weight is the sum of each association <img src="/tex/b20396e46c8eadc3df6f07e271b44b7b.svg?invert_in_darkmode&sanitize=true" align=middle width=165.7869972pt height=32.256008400000006pt/>

Similarly, for a weight between hidden layers, in our case between the input layer and our first hidden layer, we have

<p align="center"><img src="/tex/18265f3ddb6ce1150d17f53abab8b33f.svg?invert_in_darkmode&sanitize=true" align=middle width=586.66213815pt height=48.18280005pt/></p>

Here the calculations are *slightly* more complex. Let's analyze the delta term <img src="/tex/a3ec72e0f05115605b57d81cfab96e7d.svg?invert_in_darkmode&sanitize=true" align=middle width=18.37621829999999pt height=22.831056599999986pt/> and understand how we got there. We start by calculating the partial derivative of <img src="/tex/f8bbbfffa921d3289fa9fdb9a1cf47c4.svg?invert_in_darkmode&sanitize=true" align=middle width=20.48055239999999pt height=14.15524440000002pt/> in respect to the error by using the chain rule

<p align="center"><img src="/tex/27d8e79faef062f9f5ee979318bc8d9f.svg?invert_in_darkmode&sanitize=true" align=middle width=120.17064509999999pt height=38.5152603pt/></p>

<p align="center"><img src="/tex/45a199a55d24da3882cf776b427f046a.svg?invert_in_darkmode&sanitize=true" align=middle width=515.20994085pt height=48.18280005pt/></p>

Remember that our activation function <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is the sigmoid function and that its derivative is <img src="/tex/63905ec601ca88b13ff9a43d55aee30f.svg?invert_in_darkmode&sanitize=true" align=middle width=105.09150299999999pt height=24.65753399999998pt/>

The change of a weight for <img src="/tex/2f118ee06d05f3c2d98361d9c30e38ce.svg?invert_in_darkmode&sanitize=true" align=middle width=11.889314249999991pt height=22.465723500000017pt/> associations is the accumulation of each association

<p align="center"><img src="/tex/f8986238597442b770c458c54abaccdc.svg?invert_in_darkmode&sanitize=true" align=middle width=145.85336865pt height=47.60747145pt/></p>

### Algorithm summary

First, initialize network weights to a small random value. 

Repeat the steps below until the error is about 0​

- for each association, propagate the network forward and get the outputs
  - calculate the <img src="/tex/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928075099999989pt height=22.831056599999986pt/> term for each output layer node (<img src="/tex/1d97e78591ccde2409d0caec79b07c35.svg?invert_in_darkmode&sanitize=true" align=middle width=103.91742074999999pt height=22.831056599999986pt/>)
  - accumulate the gradient for each output weight (<img src="/tex/edccb26d12f0480f8c7355b0daba34ab.svg?invert_in_darkmode&sanitize=true" align=middle width=120.49785659999999pt height=22.831056599999986pt/>)
  - calculate the <img src="/tex/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928075099999989pt height=22.831056599999986pt/> term for each hidden layer node (<img src="/tex/c0162beede82157cf166e9042561e60e.svg?invert_in_darkmode&sanitize=true" align=middle width=215.17631189999994pt height=32.256008400000006pt/>)
  - accumulate the gradient for each hidden layer weight (<img src="/tex/afab97b9eca9c4a3f27445364705c195.svg?invert_in_darkmode&sanitize=true" align=middle width=115.90401075pt height=22.831056599999986pt/>)
- update all weights and reset accumulated gradients (<img src="/tex/312098b63ae09679d41449dd93350d67.svg?invert_in_darkmode&sanitize=true" align=middle width=99.88377299999999pt height=22.465723500000017pt/>)

### Visualizing backpropagation

In this example, we'll use actual numbers to follow each step of the network. We'll feed our 2x2x1 network with inputs <img src="/tex/e4f0b9bce59fcd6b7ace485069c84ced.svg?invert_in_darkmode&sanitize=true" align=middle width=58.44761669999998pt height=24.65753399999998pt/> and we will expect an output of <img src="/tex/51d89c4114d0201a214771c31c6bff9f.svg?invert_in_darkmode&sanitize=true" align=middle width=30.137091599999987pt height=24.65753399999998pt/>. To make matters simpler, we'll initialize all of our weights with the same value of <img src="/tex/cde2d598001a947a6afd044a43d15629.svg?invert_in_darkmode&sanitize=true" align=middle width=21.00464354999999pt height=21.18721440000001pt/>. However, keep in mind that normally weights are initialized using random numbers. We will also design the network with a sigmoid activation function for the hidden layer and the identity function for the input and output layers and we'll use <img src="/tex/49f90d73df04e657a300620d7243bc8a.svg?invert_in_darkmode&sanitize=true" align=middle width=57.813874799999994pt height=21.18721440000001pt/> as our learning rate. 

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

On the way back, we first calculate the <img src="/tex/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928075099999989pt height=22.831056599999986pt/> term for the output node, <img src="/tex/3a6f0f1e5bd86ecdcec802bd7df7128a.svg?invert_in_darkmode&sanitize=true" align=middle width=254.62499204999997pt height=22.831056599999986pt/>

![backpropagation-visual](readme-images/backprop-visual-6.jpg)

And using the <img src="/tex/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode&sanitize=true" align=middle width=7.928075099999989pt height=22.831056599999986pt/> term we calculate the gradient for each weight between <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> and <img src="/tex/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode&sanitize=true" align=middle width=15.13700594999999pt height=22.465723500000017pt/> layer nodes: <img src="/tex/7c8a5cedf3d6bc7b8e6fe760613a03b8.svg?invert_in_darkmode&sanitize=true" align=middle width=288.30400004999996pt height=22.831056599999986pt/>

![backpropagation-visual](readme-images/backprop-visual-7.jpg)

We then do the same thing for each hidden layer (just the one in our case): <img src="/tex/7dd64c125a608844d8c312950bf44045.svg?invert_in_darkmode&sanitize=true" align=middle width=667.39760505pt height=32.256008400000006pt/>

![backpropagation-visual](readme-images/backprop-visual-8.jpg)

And calculate the gradient for each weight between <img src="/tex/21fd4e8eecd6bdf1a4d3d6bd1fb8d733.svg?invert_in_darkmode&sanitize=true" align=middle width=8.515988249999989pt height=22.465723500000017pt/> and <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> layer nodes: <img src="/tex/18a461f126c5140c2e3cb5cddab946e9.svg?invert_in_darkmode&sanitize=true" align=middle width=291.29291564999994pt height=22.831056599999986pt/>

![backpropagation-visual](readme-images/backprop-visual-9.jpg)

The last step is to update all of our weights using the calculate gradients. Note that if we had more than one association, then we would first accumulate the gradients for each association and then update the weights. 

<img src="/tex/baba6c7acdeb81dd49a08ab652816183.svg?invert_in_darkmode&sanitize=true" align=middle width=415.8191466pt height=22.465723500000017pt/>

<img src="/tex/b3d95619b1cf7580dde39b955cbcf7c9.svg?invert_in_darkmode&sanitize=true" align=middle width=423.66458969999996pt height=22.465723500000017pt/>

![backpropagation-visual](readme-images/backprop-visual-10.jpg)

As you can see the weights changed by a very little amount, but if we were run a forward pass again using the updated weights, we should normally get a smaller error than before. Let's check...

We had <img src="/tex/371bbbdd40fff829396823dfeddfb94f.svg?invert_in_darkmode&sanitize=true" align=middle width=74.79458414999999pt height=21.18721440000001pt/> on our first iteration and we get <img src="/tex/a4395fb18072a7710c2b1c84536b085a.svg?invert_in_darkmode&sanitize=true" align=middle width=92.66752769999998pt height=21.18721440000001pt/> after the weight changes. 

We had <img src="/tex/269cef8bad3668ba4ee6e04ccfab9729.svg?invert_in_darkmode&sanitize=true" align=middle width=115.42998224999998pt height=21.18721440000001pt/> and we get <img src="/tex/3768a90428bdba0b51a626d7c193cd96.svg?invert_in_darkmode&sanitize=true" align=middle width=140.08761029999997pt height=21.18721440000001pt/> after the weight changes. 

We successfully reduced the error! Although these numbers are very small, they are much more representative of a real scenario. Running the algorithm many times over would normally reduce the error down to almost 0 and we'd have completed training our network. 

## Code example

The example teaches a 2x2x1 network the XOR operator. 

<p align="center"><img src="/tex/45e155848e9c53acf0d057548b843990.svg?invert_in_darkmode&sanitize=true" align=middle width=383.295594pt height=39.452455349999994pt/></p>

![code-example](readme-images/backpropagation-code-example.jpg)

Where <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is the sigmoid function for the hidden layer nodes. 

Note that the XOR operation could not be solved with the linear network used in *part 1* because the dataset is distributed non-linearly. Meaning you could not pass a straight line between the four XOR inputs to divide them into the correct two categories. If we replaced the hidden layer node activation functions from sigmoid to identity, this network wouldn't be able to solve the XOR problem as well. 

Feel free to try it out yourself and experiment with different activation functions, learning rates and network topologies. 

## References

- Artificial intelligence engines by James V Stone (2019)
- Complete guide on deep learning: http://neuralnetworksanddeeplearning.com/chap2.html
- Flow of backpropagation visualized: https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/
- Activation functions: https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0
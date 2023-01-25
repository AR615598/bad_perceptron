<h1>Writing a ML ALG from scratch</h1> 
<body>Im pretty much just doing this for fun since I’m interested in machine learning but as of writing this I only have a surface level understanding since I have only used libraries so far like scikit-learn.</body>

<h2>Basic understanding</h2>
For this project I will be attempting to create the simplest algorithm I can think of a perceptron. Im not sure how many layers I will be able to do so I will be starting with just one layer. A single-layer perceptron is a neural network and takes in an input and outputs a binary classification that is either a 1 or 0 which maps to “yes” or “no”. An example classification problem is is this a doughnut, if the provided input is a doughnut we want it to return a 1 otherwise a 0. It should also be noted that the perceptron is a linear classification algorithm, so the problem needs to have a linear decision boundary. Though being completely honest, I am not sure what makes one algorithm linear and others not, but that’s the purpose of this project. But I’m assuming that it is linear since the algorithm will tune the decision boundary which itself is linear. It is imitating a neuron, they accept inputs via its dendrites, and through neurotransmitters send information to other neurons. But to simulate this I will aim to have my algorithm to take in these inputs to create a linear function that will decide its output called activation. 

activation = sum(weight_i * x_i) + bias

With this value we will be able make a prediction function that will classify the input. An example is:

prediction = 1.0 if activation >= 0.0 else 0.0

With this prediction function we will always have two outputs or classes, 1 or 0, separated by some linear decision boundary function 

<h2>Deciding the weights</h2>
In order for the algorithm to actually make the linear function it need to have appropriate weights to fit the training set. To do this we will be using gradient decent, gradient decent is an optimization algorithm using the cost function within gradient decent which acts like a barometer, Gauging its accuracy with each iteration of parameter updates. Until the function is close or equal to zero the model will continue to adjust its parameters to yield the smallest possible error. 
Steps of gradient decent 
1. Find the slope of the objective function in respect to each of the features. In other words compute the gradient of the function 
2. Pick a random value for the parameters 
3. Update the function by changing the parameter values
4. Calculate the step sizes for each feature 
    1. step_size = gradient * learning rate 
5. New params = old params - step_size 
6. Repeat step 3-5 until gradient is close or equal to zero.
It should be noted that the learning rate is very flexible with the larger value making it take larger steps speeding up the algorithm, but because it takes large steps down the slope it is possible to skip the minimum value, so generally it is better to keep the value low. But in theory it may be possible to change the learning rate to change in relation to each iteration, decreasing with lower slope and increasing at higher ones.(Just an idea but seems interesting and plausible)

<h2>Stochastic Gradient Descent (SGD)</h2>
With normal gradient descent it struggles with problems with many features and datapoints because of the high overhead making it inefficient. One very simple way to reduce its overhead is to introduce some randomness. We will randomize which data points we will be using enabling us to reduce the overhead but with the sacrifice of accuracy but overall the loss is not significant and the time saved outweighs it. But with smaller datasets it may be better to test on the whole thing. 

With SGD we can find the weights with the lowest error, but it should be noted this is done on each feature one at a time. 

w = w + learning_rate * (expected - predicted) * x

<h2>Setting Everything Up </h2>
First we need to set up our algorithm design, so we need a problem. For this project I will be using Connectionist Bench (Sonar, Mines vs. Rocks) Data Set which can be used for binary classification since it has two classes M and R. Each row has 59 features not including the classification. So we need to create a vector with 60 elements so we can find the dot product of each rows features against the weights. I added another element in the vector to account for bias. Bias allows us to shift our activation function to the left and right, you can think of it as the b term in y = mx + b. With each rows dot product we can compare it against the threshold with the decision boundary and get the predicted class.  But since the weights are going to be some value, likely zero the majority of our predictions will be incorrect, so we need to update the weights using SGD. But since I don’t want to hard code this I need to implement fit and predict and allow it to work on any number of dimensions. 
 
<h2>Notes</h2>
This is not completed there are some things i want to change and optimize. Currently the model struggles with overfit and recognising false positves, this is due to how I did not randomize the dataset so the training set was almost completely composed of the positive class, so it had very little exposure to data points of the negative class. By testing on the entire dataset the model had similar issues and had a very low accuracy of around 50%, still struggling with false positives. 

<h2>Refrences</h2>
https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31
https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
https://towardsdatascience.com/6-steps-to-write-any-machine-learning-algorithm-from-scratch-perceptron-case-study-335f638a70f3


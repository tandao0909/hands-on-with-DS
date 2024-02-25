1. Because the exercise simply ask us to play with the [TensorFlow playground](https://playground.tensorflow.org), there is no question to answer.
2. This exercise asks us to draw, so see it [here]().
3. 
- Recall that logistic regression classifier is similar to a classic perceptron, the only difference is the former uses the sigmoid function as the activation function, while the latter uses the step function.
- Because using a step function, classic perceptron can't predict the probability for each class, it only can predict which class is most likely the answer. In contrast, logistic regression classifier will predict the class probabilities.
- A classical perceptron will converge only if the dataset is linearly separable, while a logistic regression will generally converge to a reasonably good solution even if the dataset is not linearly separable.
- If you want to tweak a classical perceptron to make it equivalent to a logistic regression classifier, you just change the activation function to the sigmoid function and train the the perceptron using Gradient Descent (or other optimization algorithms to minimize the cost function, typically cross entropy).
4. The reasons the sigmoid activation function is a key ingredient in training the first MLPs:
- Its derivate is positive everywhere, so Gradient Descent can always roll down the slope.
- In contrast, the derivate of step function is either 0 or undefined, so Gradient Descent cannot move, as there's no slope at all.
6. 
- The shape of the input matrix $\textbf{X}$ is $m \times 10$, where m is the training batch size.
- The shapes of the hidden layer's weight matrix $\textbf{W}_h$ is $10 \times 50$, of the bias vector $\textbf{b}_h$ is $50$.
- The shapes of the output layer's weight matrix $\textbf{W}_o$ is $50 \times 3$, of the bias vector $\textbf{b}_o$ is $3$.
- The shape of the network's output matrix $\textbf{Y}$ is $m \times 3$.
- The equation to compute network's output matrix is:
    $$\textbf{Y} = \phi_o(\phi_h(\textbf{X}\textbf{W}_h + \textbf{b}_h)\textbf{W}_o + \textbf{b}_o)$$
    where $\phi_o$ is the activation function of the output layer and $\phi_h$ is the activation function of the hidden layer.
7. 
- Classifying email into spam or ham is a binary classification task, thus we need only one neuron in the output layer to predict the probability of the positive class (can be either spam or ham in this case). THe activation function we should use is the sigmoid function.
- If instead we deal with the MNIST dataset, then this is a multiclass classification task, hence we need 10 neurons in the output layer (one for each class, and there're 10 classes in the MNIST dataset).
- If we want our network to predict housing prices in the California housing dataset, then this is a regression task, and we only need to predict one thing (the housing price), hence one neuron is enough for the output layer. We don' need any activation function in this situation.
- There's a note from the author though: When the value we need to predict can vary many orders of magnitude, then it can be better to predict the log of the target value instead of the target value directly.
8. 
- Backpropagation is a technique used in ML. It calculates how much each weight and bias influence the cost function (e.g., the gradient of the cost function with regard to that hyperparameter), and apply a Gradient Descent step using these gradient.
- Reverse-mode autodiff is an algorithm used by backpropagation to efficiently calculate these gradients. It performs a forward pass through the computational graph, calculate the value at each node in the graph for the current training batch, then performs a reserve pass to compute all the gradients at once.
- The difference is backpropagation refers to the whole process of performing multiple backpropagation steps, each of which computes the gradients and uses them to perform a Gradient Descent step, while reverse-mode autodiff is just a technique used by backpropagation to compute the gradients efficiently.
9. Here is the list of all the hyperparameters we can tweak in a basic MLP:
- The number of hidden layers
- The number of neurons in each hidden layer.
- The learning rate
- The optimizer
- The batch size
- The activation function in each hidden layer and the output layer.
Here is some notes from the author about how to choose the activation function:For the output layer, you want to use the sigmoid for binary classification, softmax for multiclass classification or none for regression. For hidden layers, ReLU (or one of its variant, will discuss in chapter 11) is a good default.
If the MLP overfits the training data, then you can:
- Reduce the number of hidden layers 
- Reduce the number of neurons in each hidden layer
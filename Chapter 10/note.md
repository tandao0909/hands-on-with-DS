# From biological to Artificial

## Logical Computations with Neurons

- Artificial neurons can be seen as a node, symbolize for a function which has many inputs and one output.
- Each input is binary (on/off) and output is also binary
- Here is [Tensorflow's playground](https://playground.tensorflow.org/) to play with.
- Conclusion after playing (Maybe wrong):
    - The activation affect a lot on the shape of the decision boundaries, in case of ReLU, the decision boundaries are straight lines.
    - The more hidden layers, the more complex structure that the neural network can learn.
    - On the other hands, the smaller the neural network, the incapable it is to capture the complex structure of training data set.

## The perceptron:

- The perceptron is one of the simplest ANN architectures.
- It is based on a slightly different artificial neuron named *threshold logic unit* (TLU) or *linear threshold unit* (LTU).
- The inputs and outputs are numbers (instead of on/of binary values):
    - Each input connection is associated with a weight.
    - Each output number is associated with a bias.
- The TLU first calculate the dot product between the inputs and the weights, then add the bias:
    $$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = w^Tx+b$$
- Then we apply a step function (A step function is a function changes its value at certain points and remains constant between these points) to the result.
- This is nearly the same as logistic regression (chapter 4), we just apply the step function in stead of sigmoid.
- Like the logistic regression, the hyperparameters are the weights term w and the bias term b.
- The most common step function is *Heaviside step function*:
    $$heaviside (z) = \begin{cases} 
    0 \text{ if } z < 0 \\
    1 \text{ if } z \geq 0 \\
    \end{cases}$$
- Sometimes the sign function can be used:
    $$sign (z) = \begin{cases} 
        -1 \text{ if } z < 0 \\
        0 \text{ if } z = 0 \\
        1 \text{ if } z > 0 \\
        \end{cases}$$
- A single TLU can be used to train simple linear binary classification. It computes a linear function of its inputs. Then if the result exceeds a certain threshold, it predicts the positive class; otherwise it predicts the negative class.
- This is ver similar to logistic regression (Chapter 4) and linear support vector machine classification (Chapter 5).
- A perceptron is a layer of one or more TLU composed together, where every TLU is connected to every input.
- Such layer is called a fully connected layer, or a dense layer.
- This layer of TLUs produces the final outputs, which is called the output layer.
- Because we can create multiple TLUs at the same neural network, which means we can create a multiclass classifier.
- Using linear algebra, we can calculate the result of multiple TLUs at the same time:
    $$h_{W, b} (X) = \phi(XW+b)$$
- In this equation:
    - $\phi$ is the activation function.
    - X is the input matrix, with each row is an instance and each column is an feature.
    - W is the weight matrix, with each row is an input neuron and each column is an output neuron.
    - b is the bias row vector, with each element is associated with an output neuron.
- Note: In math, add a matrix and a vector makes no sense. However, in CS, we allow something called "broadcast", which means after calculating the matrix, we add each row of the matrix with the vector. After that, we apply $\phi$ element-wise to every element in the result matrix.
- The way we train the neuron network based on an observation on biological neuron: "Cells that fire together, wire together". 
- So we encourage neuron connection that reduces the error.
- The *weight update* equation is:
    $$w_{i, j} ^{\text{next step}} = w_{i, j} - \eta(\hat{y_j}-y_j)x_i$$
- In this equation:
    - $w_{i, j}$ is the weight between the connection of the i-th input neuron and j-th output neuron.
    - $x_i$ is the i-th feature of the current training instance.
    - $\hat{y_j}$ is the output of the j-th neuron of the current training instance.
    - $y_j$ is the j-th target output for the current training instance.
    - $\eta$ is the learning rate (Similar to Logistic regression in chapter 4).
- The decision boundaries of each output neuron is linear, so Perceptrons are incapable of learning complex structure (similar to Logistic Regression).
- However, if the training dataset is linearly separable, then Rosenblatt, the author of Perceptron, [proofs](https://en.wikipedia.org/wiki/Perceptron#Convergence_of_one_perceptron_on_a_linearly_separable_dataset) that this algorithm would converge to a solution. Note that this solution is not unique, as there are an infinity amount of hyperplane that can separate a linearly separable dataset.
- The Perceptron's implementation is very similar to SGD. In fact, in scikit-learn, we can implement Perceptron using SGDClassifier, as shown in the learn notebook.
- However, perceptron has some serious weaknesses, like incapable of solving some trivial problems (e.g. the *exclusive OR* (XOR) classification problem). Of course other linear classification model also suffer from the same problems.
- But, we can overcome some limitations of Perceptrons by stacking several layers of them on each other. 

## The Multilayer Perceptron and Backpropagation

- An MLP(Multi Layer of Perceptrons) is composed of :
    - An input layer
    - An output layer
    - One or many layer between input and output
- The layers close to the input layer are usually called *lower layers*, while the layers close to the output layer are usually called *upper layers*.
- Notice that the calculation is flown only in one direction, hence this architecture is an example of *feed forward network*(FNN).
- When an ANN contains a deep stack of hidden layers, it is called a *deep neural network*(DNN). The number of hidden layers for a neural network to be considered deep is somewhat unclear, but the number is usually more than 2.
- Many generations of researchers have studied how to train a neural network and we, stand on the shoulder on many giants, have some best practices passed down to us by them:
    - Reverse-mode automatic differentiation (or reversed-mode auto diff for short): Using two passes through the network (one forward, then one backward), we can calculate the gradients of the neural network's errors with regard to every weights and biases. In other words, we can find out how to tweak each weights and biases to reduce the error the most efficiently. 
    - We combine this with gradient descent and have *the most popular AI algorithms* nowadays, named **Backpropagation**.
- Backpropagation can be applied to all sorts of computational graph, not only neural networks.
- Let's walk through the process of backpropagation step by step:
    - It handles one mini-batch at a time (for example, containing 32 instances each), and it goes through the full training multiple times. Each pass is called an *epoch*.
    - Each mini-batch enters the network through the input layer. At each layer, we computes the output using the input from the previous layer, then pass to the next layer. This is the *forward pass*: This is exactly the same as making predictions, expect we save all the intermediate result for the backward pass.
    - Next, the algorithm computes the error. That is, we use the cost function to compare the predict output and the desired output and return some measure of error.
    - Then we computes how much each weight and bias of the output layer influences the error (i.e. what is the partial derivate of the error function with regard to the weight and bias). Note that this is a **number**, not a function with many variable, because we have plugged the number into the function. We done this analytically by applying *the chain rule*.
    - After that, the algorithm continues to measures how much the weights and biases of the below layers contributes to the error, again using the chain rule, working backward until we reach the input layer. This reverse pass effectively computes the error gradient across all the connection weights and biases in the network by propagating (a synonym of spread) the error gradient backward through the neural network (hence the name of the algorithm).
    - Finally, the algorithm performs a gradient descent step to tweak all the weights and biases using the gradient we just computed.
- Note that it is crucial to initialize the weights and biases randomly. For example, if you initialize all the weight and bias to the same number, then all neurons in a given layer will be perfectly identical. So the propagation step will treat all of them equally, so after the gradient descent step, all weights and biases in the same layer will remain the same. So at the end, your layer will perform just like it has only one neuron, leads to the fact that it would not be smart. if you initialize the parameters randomly instead, you break the symmetry of the network and allow different patterns can emerge form the neurons.
- In short, backpropagation predicts a mini-batch (forward-pass), computes the error, and then traverse the layers back and calculate the gradient error for each parameter (backward-pass), after that, it performs a gradient descent step on every parameter.
- So if you backpropagation to work, you need a cost function that is differentiable and the its differentiation is not 0. because gradient descent cannot move on a flat surface.
- That is the reason why we don't use step functions as an activation function nowadays, because step functions consist of many flat segments, which means gradient descent will perform poorly on it.
- Three popular choice nowadays is:
    - Sigmoid function:
    $$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$
    This function outputs a number ranges from 0 to 1 in a smooth shape. It is also continuous and differentiable.
    - Tanh function:
    $$tanh(z) = \frac{e^x-e^{-x}}{e^x-e^{-x}} = 2\sigma(2z)- 1$$
    This function is somewhat similar to sigmoid function. It outputs a number ranges from -1 to 1 and has S-shape, continuous and differentiable. This range helps the each layer's output closer to 0 at the beginning of training, which helps speed up converge.
    - The rectified linear unit function (ReLU):
    $$ReLU(z) = max(0, z)$$
    The ReLU is continuous but not differentiable at 0 and its derivate when $z<0$ is 0. So around 0, the slope change suddenly, so the gradient descent would bounce the model around. In practice, however, ReLU works very well and has the advantage of being fast to compute, so it has become the default (this is a case where the biology model can be misleading, because while it seems like biological neuron follow a roughly S-shape activation function, but ReLU is better in practice).
- But why we need activation functions in the first place? 
    > Imagine we don't have activation functions in the first place. Then we would chain multiple linear transforms and apply them sequentially. But, if you apply linear transform on a line, you obtain a line. So in the end, you just have a line. In conclusion, if you don't have some form of nonlinearity, then even with a deep stack of neuron layers, you can't solve a complex problem with it. In contrast, a large enough with nonlinear activation functions can [approximate any function](https://www.youtube.com/watch?v=TkwXa7Cvfr8&t=230s) to any degree of precisions.

## Regression MLPs

- MLPs can be used to train a regression model. Then the number of neurons in the input layer is the number of features and the number of neurons in the output layer is the number of output dimensions.
- You can use MLPRegressor in scikit-learn to implement MLPs. The implementation detail is in the learn notebook.
- There should be no activation function in the output layer, so the neural network is free to output any values it wants.
- But if you want to add some restrictions on top of the output, then you can apply an activation function to the output player. Here are some usually encountered situations and their solutions:
    - You want the output to be non-negative. Then you can apply a ReLU to the output layer. Alternatively, you can try *softplus* function, which is a smooth variant of ReLU:
        $$\text{softplus}(z) = \ln(1 + e^z)$$
    - You want the output to fall within a given range, then you can use sigmoid or tanh, and scale it to your desired range after that: 0 to 1 for the logistic function, -1 to 1 for the hyperbolic tangent function.
- However, there are some limitations of MLPs in scikit-learn:
    - Unbounded, so it can be output a very large number, which can lead to numerical overflow.
    - Dying ReLU problem: If the parameters are happened to output negative weighted sum with all the training instances, then apply ReLU will just output 0. Then the backpropagation algorithm will not affect the neuron at all. In CS's words, the neuron is "sleeping", which means it is not learning anymore temporally. There are some variants of ReLU to counter back this disadvantage, such as Leaky ReLU.
- The default loss function is MSE, but if you expect the dataset to have a lot of outliers, then you can use Huber loss function, which is a combination of MSE and MAE (mean absolute error):
    $$\text{Huber}(z) = \begin{cases}
        \frac{1}{2} z^2 \text{ if } z < \delta \\
        \delta \left(|z| - \frac{1}{2} \delta\right) \text{otherwise} \\
    \end{cases}$$
    where $\delta$ is the function's hyperparameter to control how it behaves:
    - If $z < \delta$, then the model will converges fast.
    - If $z \geq \delta$, then the model will not be so sensitive to outliers.

| Hyperparameter              | Typical value                |
|-----------------------------|------------------------------|
| # input neuron              | One per input feature (e.g. 28 x 28 = 784 for MNIST) |
| # hidden layers             | Depend on the problem, but typically 1 to 5 |
| # neurons per hidden layer  | Depend on the problem, but typically 10 to 100 |
| # output neurons            | 1 per output dimension |
| Hidden activation           | ReLU (or SELU, in chapter 11) |
| Output activation           | None, or ReLU/Softplus (if positive outputs), or sigmoid/tanh (if bounded outputs) |
| Loss function               | MSE/Huber |

## Classification MLPs

- MLPs can be used for classification tasks.
- For a 
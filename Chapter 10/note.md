# From biological to Artificial

## Logical Computations with Neurons

- Artificial neurons can be seen as a node, symbolize for a function which has many inputs and one output.
- Each input is binary (on/off) and output is also binary.
- Here is [TensorFlow's playground](https://playground.tensorflow.org/) to play with.
- Conclusion after playing (Maybe wrong):
    - The activation affect a lot on the shape of the decision boundaries, in case of ReLU, the decision boundaries are straight lines.
    - The more hidden layers, the more complex structure that the neural network can learn.
    - On the other hands, the smaller the neural network, the incapable it is to capture the complex structure of training data set.

## The perceptron:

- The perceptron is one of the simplest ANN architectures. It is a node in the neural network which receives input from every node in the previous layer and apply a weighted sum to there inputs to output a single number.
- It is based on a slightly different artificial neuron named *threshold logic unit* (TLU) or *linear threshold unit* (LTU).
- The inputs and outputs are numbers (instead of on/of binary values):
    - Each input connection is associated with a weight.
    - Each output number is associated with a bias.
- The TLU first calculate the dot product between the inputs and the weights, then add the bias:
    $$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = w^Tx+b$$
- Then we apply a step function (A step function is a function changes its value at certain points and remains constant between these points) to the result.
- This is nearly the same as logistic regression (chapter 4), we just apply the step function instead of sigmoid.
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
- This is very similar to logistic regression (Chapter 4) and linear support vector machine classification (Chapter 5).
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
- Note: In math, add a matrix and a vector makes no sense. However, in CS, we allow something called "broadcasting", which means after calculating the matrix, we add each row of the matrix with the vector. After that, we apply $\phi$ element-wise to every element in the result matrix.
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
- However, if the training dataset is linearly separable, then Rosenblatt, the author of Perceptron, [proofed](https://en.wikipedia.org/wiki/Perceptron#Convergence_of_one_perceptron_on_a_linearly_separable_dataset) that this algorithm would converge to a solution. Note that this solution is not unique, as there are an infinity amount of hyperplane that can separate a linearly separable dataset.
- The Perceptron's implementation is very similar to SGD. In fact, in Scikit-learn, we can implement Perceptron using SGDClassifier, as shown in the learning notebook.
- However, perceptron has some serious weaknesses, like incapable of solving some trivial problems (e.g. the *exclusive OR* (XOR) classification problem). Of course other linear classification model also suffer from the same problems.
- But, we can overcome some limitations of Perceptrons by stacking several layers of them on each other. 
- Contrary to Logistic regression classifiers, perceptrons don't output a class probability. This is one reason to prefer logistic regression over perceptrons. Moreover, perceptrons do not use any regularization by default, and training stops as soon as there is noe more prediction errors on the training set, so the model typically doesn't generalize as well as logistic regression or a linear SVM classifier. However, perceptrons may train a bit faster.

## The Multilayer Perceptron and Backpropagation

- An MLP (Multi Layer of Perceptrons) is composed of :
    - An input layer
    - An output layer
    - One or many layer between input and output
- The layers close to the input layer are usually called *lower layers*, while the layers close to the output layer are usually called *upper layers*.
- Notice that the calculation is flown only in one direction, hence this architecture is an example of *feed forward network*(FNN).
- When an ANN contains a deep stack of hidden layers, it is called a *deep neural network*(DNN). The number of hidden layers for a neural network to be considered deep is somewhat unclear, but the number is usually more than 2.
- Many generations of researchers have studied how to train a neural network and we, stand on the shoulder on many giants, have some best practices passed down to us by them:
    - Reverse-mode automatic differentiation (or reversed-mode auto diff for short): Using two passes through the network (one forward, then one backward), we can calculate the gradients of the neural network's errors with regard to every weights and biases. In other words, we can find out how to tweak each weights and biases to reduce the error the most efficiently. 
    - We combine this with gradient descent and have *the most popular AI algorithms* nowadays, named **Backpropagation**.
- Backpropagation can be applied to all sorts of computational graphs, not only neural networks.
- Let's walk through the process of backpropagation step by step:
    - It handles one mini-batch at a time (for example, containing 32 instances each), and it goes through the full training multiple times. Each pass is called an *epoch*.
    - Each mini-batch enters the network through the input layer. At each layer, we computes the output using the input from the previous layer, then pass to the next layer. This is the *forward pass*: This is exactly the same as making predictions, expect we save all the intermediate result for the backward pass.
    - Next, the algorithm computes the error. That is, we use the cost function to compare the predict output and the desired output and return some measure of error.
    - Then we computes how much each weight and bias of the output layer influences the error (i.e. what is the partial derivate of the error function with regard to the weight and bias). Note that this is a **number**, not a function with many variable, because we have plugged the number into the function. We done this analytically by applying *the chain rule*.
    - After that, the algorithm continues to measures how much the weights and biases of the below layers contributes to the error, again using the chain rule, working backward until we reach the input layer. This reverse pass effectively computes the error gradient across all the connection weights and biases in the network by propagating (a synonym of spread) the error gradient backward through the neural network (hence the name of the algorithm).
    - Finally, the algorithm performs a gradient descent step to tweak all the weights and biases using the gradient we just computed.
- Note that it is crucial to initialize the weights and biases randomly. For example, if you initialize all the weight and bias to the same number, then all neurons in a given layer will be perfectly identical. So the propagation step will treat all of them equally, so after the gradient descent step, all weights and biases in the same layer will remain the same. So at the end, your layer will perform just like it has only one neuron, leads to the fact that it would not be smart. if you initialize the parameters randomly instead, you break the symmetry of the network and allow different patterns can emerge form the neurons.
- In short, backpropagation predicts a mini-batch (forward-pass), computes the error, and then traverse the layers back and calculate the gradient error for each parameter (backward-pass), after that, it performs a gradient descent step on every parameter.
- So if you want backpropagation to be working, you need a cost function that is differentiable and the its differentiation is not 0, because gradient descent cannot move on a flat surface.
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
- In this chapter, we will implement a Dense layer, which is just a layer consists of many Perceptron.

## Regression MLPs

- MLPs can be used to train a regression model. Then the number of neurons in the input layer is the number of features and the number of neurons in the output layer is the number of output dimensions.
- You can use MLPRegressor in Scikit-learn to implement MLPs. The implementation detail is in the learning notebook.
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
| Loss function               | MSE or Huber if outliers |

## Classification MLPs

- MLPs can be used for classification tasks.
- For a binary classification problem, you just need to use a single output neuron using the sigmoid activation function: the output will be a number between 0 and 1, which you can interpret as the estimated probability of the positive class. The estimated probability of the negative class is equal to 1 minus that number.
- MLPs can also handle multilabel classification tasks (mentioned in chapter 3).
- For example, you can have an email classification system that predicts whether or not each incoming email is spam or ham, and simultaneously predicts whether it is urgent or non-urgent email. In this case, you would need two output neurons, both using the sigmoid activation function: the first would output the probability hat the email is spam and the second would output the probability that it is urgent. 
- More generally, you would dedicate one output neuron for each positive class. Note that the output probabilities do not necessarily add up to 1. This lets the model output any combination of labels: you can have non-urgent ham, urgent ham, non-urgent spam, and perhaps even urgent spam (although that would probably be an error).
- If each instance can belong only to a single class, out of three or more possible classes (e.g., classes 0 through 9 for digit image classification), then you need to one output neuron per class, and you should use the softmax activation function for the whole output layer. The softmax function (mentioned in chapter 4) will ensure that all the estimated probability are between 0 and 1 and they add up to 1, since the classes a exclusive. As you saw in chapter 3, this is called multiclass classification.
- Regrading the loss function, since we are predicting probability distributions, the cross-entropy loss (or *x-entropy* or log loss for short, see chapter 4) is generally a good choice.
- Scikit-learn has an `MLPClassifier` class in the `sklearn.neural_network` package. It is almost identical to the `MLPRegressor` class, expect that it minimizes the cross entropy instead of MSE.
- Typically classification MLP architecture:

| Hyperparameter              | Binary classification         | Multilabel binary classification | Multiclass classification |
|-----------------------------|-------------------------------|----------------------------------|----------------------------|
| Hidden layers               | Typically 1 to 5 layers, depending on the task |                                                        
| # Output neurons            | 1                             | 1 per label                      | 1 per class                |
| Output layer activation     | Sigmoid function              | Sigmoid function                 | Softmax function           |
| Loss function               | X-entropy                     | X-entropy                        | X-entropy                  |

# Implementing MLPs with Keras

- Keras is TensorFlow's high-level deep learning API: it allows you to build, train , evaluate, and execute all sorts of neural networks.

## Building an Image Classifier Using the Sequential API

### Using Keras to load the dataset

- We will use the Fashion MNIST dataset. This dataset is similar to MNIST, but consists of fashion items instead of handwritten digits.
- This dataset is significantly more challenging than MNIST.
- Most of this part will be in the learning notebook, as it focuses more on code instead of knowledge.

### Creating the model using the sequential API

- We go through the first code block in the learning notebook line by line:
    - First, we set TensorFlow's random seed to make the result reproducible: the random weights of the hidden layers and the output layer will be the same every time you run the notebook. You could use `tf.keras.utils.set_random_seed()` function, which conveniently set the random seeds for TensorFlow, Python (`random.seed()`), and NumPy (`np.random.seed()`).
    - The next line creates a `Sequential` model. This is the simplest kind of Keras model for neural networks that are just composed of a single stack of layers connected sequentially. This is called the sequential API.
    - Next, we build the first layer (an `Input` layer) and add it to the model. We specify the input `shape`, which doesn't include the batch size, only the shape of the instances. TensorFlow needs to know the shape of the inputs so it can determine the shape of the connection weight matrix of the first hidden layer.
    - Then we add a `Flatten` layer. Its role is to convert each input image into an 1D array: for example, if it receives a batch of shape [32, 28, 28], it will reshape this batch to [32, 784]. In other words, if it receives input data `X`, it computes `X.reshape(-1, 784)`. This layer doesn't have any parameters, it's juts there to do some simple preprocessing.
    - Next we add a `Dense` hidden layer with 300 neurons. It will use the ReLU activation function. Each `Dense` layer manages its own weight matrix, containing all the connection weights between the neurons and their inputs. It also manages a vector of bias terms (one per neuron). When it receives some input data, it computes the hypothesis function in [this section](#the-perceptron).
    - Then we add a second `Dense` hidden layer with 100 neurons, also using the ReLu activation function.
    - Finally, we add a `Dense` output layer with 10 neurons (one per class), using the softmax activation function because the classes are exclusive.
- Specifying `activation="relu"` is equivalent to specifying `activation=tf.keras.activations.relu`. Other activation functions are available in the `tf.keras.activations` package. We will many of them, and you can find the full list in [the documentation](https://keras.io/api/layers/activations). We will also define our own custom activation functions in Chapter 12.
- The model's `summary()` method displays all the model's layers, including:
    - Each layer's name (which is automatically generated unless you set it when creating the layers).
    - Each layer's shape (`None` means the batch size can be anything)
    - Each layer's number of parameters.
    - Ends with the total number of parameters, including trainable and non-trainable parameters.
- Alternatively, you can use `tf.keras.utils.plot_model()` to generate an image of your model.
- Note that `Dense` layers often have a lot of parameters. For example, the first hidden layer has $784 \times 300$ connection weights, plus 300 bias terms, which adds up to 235,500 parameters. This gives the model quite a lot of flexibility to fit the training set, but it also means the model has a risk of overfitting, especially when you don't have a lot of training data.
- Each layer has a unique name (e.g., `"dense_2"`). You can set the name explicitly using the construct's `name` argument, but it generally is simpler to let Keras name the layers automatically. Keras takes the layer's class name and converts it to snake case (e.g., a layer from the `"MyAbsolutelyNormalLayer"` will be named `"my_absolutely_normal_layer"` by default).
- Keras also ensures that the name is globally unique, even across the same model, by appending an index if needed, as in `"dense_2"`. The reason why Keras tries to make the name unique across models is to make it possible to merge models easily without getting name conflicts.
- All global state managed by Keras is stored in a *Keras session*, which you can clear using `tf.keras.backend.clear_session()`. In particular, this resets the name counters.
- Notice that the `Dense` layer initialized the connection weights randomly (which is needed to break symmetry, as discussed earlier), and the biases are initialized to zeros, which is fine. If you want to use a different initialization method, you can set `kernel_initializer` (*kernel* is another name for the matrix of the connection weights) or `bias_initializer` when creating the layer. We will discuss initializers further in chapter 11, and the full list can be found in [the documentation](https://keras.io/api/layers/initializers).
- The shape of the weight matrix depends on the number of inputs, which is why we need to specify the `input_shape` when creating the model. If you do not specify the input shape, it's OK: Keras will simply wait until it knows the input shape before it actually builds the model parameters. This will happen either when you feed it some data (e.g., during training) or when you call its `build()` method. Until the model parameters are built, you will not be able to do certain things, such as display the model's summary or save the model. So if you know the input shape when creating the model, it is best to specify it.

### Compiling the model

- After a model is created, you must call its `compile()` method to specify the loss function and the optimizer to use. Optionally, you can specify a list of extra metrics to compute during training and evaluation.
- Using `loss="sparse_categorical_crossentropy"` is the equivalent of using `loss=tf.keras.sparse_categorical_crossentropy`. Similarly, using `optimizer="sgd"` is the equivalent of using `optimizer=tf.keras.optimizers.SGD()`, and using `metrics=["accuracy"]` is the equivalent of using `metrics=[tf.keras.metrics.sparse_categorical_accuracy]` (when using this loss function). We will use many other losses, optimizers, and metrics; for the full list, see the the documentation for [losses](https://keras.io/api/losses), [optimizers](https://keras.io/api/optimizers) and [metrics](https://keras.io/api/metrics).
- We use the `"sparse_categorical_crossentropy"` loss because we have sparse labels (i.e., for each instances, there is just a target index, from 0 to 9 in this case), and the classes are exclusive. 
- Note that "sparse" term here is the format of the label, not the absence or presence of the labels. We use "sparse", as it more memory-efficient to store each target class as a single integer, instead of a one-hot vector.
- If instead we had one target probability per class for each instance (such as one-hot vectors, e.g., `[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]` to present class 3), then we would need to use the `"categorical_crossentropy"` loss instead.
- If we were doing binary classification or multilabel binary classification, then we would use the `"sigmoid"` activation function in the output layer instead of the `"softmax"` activation function, and we would use the `"binary_crossentropy"` loss.
- If you want to convert sparse labels (i.e., class indices) to one-hot vector labels, use the `tf.keras.utils.to_categorical()` function. To go the other way round, use the `np.argmax()` function with `axis=1`.
- Regarding the optimizer, `"sgd"` means that we will train the model using stochastic gradient descent. In other words, Keras will perform the backpropagation algorithm described earlier (i.e., reversed-mode autodiff plus gradient descent). There are more efficient optimizers, which we will discussed in chapter 11. They improve gradient descent, not autodiff.
- When using the `SGD` optimizer, it is important to tune the learning rate. So you will generally want to use `optimizer=tf.keras.optimizer.SGD(learning_rate=__???__)` to set the learning rate, rather than `optimizer="sgd"`, which defaults to a learning rate of 0.01.
- Finally, since this is a classifier, it's useful to measure its accuracy during training and evaluation, hence we set `metrics=["accuracy"]`.

### Training and evaluating the model

- Now we can train the model by simply call its `fit()` method.
- We pass it the input features (`X_train`) and the target class (`y_train`), as well as the number fo epochs to train (or else it would default to 1, which would definitely not enough to converge t a good solution).
- We also pass a validation (this is optional). Keras will measure the loss and the extra metrics on this set at the end of each epoch, which is very useful to see how well the model really performs.
- If the performance on the training set is much better than the validation set, your model is probably overfitting the training set, or there is a bug, such as a data mismatch between the training set and the validation set.
- At each epoch during training, Keras displays the number of mini-batches processed so far on the left side of the progress bar.
- The batch size is 32 by default, and the training set consists of 55,000 images, the model goes through 1,719 mini-batches per epoch: 1,718 of size 32, 1 of size 24. 
- After the progress bar, you can see the mean training time per sample, and the loss and accuracy (and any other extra metrics you asked for earlier) on both the training set and the validation set.
- Instead of passing a validation set using `validation_data` argument, you could set `validation_split` to the ratio of the training set that you want Keras to use for validation. For example, `validation_split=0.1` tells Keras to use the last 10% of the data (before shuffling) for validation.
- If the dataset was very skewed, with some classes being overrepresented and others underrepresented, it would be useful to set the `class_weight` argument when calling the `fit()` method, to give a larger weight to underrepresented classes and a lower weight to overrepresented classes. These weights would be used by Keras when computing the loss.
- If you need per-instance weights, set the `sample_weight` argument. 
- If both `class_weight` and `sample_weight` are provided, then Keras multiplies them.
- Per-instances could be useful, for example, if some instances are labeled by experts, while others are labeled using a crowdsourcing platform: you might want to give more weights to the former.
- You can also provide sample weights (not class weights) for the validation set by adding them as a third item in the `validation_data` tuple.
- The `fit()` method returns a `History` object containing the training parameters (`history.params`), the list of epochs it went through (`history.epoch`), and most importantly a dictionary (`history.history`) containing the loss and extra metrics it measured at the end of each epoch on the training set and the validation set.
- You can see the plot of `history.history` in the learning notebook:
    - Both the training accuracy and the validation accuracy steadily increase during training, while the training loss and the validation loss decrease. This is good.
    - The validation curves are relatively close to each other at first, but they get further apart over time, showing a bit of overfitting.
    - In this particular case, the model looks like it performed better on the validation set than on the training set at the beginning of training, but that's actually not the case.
    - The validation error is computed at the *end* of each epoch, while the training error is computed *during* each epoch, so the validation curve should be shifted by half an epoch to the left.
    - If you do that, you will see that the training and validation curves overlap almost perfectly at the beginning of the training.
- The training set performance ends up beating the validation performance, as is generally the case when you train for long enough.
- You can tell that the model has not quite converged yet, as the validation loss is still going down, so you should probably continue training. This is as simple as calling the `fit()` method, since Keras will continue training where it left off.
- If you are not satisfied with your model's performance, you should go back and tune the hyperparameters:
    - The first one to check is the learning rate. If that doesn't help, try another optimizer (and always retune the learning rate after changing any hyperparameter).
    - If the performance is still not great, then try tuning the model hyperparameters such as the number of layers, the number of neurons per layer, and the type of activation functions to use for each hidden layer.
    - You can also try tuning other hyperparameters, such as the batch size (it can be set in the `fit()` method using the `batch_size` argument, which defaults to 32).
- Once you are satisfied with your model's validation accuracy, you should evaluate it on the test set to estimate the generalization error before you deploy your model to production.
- As discussed in chapter 2, it i common to get slightly worse performance on the test set than on the validation set, because the hyperparameters tuning is done on the validation set, not the test set. In our case, because we do not perform any hyperparameters tuning, the lower accuracy is mere bad luck.
- Remember to resist the temptation to tweak the hyperparameters on the test set, or else your estimate of the generalization error will be too optimistic.

### Using the model to make predictions

- We can use the model's `predict()` method to make predictions on new instances.
- In our case, for each instance, the model estimates one probability per class, from class 0 to class 9. This is similar to the output of `predict_proba()` method in Scikit-learn classifiers.
- If you only care about the class with the highest estimated probability (even if that probability is quite low), then you can use the `argmax()` method to get the highest probability class index for each instance.

## Building a Regression MLP Using the Sequential API

- We switch back to the California housing price problem in chapter 2 and tackle it using the same MLP as earlier, with 3 hidden layers composed of 50 neurons each, but this time building it with Keras.
- Using the sequential API to build, train, evaluate and use a regression MLP is quite similar to what we did for classification.
- The main differences are the output layer has a single neuron (since we only want to predict a single value) and it uses no activation function, the loss function is mean squared error, the metric is RMSE, and we're using an Adam optimizer like Scikit-learn's `MLPregressor` did.
- Moreover, in this example, we don't need a `Flatten` layer, and instead we use a `Normalization` layer as the first layer: it does the same thing as Scikit-learn's `StandardScaler`, but it must be fitted ot the training data using its `adapt()` method before you call the `fit()` method.
- The `Normalization` layer learns the feature means and standard deviations from the training data when you call its `adapt()` method. 
- Yet when you display the model's summary, these statistics are listed as non-trainable. This is because these these parameters are not affected by gradient descent.

## Building Complex Models Using the Functional API

- As you can see, the sequential API is quite clean and straightforward. However, although `Sequential` models are very common, it's sometimes useful to build neural networks with more complex topologies, or with multiple inputs or outputs. That's why Keras offers the functional API.
- One example of nonsequential neural networks is a *Wide & Deep* neural network. This neural network architecture was introduced in a 2016 paper by [Heng-Tze Cheng et al](https://arxiv.org/abs/1606.07792).
- This architecture connect all or part of the inputs directly to the output layer. This makes it possible for the neural network to learn both the deep patterns (using the deep path) and the simple rules (through the short path).
- This is in contrast to MLP, as a regular MLP forces all the data to flow through a full stacks of layers; thus, simple patterns in the dataset may end up being distorted by this sequence of transformations.
- We build a simple neural network with this architecture in the learning notebook. Here I go through this code for more detail:
    - First, we create five layers: a `Normalization` layer to standardize the inputs, two `Dense` layers with 30 neurons each, using the ReLU activation function, a `Concatenate` layer, and one more `Dense` layer with a single neuron for the output layer, without any activation function.
    - Next, we create an `Input` object (the name `input_` is used to avoid overshadowing the built-in `input()` function in Python). This is a specification of the kind of input the model will get, including its `shape` and optionally its `dtype`, which defaults to 32-bit floats. A model may actually have multiple inputs, as you will see shortly.
    - Then we use the `Normalization` layer just like a function, passing it the `Input` object. This is why this is called the functional API. Note that we are just telling Keras how it should connect the layers together; no actual data is being processed yet, as the `Input` object is just a data specification. In other words, it is a symbolic input. The output of this call is also symbolic: `normalized` does not store any actual data, it just need to construct the model.
    - In the same way, we then pass `normalized` to the `hidden_layer1`, which outputs `hidden1`, and we pass `hidden1` to `hidden_layer2`, which outputs `hidden2`.
    - So far, we have connected the layers sequentially, but then we use the `concat_layer` to concatenate the normalized input and the second hidden layer's output. Again, no actual data is concatenated yet: it's all symbolic, to build the model.
    - Then we pass `concat` to the `output_layer`, which gave us the final output.
    - Lastly, we create a `Keras Model`, specifying which inputs and outputs to use.
    - Note that, all we did above is all symbolic. In other words, this is just a blueprint of how the pipeline would be structured. It does not process any actual data yet.
- After creating this Keras model, everything else is the same as earlier: You compile the model, adapt the `Normalization` layer, fit the model, evaluate it and use it to make predictions.
- But what if you want to send a subset of features through the wide path and a different subset (possibly overlapping) through the deep path? The code in the learning notebook shows how we can do this in Keras.
- There are a few things to note in this example, compared to the previous one:
    - Each `Dense` layer is created and called on the same line. This is a common practice, as it makes the code more concise without losing clarity. However, we can't do this with the `Normalization` layer since we need a reference to the layer to be able to call its `adapt()` method before fitting the model.
    - We used `tf.keras.layers.concatenate()`, which creates a `Concatenate` layer and calls it with the given inputs.
    - We specified `inputs=[input_wide, input_deep]` when creating the model, since there are two inputs.
- Now we can compile the model as usual, but when we call the `fit()` method, instead of passing a single input matrix `X_train`, we must pass a pair of matrices `(X_train_wide, X_train_deep)`, one per input. The same for `X_valid` and also for `X_test` and `X_new` when you call `evaluate()` or `predict()`.
- Instead of passing a tuple `(X_train_wide, X_train_deep)`, you can pass a dictionary `{"input_wide": X_train_wide, "input_deep": X_train_deep}`, if you set `name="input_wide"` and `name="input_deep"` when creating the input layers. This is highly recommended when there are many inputs, to clarify the code and avoid getting the order wrong.
- There are also many use cases in which you may want to have multiple outputs:
    - The task may demand it. For instance, you may want to locate and classify the main object in a picture. This is both a regression task and a classification task.
    - Similarly, you may have multiple independent tasks based on the same data. Sure, you can train one neural networks per task, but in many cases, you will get better results on all tasks by training a single neural network with one output neuron per task. This is because the neural network can learn features in the data that are useful across tasks. For example, you could perform *multitask classification* on pictures of face, using one output to classify the person's facial expression (smiling, surprised, angry, etc.) and another output neuron to identify whether they are wearing glasses or not.
    - Another use case is as a regularization technique (i.e., a training constraint whose objective is to reduce overfitting and thus improve the model's ability to generalize). For example, you may want to add an auxiliary output in a neural network architecture to ensure the underlying part of the network learns something useful on its own, without relying on the rest of the network.
- Adding an extra output is quite easy: we just connect it to the appropriate layer and add it to the model's list of outputs.
- Each output will need its own loss function. Therefore, when we compile the model, we should pass a list of losses. If we pass a single loss, Keras will assume that the same loss must be used for all outputs.
- By default, Keras will compute all the losses and simply add them up to get the final loss used for training.
- Since we care much more about the main output than about the auxiliary output (as it is just used for regularization), we want to give the main output's loss a much greater weight. Luckily, it is possible to set all the loss weights when compiling the model.
- Instead of passing a tuple `loss=("mse", "mse")`, you can a dictionary `loss={"output": "mse", "aux_output": "mse"}`, assuming you created the output layers with `name="output"` and `name="aux_output"`. Just like for the inputs, this clarifies the code and avoid errors when there are several outputs. You can also pass a dictionary for `loss_weights`.
- Now when we train the model, we need to provide the labels for each output. In this example, the main output and the auxiliary output should try to predict the same thing, so they should use the same labels.
- So instead of passing `y_train`, we need to pass `(y_train, y_train)` or a dictionary `{"output": y-train, "aux_output": y_train}` if the outputs were named `"output"` and `"aux_output"`. The same goes for `y_valid` and `y_test`.
- When we evaluate the model, Keras returns the weighted sum of the losses, as well as all individual losses and metrics.
- If we set `return_dict=True`, then `evaluate()` will return a dictionary instead of a big tuple.
- Similarly, the `predict()` method will return predictions for each outputs.
- The `predict()` method returns a tuple, and it does not have a `return_dict` argument to get a dictionary instead. You can create one yourself using `model.output_names`.

## Using the Subclassing API to Build Dynamic Models

- Both the sequential API and functional API are declarative: you start by declaring which layers you want to use and how they should be connected, and only then you can start feeding the model some data for training or inference.
- This has many advantages:
    - The model can be easily saved, cloned and shared.
    - Its structure can be displayed and analyzed.
    - The framework can infer shapes and check types, so errors can be caught early (i.e., before any data ever goes through the model).
    - It's also fairly straightforward to debug, since the whole model is a static graph of layers.
- The flip side is also the fact that the model is static.
- Some models involve loops, varying shapes, conditional branching, and other dynamic behaviors.
- For such cases, or simply if you prefer a more imperative programming style, the subclassing API is for you.
- With this approach, you subclass the `Model` class, create the layers you need in the constructor, and use them to perform the computations you want in the `call()` method. 
- You must implement the `call()` method, as it is used to define the forward pass of your model, e.g., how to compute the outputs given the inputs.
- After having the model instance, we can compile it, adapt its normalization layers, fit it, evaluate it and use it to make predictions, exactly what we did when using the Functional API.
- The difference with this API is that you can include pretty much anything you want in the `call()` method: `for` loops, `if` statements, low-level TensorFlow operations, etc (we will do this in chapter 12).
- This makes it a great API when experimenting with new ideas, especially for researchers.
- However, this flexibility comes at a cost:
    - Your model's architecture is hidden within the `call()` method, so Keras cannot easily inspect it.
    - The model cannot be cloned using `tf.keras.models.clone_model()`.
    - When you call the `summary()` method, you only get a list of layers, without any information on how they are connected to each other.
    - Keras cannot check types and shapes ahead of time.
    - Is easier to make mistakes.
- Unless you really need tat extra flexibility, you better stick to the sequential API and the functional API.
- Keras models can be used as regular layers, so you can easily combine them to build complex architectures.

## Saving and Restoring a Model

- You simply save a model by calling its `save()` method.
- When you set `save_format="tf"`, Keras saves the model using TensorFlow's `SavedModel` format: this is a directory (with the given name) containing several files and subdirectories.
- In particular:
    - The `save_model.pb` file contains the model's architecture and logic in the form of a serialized computation graph, so you don't need to deploy the model's source code in order to use it in production; the SavedModel is sufficient, you will see how this works in chapter 12.
    - The `keras_metadata.pb` file contains extra information needed by Keras.
    - The `variables` subdirectory contains all the parameter values (including the connection weights, the biases, the normalization statistics, and the optimizer's parameters), possibly split across multiple files if the model is very large.
    - The `assets` directory may contain extra files, such as data samples, feature names, class names, and so on. By default, the `assets` directory is empty.
- Since the optimizer is also saved, including all its hyperparameters and any state it may have, after loading the model you can continue training if you want.
- If you set `save_format="h5"` or use a filename that ends with `.h5`, `.hdf5` or `.keras`, then Keras will save the model to a single file using a Keras-specific format based on the HDF5 format. However, most TensorFlow deployment tools require the SavedModel format instead.
- You will typically have a script that trains a model and saves it, and one or more scripts (or web services) that load the model and use it to evaluate and make predictions.
- You can load the model easily by using `tf.keras.models.load_model()` function.
- You can also use `save_weights()` and `load_weights()` to save and load only the parameter values. This includes the connection weights, biases, preprocessing stats, optimizer state, etc.
- The parameter values are saved in one or more files such as `my_weights.data-00000-of-00001`, plus an index file like `my_weights.index`.
- Saving just the weights is faster and use less disk space than saving the whole model, so it's perfect to save quick checkpoints during training.
- If you train a big model, and it take hours or days, then you must save checkpoints regularly in case the computer crashes.

## Using Callbacks

- The `fit()` method accepts a `callbacks` argument that lets you specify a list of objects that Keras will call before and after training, before and after each epoch, and even before and after processing each batch.
- For example, the `ModelCheckpoint` callback saves checkpoints of your model at regular intervals during training, by default at the end of each epoch.
- Moreover, if you use a validation set during training, you can set `save_best_only=True`. In this case, it will only save your model when its performance on the validation set is the best so far.
- This way, you do not need to worry about training for too long and overfitting the training set: simply restore the last saved model after training, and that will be the best model on the validation set. This is one way to implement early stopping (discussed in chapter 4), but it won't actually stop training.
- Another way is to use the `EarlyStopping` callback. It will interrupt training when it measures no progress on the validation set for a number of epochs (define by the `patience` argument), and if you set `restore_best_weights=True` it will roll back to the best model at the end of training.
- You can combines both callbacks to save checkpoints of your model is case your computer crashes, and interrupt training early when there is no more progress, to avoid wasting time and resources and to reduce overfitting.
- The number of epochs can be set to a large value since training will stop automatically when there is no more progress (just make sure the learning rate is not too small, or else it might keep making slow progress until the end). The `EarlyStopping` callback will store the weights of the best model in RAM, and it restore it for you at the end fo training.
- Many other callbacks are available in the [`tf.keras.callbacks` package](https://keras.io/api/callbacks/).
- If you need extra control, you can easily write your own custom callbacks.
- You can implement `on_train_begin()`, `on_train_end()`, `on_epoch_begin()`, `on_epoch_end()`, `on_batch_begin()` and `on_batch_end()`.
- Callbacks can be used during evaluation and predictions, if you ever need them (e.g., for debugging).
- For evaluation, you should implement `on_test_begin()`, `on_test_end()`, `on_test_batch_begin()` or `on_test_batch_end()`, which are called by `evaluate()`.
- For prediction, you should implement `on_predict_begin()`, `on_predict_end()`, `on_predict_batch_begin()` or `on_predict_batch_end()`, which are called by `predict()`.

## Using TensorBoard for Visualization

- TensorBoard is a great interactive visualization tool that you can use to view the learning curves during training, compare curves and metrics between multiple runs, visualize the computation graph, analyze training statistics, view images generated by your model, visualize complex multidimensional data projected down to 3D and automatically clustered for you, *profile* your network (i.e., measure its speed to identify bottlenecks), and more!
- To use TensorBoard, you must modify your program so that it outputs the data you want to visualize to special binary logfiles called *event files*. Each binary data record is called a *summary*.
- The TensorBoard server will monitor the log directory, and it will automatically pick up the changes and update the visualizations: this allows you to visualize live data (with a short delay), such as the learning curves during training.
- In general, you want to point the TensorBoard server to a root log directory and configure your program so that it writes to a different subdirectory every time it runs. This way, the same TensorBoard server instance will allow you to visualize and compare data form multiple runs fo your program, without getting everything mixed up together.
- Keras provides a convenient `TensorBoard()` callback that will take care of creating the log directory for you (along with its parent directories if needed), and it will create event files and writes summaries to them during training. 
- The callback will measure your model's training and validation loss (in this case, MSE and RMSE), and it will also profile the neural network.
- In our example, it will profile the network between batches 100 and 200 during the first epoch. Why 100 and 200? Well, if often takes a few batches for the neural network to "warm up", so you don't want to profile too early, and profiling used resources, so it's best ot to do it for every batch.
- There's one directory per run, each containing one subdirectory for training logs and one for validation logs. Both contain event files, and the training logs also include profiling traces.
- The server listens on the first available TCP port greater than or equal to 6006 (or you can set the port you want using `--port` option).
- You can run TensorBoard directly from the terminal by executing `tensorboard --logdir=./my_logs`. You must first activate the Conda environment in which you installed TensorBoard, and go to the desired directory. Once the server is started, visit [`http://localhost:6006`](http://localhost:6006).
- Now, let's have a look at the TensorBoard's user interface.
- We go to the SCALARS tab to see the learning curves:
    - At the bottom left, select the logs you want to visualize (e.e., the training logs from the first and second run). Notice that the training loss went down nicely in both runs, but in the second run, it's a low smaller thanks to higher learning rate.
    - You can visualize the whole computation graph in the GRAPHS tab.
    - You can visualize the learned weights projected to 3D in the PROJECTOR tab.
    - You can visualize the profiling traces in the PROFILE tab.
- The `TensorBoard()` callback has options to log extra data too (see [the documentation](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) for more details).
- You can click the refresh button (⟳) at the top right to make TensorBoard refresh data, and you can click the setting button (⚙) to activate auto-refresh and specify the refresh interval.
- Additionally, TensorFlow offers a lower-level API in the `tf.summary` package.
- We can create a `SummaryWriter` object using the `create_file_writer()` function. Then we can use this writer as a Python context to log scalars, histograms, images, audio and text, all of which can be visualized using TensorBoard.

# Fine-tuning Neural Network Hyperparameters

- The flexibility of neural networks is also one of their main drawbacks: there are many hyperparameters to tweak.
- Not only you can use any imaginable architecture, but even in a basic MLP, you can change the number of layers, the number of neurons and the type of activation function to use in each layer, the weight initialization logic, the type of optimizer to use, its learning rate, the batch size, and more.
- One way is to convert you Keras model to a Scikit-learn estimator, and then use `GridSearchCv` or `RandomizedSearchCV` to fine-tune the hyperparameters, as we did in Chapter 2.
- For this, you can use the `KerasRegressor` and `KerasClassifier` wrapper classes from the [SciKeras library](https://github.com/adriangb/scikeras).
- However, there's a better way: you can use the *Keras Tuner* library, which is a hyperparameter tuning library for Keras models. It offers several tuning strategies, it's highly customizable, and it has excellent integration with TensorBoard.
- Here is how to use this library:
    - Import `keras_tuner`, usually as `kt`.
    - Write a function that builds, compiles and returns a Keras model.
    - This function must take a `kt.Hyperparameters` object as an argument, which it can use to define hyperparameters (integers, floats, strings, etc.) along with their range of possible values, and these hyperparameters may be used to build and compile the model.
- We will walk through the creation of a function defined in the learning notebook:
    - The first part of the function defines the hyperparameters.
    - `hp.Int("n_hidden", min_value=0, max_value=8, default=2)` checks whether a hyperparameter named `"n_hidden"` is already present in the `Hyperparameters` object `hp`, and if so it returns its value. If not, then it registers a new integer hyperparameter named `"n_hidden"`, whose possible values range from 0 to 8 (inclusive) and it returns the default value, which is 2 in this case (when `default` is not set, it returns `min_value` instead).
    - The `"n_neurons"` hyperparameter is registered in a similar way.
    - The `"learning_rate"` hyperparameter is registered as a float ranging from $10^{-4}$ to $10^{-2}$, and since `sampling="log"`, learning rate will be sampled using a log distribution.
    - Lastly, the `optimizer` hyperparameter is registered with two possible values: `"sgd"` or `"adam"` (the default value is the first one, which is `"sgd"` in this case).
    - Depending on the value of `optimizer`, we create an `SGD` optimizer or an `Adam` optimizer with the given learning rate.
    - The second part of the function just builds the model using the hyperparameters values.
    - It creates a `Sequential` model starting with a `Flatten` layer, followed by the requested number of hidden layers (as determined by the `n_hidden` hyperparameter) using the ReLU activation function, and an output layer with 10 neurons (one per class) using the softmax activation function.
    - Lastly, the function complies the model and returns it.
- If you want to do a basic random search, you can create a `kt.RandomSearch` tuner, passing the `build_model` function to the constructor, and call the tuner's `search()` method.
- The `RandomSearch` tuner first calls `build_model()` once with an empty `Hyperparameters` object, just to gather all the hyperparameters specifications.
- In our example, it runs 5 trials; for each trail it builds a model using hyperparameters sampled randomly within their respective ranges, then it trains that model for 10 epochs and saves it to a subdirectory of the *`my_fashion_mnist/my_rnd_search`* directory.
- Since `overwrite=True`, the `*my_rnd_search`* directory is deleted before training starts. If you run this code a second time but with `overwrite=False` and `max_trials=10`, the tun will continue where it left off and run 5 more trials: this means you don't have to run all the trials in one shot.
- Lastly, since `objective="val_accuracy"`, the tuner prefers models with a higher validation accuracy.
- Each tuner is guided by a so-called *oracle*: before each trial, the tuner asks the oracle what the next trial should be.
- The `RandomSearch` tuner uses a `RandomSearchOracle`, which is pretty simple: it just picks the next trial randomly.
- If you are happy with your model's performance, you may continue training it for a few epochs on the full training set (`X_train_full` and `y_train_full`, in our example), then evaluate it on the test set, and deploy it to production.
- In some cases, you may want to fine-tune data preprocessing hyperparameters, or `model.fit()` arguments, such as the batch size. 
- For this, you must use a slightly different technique: instead of writing a `build_model()` function, you must subclass the `kt.HyperModel` class and define two methods, `build()` and `fit()`:
    - The `build()` method does the exact same things as the `build_model()` function.
    - The `fit()` method takes a `HyperParameters` object and a complied model as an argument, as well as all the `model.fit()` argument, and fits the model and returns the `History` object.
    - Crucially, the `fit()` method may use hyperparameters to decide how to preprocess data, tweak the bach size, and more.
- You can then pass an instance of this class to you desired tuner, instead of passing the `build_model()` function.
- The `kt.Hyperband` tuner is similar to the `HalvingRandomSearchCV` class we discussed in chapter 2:
    - It starts by training many different models for a few epochs
    - Then eliminates the worst models and keep only the top `1 / factor` models (i.e., the top third in our case)
    - Repeating this whole process until we're left with a single model.
    - The `max_epochs` argument controls the max number of epochs that the best model will be trained for.
    - The whole process is repeated twice in our case (`hyperband_iterations=2`).
    - The total number of training epochs across all models for each hyperband iteration is about `max_epochs * (log(max_epochs) / log(factor)) ** 2`, so it's about 44 epochs in our example.
    - The other arguments are the same as for `kt.RandomSearch`.
- If you open TensorBoard, pointing `--logdir` to the *`my_fashion_mnist/hyperband/tensorboard`* directory, you will see all the trial results as they unfold.
- You should visit the HPARAMS tab: it contains a summary of all the hyperparameters combination that were tried, along with the corresponding metrics.
- Notice that there are three tabs inside the HPARAMS tab: a table view,  a parallel coordinates view, and a scatterplot matrix view.
- In the lower part of the left panel, uncheck all metrics expect for `validation.epoch_accuracy`: this will make the graphs clearer.
- In the parallel coordinates view, you select a range of high values in the `validation.epoch_accuracy` column: this will filter only the hyperparameter combinations that reached a good performance.
- Hyperband is smarter than pure random search in the way it allocates resources, but at its core it still explores the hyperparameter space randomly; it's fast, but coarse.
- Keras Tuner also includes a `kt.BayesianOptimization` tuner: this algorithm gradually learns which regions of the hyperparameter space are most promising by fitting a probabilistic model called a *Gaussian process*. This allow it to gradually zoom in on the best hyperparameters.
- The downside is that the algorithm has its own hyperparameters: `alpha` represents the level of noise you expect in the performance measures across trails (it defaults to $10^{-4}$), and `beta` specifies how much you want the algorithm to explore, instead of simply exploiting the known good regions  of hyperparameter space (it defaults to 2.6).
- Other than that, this tuner can be used just like the previous one.
- Hyperparameter tuning is still an active area of research and many other approaches are being explored:
    - We can look at DeepMind's [2017 paper](https://arxiv.org/pdf/1711.09846.pdf), where the authors used an evolutionary algorithm ot jointly optimize a population of models and their hyperparameters.
    - Google has also used an evolutionary approach, not just to search for hyperparameters but also explore all sort of model architectures: it powers their AutoML service on Google Vertex AI.
    - The term *AutoML* refers to any system that takes care of a large part of the ML workflow.
    - Evolutionary algorithms have been used successfully to train individual neural networks, replacing the ubiquitous gradient descent!
    - For example, a [blog post](https://www.uber.com/en-VN/blog/deep-neuroevolution/) by Uber in 2017 where the authors introduce their *Deep Neuroevolution* technique.
- Despite all the exciting progress and all these tools and services, it stills helps to have an idea of what values are reasonable for each hyperparameter, so that you can build a quick prototype and restrict the search space.

## Number of Hidden Layers

- For many problems, you can start with a single hidden layer and get modest result.
- An MLP with a single hidden layer can theoretically model even complex functions, given it has sufficient neurons.
- But for complex problems, deep network have a much higher *parameter efficiency* than shallow ones: they can model complex functions using exponential fewer neurons than shallow networks, allowing them to reach much better performance with the same amount of training data.
- To understand why, assume you want to draw a forest using some softwares, but are not allowed to copy and paste anything. It would take an enormous amount of time: you would have to draw each tree independently, branch by branch, leaf by leaf.
- If you could instead draw a leaf, copy and paste it to draw a branch, then copy and paste that branch to create a tree, and finally copy and paste that tree to make a forest, you can finished in no time.
- Real-world data is often structured in such hierarchial way, and deep neural networks automatically take advantages of this fact: lower hidden layers model low-level structures (e.g., line segments of various shapes and orientations), intermedia hidden layers combine these low-level structures to model intermedia-level structures (e.g., squares, circles), and the highest hidden layers and the output layer combine these intermedia to model high-level structures (e.g., faces, digits).
- Not only does this hierarchial help DNNs converge faster to a good solution, but it also improves their capability to generalize to new datasets.
-For example, you have trained a model to recognize faces in pictures already and you now want to train anew neural network to recognize hair styles, you can kickstart the training by reusing the lower layers of the first neural network.
- Instead of randomly initializing the weights and biases of the first few layers of the new neural network, you can initialize them to the value of the weights and biases of the lower layer of the first neural network.
- This way, the network will not have learn from scratch all the low-level structures that occur in most pictures, it will only have learn the higher-level structures (e.g., hairstyles).
- This is called *transfer learning*.
- In summary, for many problems, you can start with juts one or two hidden layers and the neural network will work just fine.
- For instance, you can reach above 97% accuracy on the MNIST dataset using just one hidden layer with a few hundred neurons, and above 98% accuracy using two hidden layers with the same total number of neurons, in roughly the same amount of training time.
- For more complex problems, you can ramp up the number of hidden layers until you start overfitting the training set.
- Very complex tasks, such as large image classification or speech recognition, typically involve networks with dozens of layers (or even hundreds, but not fully connected ones, as we will see in chapter 14), and they need a huge amount of training data.
- You will rarely have to train such networks from scratch: it's much more common to reuse parts of a pretrained state-of-the-art network that performs a similar task. Training will then be a lot faster and require much less data.

## Number of Neurons per Hidden Layer

- The number of neurons in the input and output layers is determined by the type of input and output your task requires. For example, the MNIST task requires $28 \times 28 = 784$ inputs and output neurons.
- As for the hidden layers, it used to be common to size them to form a pyramid, with fewer and fewer neurons at each layer - the rationale being many low-level features can coalesce into far fewer high-level features. For example, a neural network for MNIST might have 3 hidden layers, the first with 300 neurons, the second with 200 and the third with 100.
- However, this practice has been largely abandoned because it seems that using the same number of neurons in all hidden layers performs just well, or even better, in most cases. Moreover, there is inly one hyperparameter to tune, instead of one per layer.
- That said, depending on the dataset, it can sometimes help to make the first hidden layer bigger than the others.
- Just like the number of layers, you can try increasing the number of neurons gradually until the neural network starts overfitting.
- Alternatively, you can try building a model with slightly more layers and neurons than you actually need, and use other regularization techniques to prevent it from overfitting too much.
- Vincent Vanhoucke, a scientist at Google, has dubbed this as the "stretch pants" approach: instead of wasting time looking for pants that perfectly match your size, just use large stretch pants that will shrink down to the right size.
- With this approach, you avoid bottleneck layers that could ruin your model.
- Indeed, if a layer has too few neurons, it will not have enough representational power to preserve all the useful information from the inputs (e.g., a layer with two neurons can only output 2D data, so if it gets 3D data as inputs, some information will be lost).
- No matter how big and powerful the rest of the network is, that lost information will never be recovered.
- In general, you will get more bang for the buck by increasing the number of hidden layers instead of the number of neurons per layers.

## Learning Rate, Batch Size, and Other Hyperparameters

- Here we discuss some of the most important hyperparameters you can tweak in a MLP, and tips on how to set them
- *Learning rate*: This is arguably the most important hyperparameter. 
- In general, the optimal learning rate is about half of the maximum learning rate (i.e., the learning rate above which the learning algorithm, as we saw in Chapter 4). 
- One way to find a good learning rate is to train the model for a few hundred iterations, starting with a low learning rate (e.g., $10^{-5}$) and gradually increasing it up to a very large value (e.g., 10).
- This is done by multiplying the learning rate by a constant factor at each iteration (e.g., by $(10 / 10^{-5})^{1/500}$ to go from $10^{-5}$ to 10 in 500 iterations).
- If you plot the loss as a function of the learning rate (using a log scale for the learning rate), you will see:
    - The loss will drop at first.
    - After a while, the learning rate will to be too large, so the loss will shoot back up.
    - The optimal learning rate will be a bit lower than the point at which the loss starts to climb (typically about 10 times smaller than the turning point).
    - You can then reinitialize your model and train it normally using this good learning rate.
- We will look at more learning rate optimization techniques later in chapter 11.
- *Optimizer*: Choosing a better optimizer than plain old mini-batch gradient descent is also quite important. We will look at several optimizers later in chapter 11.
- *Batch size*: The batch size can have significant impact on your model's performance and training time.
- The main benefit of using large batch sizes is that hardware accelerators like GPUs can process them efficiently, so the training algorithm can see more instances per second.
- Therefore, many researchers and practitioners recommend using the largest batch size that fits in the GPU RAM.
- There's a catch though: in practice, large batch size often lead to training instabilities, especially in the beginning of training, and the resulting model may not generalize as well as a model trained with a small batch size.
- In [a 2018 paper](https://arxiv.org/pdf/1804.07612.pdf) by Dominic Masters and Carlo Luschi concluded that using small batches (from 2 to 32) was preferable, as small batches led to better models in less training time.
- However, in 2017, papers by [Elad Hoffer et al.](https://arxiv.org/pdf/1705.08741.pdf) and [Priya Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf) showed that it was possible to use very large batch sizes (up to 8,192) along with various techniques such as warming up the learning rate (i.e., starting training with a small learning rate, then rampling it up, as will be discussed in chapter 11) and to obtain very short training times, without any generalization gap.
- So one strategy is to try to using large batch size, with learning rate warmup, and if training is unstable or the final performance is disappointing, then try to use a small batch size (2 to 32) instead.
- *Activation function*: We discussed this earlier: In general, the ReLU activation function will be a good default for all hidden layers, but for the output layer it really depends on the task.
- *Number of iterations*: In most cases, the number of training iterations does not actually need to be tweaked: just use early stopping instead.
- The optimal learning rate depends on other hyperparameters, especially the bach size, so if you modify any hyperparameter, make sure to update the learning rate as well.
- For more best practices regrading tuning hyperparameters, check out this [2018 paper](https://arxiv.org/pdf/1803.09820.pdf) by Leslie Smith.
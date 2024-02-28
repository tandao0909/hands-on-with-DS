- In the previous chapter, we built, trained, and fine-tuned our very first artificial neural networks.
- But they were shallow nets, with just a few hidden layers. What if we need to tackle a difficult problem  , such as detecting hundreds of types of objects in high-resolution images?
- Then you may need to train a much deeper ANN, perhaps with 10 layers or many more, each containing hundreds of neurons, linked by hundreds of thousands of connections.
- Training a deep neural networks is no easy task! Here'are some problems you can run into:
    - You may be faced with the problems of gradients growing ever smaller or larger, when flowing backward through the DNN during training. Both of these problems make lower layers very hard to train.
    - You might not have enough training data for such a large network, or it might be too costly to label.
    - Training may be extremely slow.
    - A model with millions of parameters would severely risk overfitting the training set, especially if there are not enough training instances or if they are too noisy.

# The Vanishing/Exploding Gradients Problems

- As we discussed in chapter 10, the backpropagation algorithm's second phase works by going from the output layer to the input layer, propagating the error gradient along the way.
- Once the algorithm has computed the gradient of the cost function with regard to each parameter in the network, it uses these gradients to update each parameter with a gradient descent step.
- Unfortunately, gradients often get smaller and smaller as the algorithm progresses down to the lower layers (intuitively, the parameters in the lower layers have more influence than those in the higher layers).
- As a result, the gradient descent update leaves the lower layers' connection weights virtually unchanged, and training never converges to a good solution. This is called the *vanishing gradients* problem.
- In some cases, the opposite can happen: the gradients can grow bigger and bigger until layers get insanely large weight updates and the algorithm diverges. This is the *exploding gradients* problem, which surfaces most often in recurrent neural networks (will be talked about in chapter 15).
- More generally, deep neural networks suffer from unstable gradients, different layers may learn at widely different speeds.
- This unfortunate behavior was empirically observed long ago, and it was one of the reasons deep neural networks were mostly abandoned in the early 2000s. 
- It wasn't clear what caused the gradients to be so unstable when training a DNN, but we have some insights in a [2010 paper](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) by Xavier Glorot and Yoshua Bengio.
- The authors found a few suspects, including the combination of the popular sigmoid (logistic) activation function and the weight initialization technique that was most popular at the time (i.e., a normal distribution with a mean of 0 and a standard deviation of 1).
- In short, they showed that this activation function and this initialization scheme, the variance of the outputs of each layer is much greater than the variance of its inputs. Going forward in the network, the variance keeps increasing after each layer until the activation function saturates at the top layers.
- This saturation is made worse by the fact that the sigmoid function has a mean of 0.5, not 0 (the hyperbolic tangent function has a mean of 0 and behaves slightly better than the sigmoid function in deep networks).
- Looking at the sigmoid activation function, we can see that when inputs become large (negative or positive), the function statures at 0 or 1, with a derivative extremely close to 0 (i.e., the curve is flat in both extremes).
- When propagation kicks in, it has virtually no gradient, to propagate back through the network, and what little gradients exists keeps getting diluted as propagation progresses through the top layers, so there is nothing left for the lower layers.

## Xavier and He Initialization

- In their paper, Glorot and Begio propose a way to significantly alleviate the unstable gradient problem.
- They point out that we need the signal to flow properly in both directions: in the forward direction when making predictions, and in the reverse direction when backpropagating gradients.
- We don't want the signal to die out, nor want it to explode and saturate.
- For the signal to flow properly, the authors argue that we need:
    - The variance of the outputs of each layer is equal to the variance of its inputs.
    - The analogy for the variance is as followed: If you set a microphone's amplifier's knob close to zero, then no one hear your voice, but if you set it close to the max, your voice will be saturated and people won't understand what you are saying. 
    - Now imagine a chain of such amplifiers: they all need to be set properly in order for your voice to come out loud and clear at the end of such chain. Your voice should be come out of each amplifier at the same amplitude as it came in.
    - The gradients has equal variance before and after flowing through a layer in the reverse direction.
- It is actually impossible to guarantee both unless the layer has equal number of inputs and outputs (which is why we have the same number of neurons in each layer). These numbers are called the *fan-in* and *fan-out* of the layer.
- However, Glorot and Begio proposed a good compromise that has proven to work very well in practice: the connection weights of each layer must be initialized randomly in one of two following ways:
    - Normal distribution with mean 0 and variance $\sigma^2=\displaystyle\frac{1}{fan_{avg}}$
    - Uniform distribution between -r and r, with $r=\displaystyle\sqrt{\frac{3}{fan_{avg}}}$

    where $fan_{avg}=(fan_{in} + fan_{out})/2$
- This initialization strategy is called *Xavier initialization* or *Glorot initialization*.
- If we replace $fan_{avg}$ with $fan_{in}$ in the above equation, you get an initialization strategy that Yann LeCun proposed in 1990s, which he called *LeCun initialization*.
- LeCun initialization is equivalent to Glorot initialization if $fan_{in} = fan_{out}$, (which is the case in the hidden layers, expect for the first and the last hidden layer).
- Using Glorot initialization can speed up training considerably, and it is one of the practices that led to the success of deep learning.
- [Some papers](https://arxiv.org/pdf/1502.01852.pdf) have provided similar strategies for different activation functions.
- These strategies differ only by the scale of the variance and whether they use $fan_{avg}$ or $fan_{in}$, (for the uniform distribution, just use $r=\sqrt{3 \sigma^2}$).

| Initialization | Activation functions                    | $\sigma^2$ (Normal) |
|-----------------|-----------------------------------------|---------------------|
| Glorot          | None, tanh, sigmoid, softmax            | $1/fan_{avg}$       |
| He              | ReLU, Leaky ReLU, ELU, GELU, Swish, Mish | $2/fan_{in}$       |
| LeCun           | SELU                                    | $1/fan_{in}$       |


- The initialization strategy proposed for the ReLU activation function and its variants is called *He initialization* or *Kaiming initialization*, after the paper's authors names.
- For SELU, use Yann LeCun's initialization method, preferably with a normal distribution.
- By default, Keras uses Glorot initialization with a uniform distribution.
- When you create a layer, you can switch to He initialization by setting `kernel_initializer="he_uniform"` or `kernel_initializer="he_normal"`.
- ALternatively, you can obtain any of the initialization listed in the above table and more using the `VarianceScaling` initializer.

## Better Activation Functions

- One of the insights in the 2010 paper by Glorot and Bengio was that the problems with unstable gradients were in part due to a poor choice of activation function.
- Back then, people thought that if biological neurons use roughly sigmoid activation functions, they must be a great choice.
- But it turns out other activation functions behave much better in deep neural network, in particular the ReLU activation function, as it does not saturate for positivize values, and also because it is very fast to compute.
- Unfortunately, ReLU activation function is not perfect. It suffers from a problem known as *dying ReLUs*: during training, some neurons effectively "die", meaning they stop outputting anything other than 0.
- In some cases, you may find that half of your network's neurons are dead, especially if you used a big learning rate.
- A neuron dies if its weights get tweaked in such a way that the input of the ReLU function (i.e., the weighted sum of the neurons's inputs plus its bias term) is negative for all instances in the training set.
- When this happens, it just keeps outputting zeros, and gradient descent does not affect it anymore because the gradient of the ReLU function is zero when its input is negative.

### Leaky ReLU

- The Leaky ReLU activation function is defined as:
$$\text{LeakyReLU}_{\alpha}(z)=\max(\alpha z, z)$$
- The hyperparameter $\alpha$ defines how much the function "leaks": it is the slope of the function for $z<0$.
- Having a slope for $z<0$ ensures that leaky ReLUs never die; they can go into a long coma, but they have a chance to eventually wake up.
- A [2015 paper](https://arxiv.org/pdf/1505.00853.pdf) by Bing Xu et al. compared several variants of the ReLU activation function, and here are some of its conclusions:
    - Leaky variants always outperformed the strict ReLU activation function.
    - In fact, setting $\alpha=0.2$ (a huge leak) seemed to result in better performance than $\alpha=0.01$ (a small leak).
    - The paper also evaluated the *randomized leaky ReLU* (RReLU), where $\alpha$ is picked randomly from a uniform distribution and is fixed to an average value during testing.
    - RReLU also performed fairly well and seemed to act as a regularizer, reducing the risk of overfitting the training set.
    - Finally, they evaluated the *parametric leaky ReLU* (PReLU), where $\alpha$ is authorized to be learn during training: instead of being a hyperparameter, it becomes a parameter that can be modified by backpropagation like any other parameter.
    - PReLU was reported to strongly outperform the ReLU on large datasets, but on smaller datasets, it has the risk of overfitting the training set.
- Keras includes the classes `LeakyReLU` and `PReLU` in the `tf.keras.layers` package.
- Just like other ReLU variants, you should use He initialization with these.
- If you prefer, you can use `LeakyReLU` as a separate layer in your model; it makes no difference for training and predictions.
- For PReLU, replace `LeakyReLU` with `PReLU`. There is currently no official implementation of PReLU in Keras, but you can fairly easily implement your own.
- ReLU, leaky ReLU and PReLU and suffer from the same problem that they are not smooth functions: their derivates abruptly change (at z = 0). 
- As we saw in chapter 4 when we discussed lasso, this sort of discontinuity can make gradient descent bounce around the optimum, and slow down converge.
- We only look at some smooth variants of the ReLU activation function, such as ELU and SELU.

### ELU and SELU

- In a [2015 paper](https://arxiv.org/pdf/1511.07289.pdf) by Djork-Arné Clevert et al. proposed a new activation function, called the *exponential linear unit* (ELU), that outperformed all the ReLU variants in the author's experiment: training time was reduced, and the performance on the test set was better:
$$\text{ELU}_\alpha(z)=\begin{cases}
\alpha  (\exp(z)-1) \text{ if } z < 0 \\
z \text{ if } z \geq 0 \\
\end{cases}$$
- The ELU activation function looks a lot like the ReLU activation function (see in the learning notebook), with a few major differences:
    - It takes on negative values when $z < 0$, which allows the unit to have an average closer to 0 and helps alleviate the vanishing gradients problem. 
    - The hyperparameter $\alpha$ defines the opposite of the value that the ELU function approaches when $z$ is a large negative number. It is usually set to 1, but you can tweak it like any other hyperparameter.
    - It has nonzero gradient for $z<0$, which avoids the dead neurons problem.
    - If $\alpha$ is equal to 1, then the function is smooth everywhere, including around $z=0$, which helps speed up gradient descent since it does not bounce as much to the left and right of $z=0$.
- Using ELU with Keras is as easy as setting `activation="relu"`, and like with other variants of ReLU, you should use the He initialization.
- The main drawback of ELU is that it is slower to compute than the ReLU and its variants (due to the use of the exponential function).
- It has faster convergence rate during training, which may compensate for that slow computation, but still, at test time ELU will be a bit slower than ReLU.
- Not long after, a [2017 paper](https://arxiv.org/pdf/1706.02515.pdf) by Günter Klambauer et al. introduced the *scaled ELU* (SELU) activation function.
- As the name suggests, it is a scaled variant of ELU activation function (about 1.5 times ELU, using $\alpha \approx 1.67$).
- The author showed that if you build a neural network composed exclusively of a stack of dense layers (i.e., an MLP), and if all hidden layers use the SELU activation function, then the network wil *self-normalize*: the output of each layer will tend to preserve a mean of 0 and a standard deviation of 1 during training, which solves the vanishing/exploding gradients problem. 
- As a result, the SELU activation function may outperform other activation function for MLPs, especially deep ones. To use it with Keras, just set `activation="selu"`.
- However, there are a few conditions for self-normalization to happen:
    - The input features must be standardized: mean 0 and standard deviation 0.
    - Every hidden layer's weights must be initialized using LeCun normal initialization. In Keras, this means setting `kernel_initializer="lecun_normal"`.
    - The self-normalizing property is only guaranteed with plain MLPs. If you try to use SELU in other architectures, like recurrent networks (will be discussed in chapter 15) or networks with *skip connections* (i.e., connections that skip layers, such as in Wide & Deep nets), it will probably not perform ELU.
    - You cannot use regularization techniques like $\ell_1$ or $\ell_2$ regularization, max-norm, batch-norm, or regular dropout (will be discussed later in this chapter).
- These are significant constraints, so despite its promise, SELU did not gain a lot of traction.
- Moreover, three more activation functions seem to outperform it quite consistently on most task: GELU, Swish and Mish.

### GELU, Swish and Mish

- GELU was introduced in a [2016 paper](https://arxiv.org/pdf/1606.08415.pdf) by Dan Hendrycks and Kevin Gimpel.
- We can, once again, think of GELU as a smooth variant of the ReLU activation function:
    $$\text{GELU}(z)=z\Phi(z)$$
    where $\Phi$ is the standard Gaussian cumulative distribution function (CDF): $\Phi(z)$ corresponds to the probability that a value sampled randomly from a normal distribution of mean 0 and variance 1 is less than $z$.
- As you can see in the learning notebook, GELU resembles ReLU: it approaches 0 when its input z is very negative, and it approaches z when z is very positive. 
- However, whereas all the activation functions we've discussed so far were both convex and monotonic, the GELU activation function is neither: from left to right, it starts by going straight, then it wiggles down, reaches a low point around -0.17 (near $z\approx -0.75$), and finally bounces up and ends up going straight toward the top right.
- This fairly complex shape and the fact that it has curvature at every point explain why it works so well, especially for complex tasks: gradient descent may find ti easier to fit complex patterns.
- In practice, it often outperforms every other activation function discussed so far.
- However, it is a bit more computationally intensive, and the performance boost it provides is not always sufficient to justify the extra cost.
- That said, it is possible to show that it is approximately equal to $z\sigma(1.702z)$, where $\sigma$ is the sigmoid function: using this approximation also works very well, and it has the advantage of being much faster to compute.
- A [2017 paper]() by Prajit Ramachandran et al. rediscovered the Swish activation function (early named SiLU and introduced in the GELU paper):
$$Swish(z) = z\sigma(z)$$
- In their paper, Swish outperformed every other activation functions, including GELU.
- They also generalized Swish by adding an extra hyperparameter $\beta$ to scale the sigmoid function's input:
    $$\text{Swish}_\beta=z\sigma(\beta z)$$
    so GELU is approximately equal to the generalized Swish function using $\beta=1.702$.
- You can tune $\beta$ as any other hyperparameter.
- Alternatively, you can make $\beta$ trainable and let gradient descent optimizes it: much like PReLU, this can make your model more powerful, but it also runs the risk of overfitting.
- Another quite similar activation function is *Mish*, which was introduced in a [2019 paper]() by Diganta Misra:
    $$\text{Mish}(z)=z\tanh(\text{softplus}(z))$$
- Just like GELU and Swish, it is a smooth, nonconvex, and monotonic variant of ReLU.
- The author, one again, ran many experiments and found that Mish generally outperformed other activation functions, even Swish and GELU, by a tiny margin.
- As you can see in the learning notebook, Mish overlaps almost perfectly with Swish when z is negative, and overlaps almost perfectly with GELU when z is positive.
- So in the end, which activation function should you choose for the hidden layers of your neural network?
    - ReLU remains a good default for simple tasks: it's often just as good as the more sophisticated activation functions, plus it's very fast to compute, and many libraries and hardware accelerators provide ReLU-specific optimizations.
    - However, Swish is probably a better default for more complex tasks, and you can even try parametrized Swish with a learnable $\beta$ for most complex tasks.
    - Mish may give you slightly better results, but it requires a bit more compute.
    - If you care about runtime latency, then you may prefer leaky ReLU, or parametrized leaky ReLU for more complex tasks.
    - For deep MLPs, give SELU a try, but make sure to follow the constrains listed earlier. 
    - If you have spare time and computing power, you can use cross-validation to evaluate other activation functions as well.
- Keras supports GELU adn Swish out of the box; just use `activation="gelu"` or `activation="swish"`.
- However, it doesn't support Mish or generalized activation function yet (but we can implement our own, instruction in chapter 12).

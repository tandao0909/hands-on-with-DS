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
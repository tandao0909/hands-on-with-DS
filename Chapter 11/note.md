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
- A [2017 paper](https://arxiv.org/pdf/1710.05941.pdf) by Prajit Ramachandran et al. rediscovered the Swish activation function (early named SiLU and introduced in the GELU paper):
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

# Batch Normalization

- Although using HE initialization along with ReLU (or any of its variants) can significantly reduce the danger of vanishing/exploding gradients problem at the beginning of training, it doesn't guarantee that they won't come back during training.
- In a [2015 paper](https://arxiv.org/pdf/1502.03167.pdf), Sergey Ioffe and Christian Szegedy proposed a technique called *batch normalization* (BN) that addresses these problems.
- The technique consists of adding an operation in the model just before or after the activation function of each hidden layer.
- This operation simply zero-centers and normalizes each input, then scales and shifts the result using two new parameter vectors per layer: one for scaling, the other for shifting.
- In other words, the operation lets the model learn the optimal scale and mean of each of the layer's inputs.
- Here, the inputs are the output features of the previous layer (may be activated or not).
- In many cases, if you add a BN layer as the very first layer of you training set, you do not need to standardize you training set. That is, there's no need for `StandardScaler` and `Normalization`; the Bn layer will do it for you.
- In order to zero-center and normalize the input, the algorithm needs to estimate each input's mean and standard deviation. It does so by evaluating the mean and standard deviation of the input over the current mini-batch, hence the name "batch normalization".
- The whole operation can be summarized in 4 equations:
    1. $\mu_B = \displaystyle\frac{1}{m_B}\sum\limits_{i=1}^{m_B}x^{(i)}$
    2. $\sigma_B^2 = \displaystyle\frac{1}{m_B}\sum\limits_{i=1}^{m_B}\left(x^{(i)}-\mu_B\right)^2$
    3. $\hat{x}^{(i)}=\displaystyle\frac{x^{(i)}-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$
    4. $z^{(i)} = \gamma \otimes \hat{x}^{(i)} + \beta$
- In this algorithm:
    - $\mu_B$ is the vector of input means, evaluated over the the whole mini-batch $B$ (it contains one mean per input).
    - $m_B$ is the number of instances in the mini-batch.
    - $\sigma_B$ is the vector of the input standard deviations, also evaluated over the whole mini-batch (it contains one standard deviation per input).
    - $\hat{x}^{(i)}$ is the vector of zero-centered and normalized inputs for instances i.
    - $\epsilon$ is a small number that avoid division by zero and ensures the gradients don't grow too large (typically $10^{-5}$). This is called a *smoothing term*.
    - $\gamma$ is the output scale parameter vector for the layer (it contains one scale parameter per input).
    - $\otimes$ represents element-wise multiplication (each input is multiplied by its corresponding output scale parameter).
    - $\beta$ is the output shift (offset) parameter vector for the layer (it contains one offset parameter per input). Each input is offset by its corresponding shift parameter.
    - $z^{(i)}$ is the output of the BN operation. It is a rescaled and shifted version of the inputs.
- So during training, BN standardize its inputs, then rescales and offsets them.
- What about at test time? That's not a simple task. We may need to make prediction for individual instances instead of batches of instances: in this case, we will have no way to compute each input's mean and standard deviation.
- Moreover, even if we do have a batch of instances, it may be too small, or the instances may not be independent and identically distributed, so computing statistics over the batch instances would be unreliable.
- One solution could be wait until the end of training, then run the whole training set through the network and compute the mean and standard deviation of each input of the BN layer. These "final" inputs means and standard deviation could then used instead of the batch means and standard deviation when making predictions.
- However, most implementations of batch normalization estimate these final statistics during training by using a moving average of the layer's input means and standard deviations. This is what Keras does automatically when you use the `BatchNormalization` layer.
- To usm up, four parameter vectors are learned in each batch-normalized layer: $\boldsymbol{\gamma}$ (the output scale vector) and $\boldsymbol{\beta}$ (the output offset vector) are learned through regular backpropagation, and $\boldsymbol{\mu}$ (the final input mean vector) and $\boldsymbol{\sigma}$ (the final input standard deviation vector) are estimated using an exponential moving average.
- Note that $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are estimated during training, but they are used only after training (to replace the batch input means and standard deviations in the third equation).
- Ioffe and Szegedy demonstrated that batch normalization considerably improved all the deep neural network they experimented with, leading to a huge improvement in the ImageNet classification task.
- The vanishing gradients problems was strongly reduced, to the point that they could use saturating activation function such as the tanh and eve the sigmoid activation function. The networks were also much less sensitive to the weight initialization.
- The authors were able to use much larger learning rate, significantly speeding up the learning process.
- Finally, batch normalization acts like a regularizer, reducing the need for other regularization techniques, (such as dropout, discussed later in this chapter).
- Batch normalization does, however, add some complexity to the model (although it can remove the need for normalizing the input data, as talked earlier). Moreover, there's a runtime penalty: the neural network makes slower predictions due to the extra computations at each layer.
- Fortunately, it's often possible to fuse the the BN layer with the previous layer after training, there by avoiding the runtime penalty. 
- This is done by updating the previous layer's weights and biases so that it directly produces outputs of the appropriate scale and offset.
- for example, if the previous layer computes $XW + b$, then the BN layer will compute $\gamma \otimes (XW + b - \mu) / \sigma + \beta$ (ignoring the smoothing term in the denominator).
- If we define $W' = \gamma \otimes X / \sigma$ and $b' = \gamma \otimes (b  - \mu) / \sigma + \beta$, the equation then simplifies to $XW' + b'$.
- So if we replace teh previous layer's weights and biases ($W$ and $b$) with the updated weights and biasses ($W'$ and $b'$), we can get rid of the BN layer (TFLite's converter does this automatically; see chapter 19).
- You may find training is rather slow because each epoch takes much more time when you use batch normalization. This is usually counterbalanced by the fact that convergence is much faster with BN, so it will take fewer epochs to reach the same performance. All in all, *wall time will usually be shorter* (this is the time measured by the clock on your wall).

### Implementing batch normalization with Keras

- As with most things in Keras, implementing batch normalization is straightforward and intuitive: just add  a `BatchNormalization` layer before or after each hidden layer's activation function.
- You may also add a BN layer as the first layer, but a plain`Normalization` layer generally performs just as well in this location, (its only drawback is that you must call its `adapt()` method).
- You can look at the model's summary: each BN layer adds four parameters per input: $\boldsymbol{\gamma}, \boldsymbol{\beta}, \boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ (for example, the first BN layer adds 3,136 parameters, which is $4 \times 784$).
- The last two parameters, $\mu$ and $\sigma$, are the moving averages; they are not affected by backpropagation, hence Keras called them "non-trainable" (if you count the total number of BN parameters, 3,136 + 1,200 + 400, and divide by 2, you get 2,368, which is the total number fo non-trainable parameters in this model).
- The authors of the BN paper argued in favor of adding the BN layers before the activation functions, rather than after (as we just did).
- There are some debates about this, as which is preferable seems to depend on the task - you can experiment with this to see which option work best on your dataset.
- To add the BN layers before the activation function, you must remove the activation functions from the hidden layers and add them as separate layers after the BN layers.
- Moreover, since a batch normalization layer includes one offset parameter per input, you can remove the bias term from the previous, you can remove the bias term from the previous layer by passing `use_bias=False` when creating it.
- Lastly, you can usually drop the first BN layer to avoid sandwiching the first hidden layer between two BN layers.
- The `BatchNormalization` class has quite a few hyperparameters you can tweak: The defaults will usually be fine, but you may occasionally need to tweak the `momentum`.
- This hyperparameter is used by the `BatchNormalization` layer when it updates the exponential moving averages; given a new value $v$ (i.e., a new vector of input means or standard deviations computed over the current batch), the layer updates the running average $\hat{v}$ using the following equation:
    $$\hat{v} \leftarrow \hat{v} \times \text{momentum} + v \times (1-momentum)$$
- A good momentum value is typically close to 1; for example, 0.9, 0.99, or 0.999. You want more 9s for larger datasets and for smaller mini-batches.
- Another important hyperparameter is `axis`: it determined which axis should be normalized.
- It defaults to -1, meaning it will normalize the last axis (using the means and standard deviations computes across the *other* axes).
- When the input batch is 2D (i.e., the batch shape is [*batch size, features*]), this means that each input feature will be normalized based on the mean and standard deviation computed across all the instances in the batch.
- For example, the first BN layer in our learning notebook will independently normalize (and rescale and shift) each of the 784 input features.
- If you move the first BN layer before the `Flatten` layer, then the input batches will be 3D, with shape [*batch size, height, width*]; therefore, the BN layer will compute 28 means and 28 standard deviations (1 per column of pixels, computed across all instances in the batch and across all rows on the column), and it will normalize all pixels in a given column using the same mean and standard deviation. If instead you want to treat each of 784 pixels independently, then you should set `axis=[1, 2]`.
- Batch normalization has become one the most-used layers in deep neural network, especially deep convolutional neural networks (discussed in chapter 14), to the point that it is omitted in the architecture diagrams: it is assumed that BN is added after every layer.

## Gradient Clipping

- Another technique to mitigate the exploding gradients problem is to clip the gradients during backpropagation so that they never exceed some threshold. This is called [*gradient clipping*](https://arxiv.org/pdf/1211.5063.pdf).
- This technique is generally used in recurrent neural networks, where using batch normalization is tricky (as you'll see in chapter 15).
- In Keras, implementing gradient clipping is just a matter of setting `clipvalue` or `clipnorm` argument when creating an optimizer.
- This optimizer will clip every component of the gradient vector to a value between -1.0 and 1.0.
- This means that all the partial derivates of the loss (with regard to each and every trainable parameter) will be clipped between -1.0 and 1.0.
- The threshold is a hyperparameter you can tune.
- Note that it may change the orientation of the gradient vector.
- For instance, if the original gradient vector is [0.9, 100.0], it points mostly in the direction of the second axis; but once you clip it by value, you get [0.9, 1.0], which points roughly at the diagonal between the two axes.
- In practice, this approach works well. 
- If you want to ensure that gradient clipping does not change the direction of the gradient vector, you should clip by norm by setting `clipnorm` instead of `clipvalue`. This will clip the whole gradient if its $\ell_2$ norm is greater than the threshold you picked.
- For example, if you set `clipnorm=1.0`, then the vector [0.9, 100.0] will be clipped to [0.00899964, 0.9999595], preserving its orientation but almost eliminating the first component.
- If you observe that the gradients explode during training (you can track the size of the gradients using TensorBoard), you may want to try clipping by value and clipping by norm, with different thresholds, and see which option performs best on the validation set.

## Reusing Pretrained Layers

- It is generally not a good idea to train a very large DNN from scratch without first trying to fnd an existing neural network that accomplishes a similar task to the one you are trying to tackle (we will talk about how to find them in chapter 14).
- If you can find such neural networks, then you can generally reuse most of its layers, expect for the top ones.
- This technique is called *transfer learning*.
- It will not only speed up training considerably, but also require significantly less training data.
- Suppose you have access to a DNN that was trained to classify pictures into 100 different categories, including animals, plants, vehicles, and everyday objects, and you now want to train a DNN to classify specific types of vehicles. These tasks are very similar, even partly overlapping, so you should try to reuse parts of the first network.
- If the input pictures for your new task don't have the same size as the one used in the original task, you will usually have to add a preprocessing step to resize them to the size expected by the original model.
- More generally, transfer learning will work best if the inputs have similar low-level features.
- The output layer of the original model should usually be replaced because it is most likely not useful at all for the new task, and probably will not have the right numbers of outputs.
- Similarly, the upper hidden layers of the original model are less likely to be as useful as the lower layers, since the high-level features that are most useful for the new task may significantly differ from the ones that were used in the original task. 
- You need to find the right number of layers to reuse.
- The more similar the tasks are, the more layers you will want to reuse (starting with the lower layers).
- For very similar tasks, you should try to keep all the hidden layers and just replace the output layer.
- Try freezing all the reused layers first (i.e., make their weights non-trainable so that gradient descent won't modify them and they will remain fixed), then train your model and see how it performs.
- Then try unfreezing one or two of the top frozen hidden layers to let backpropagation tweak them and see if performance improves.
- The more training data you have, the more layers you can unfreeze.
- It is also helpful to reduce learning rate after unfreezing pretrained layers: this will void wrecking teri fine-tuned weights.
- If you still cannot have good performance, and you have little training data, try dropping the top hidden layer(s) and freezing all the remaining hidden layers again.
- You can iterate until you satisfy with your model's performance (i.e., find the right number of layers to reuse).
- You have plenty of training data, you may try replacing the top hidden layers instead of dropping them, and even adding more hidden layers.

### Transfer Learning with Keras

- Let's have an example.
- Suppose the Fashion MNIST dataset only contains eight classes - for example, all the classes expect for the sandal and the shirt. Someone trained a Keras model on this set and got a reasonably good performance (>90% accuracy). Let's call this model A.
- You now want to tackle a different task: you have images of T-shirts and pullovers, and now you want to train a binary classifier: positive for T-shirts (and tops), negative for (sandals).
- Your dataset is quite small; you only have 200 labeled images.
- When you train a new model for this task (let's call it model B) with the same architecture as model A, you get 93.80% test accuracy. What if you use transfer learning?
- First, you need to load model A and create a new model based on that model's layers. You decide to reuse all the layers expect for the output layer.
- Note that in the learning notebook, at the earlier code cell, `model_A` and `model_B_on_A` share some layers. When you train `model_A`, it will also affect `model_B_on_A`.
- If you want to avoid this , you need to *clone* `model_A` before reusing its layers. To do this, you clone model A's architecture with `clone_model()`, then copy its weights.
- `tf.keras.models.clone_model()` only clones the architecture, not the weights. If you don't manually copy them by using `set_weights()`, they will be initialized randomly when the cloned model is first used.
- Now you could train `model_B_on_A` for task B, but since the new output layer was initialized randomly, it will make large errors (at least during the first few epochs), so there will be large error gradients that may wreck the reused weights.
- To avoid this, set every layer's `trainable` attribute to `False` and compile the model. 
- You must always compile you model after you freeze or unfreeze layers.
- Now you train the model for a few epochs, then unfreeze the reused layers (which requires compiling the model again) and continue training to fine-tune the reused layers for task B.
- After unfreezing the reused layers, it is usually a good idea to reduce the learning rate, once again to avoid damaging the reused weights.
- If you followed all these steps, you'll find a weird result: The performance drops!
- According to the author, he tried many configurations and chose one that demonstrated a strong improvement. This is called "torturing the data until it confesses".
- So here is his advice about reading papers: If a paper just looks too positive, be suspicious. Perhaps the flashy new technique does not actually help much (in fact, it may even degrade performance), but the authors tried many variants and reported only the bets results (which may be due to sheer luck), without mentioning how may failures they encountered along the path.
- Most of the time, this is not malicious at all, but it's part of the reason why so many results in science can never be reproduced.
- Back to our case study, it turns out that transfer learning does not work very well with small dense networks, presumably because small networks learn few patterns, and dense network lean very specific patterns, which are unlikely to be useful in other tasks.
- Transfer learning works best with deep convolutional neural networks, which tends to lean features detectors that are much more general, especially with lower layers.

## Unsupervised Pretraining

- Suppose you want to tackle a complex task for which you don’t have much labeled training data, but unfortunately you cannot find a model trained on a similar task.
- First, you should try to gather more labeled training data, but if you can’t, you may still be able to perform unsupervised pretraining.
- Indeed, it is often cheap to gather unlabeled training examples, but expensive to label them. 
- If you can gather plenty of unlabeled training data, you can try to use it to train an unsupervised model, such as an autoencoder or a generative adversarial network (GAN; see Chapter 17).
- Then you can reuse the lower layers of the autoencoder or the lower layers of the GAN’s discriminator, add the output layer for your task on top, and fine-tune the final network using supervised learning (i.e., with the labeled training examples).
- Until 2010, unsupervised pretraining — typically with restricted Boltzmann machines (RBMs; see [Wikipedia](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)) — was the norm for deep nets.
- Only after the vanishing gradients problem was alleviated did it become much more common to train DNNs purely using supervised learning.
- Unsupervised pretraining (today typically using autoencoders or GANs rather than RBMs) is still a good option when you have a complex task to solve, no similar model you can reuse, and little labeled training data but plenty of unlabeled training data.
- In the early days of deep learning it was difficult to train deep models, so people would use a technique called greedy *layer-wise pretraining*:
    - First train an unsupervised model with a single layer, typically an RBM.
    - Then freeze that layer and add another one on top of it.
    - Then train the model again (effectively just training the new layer).
    - Repeat the last two steps until you satisfy with the result.
- Nowadays, things are much simpler: people generally train the full unsupervised model in one shot and use autoencoders or GANs rather than RBMs.

## Pretraining on an Auxiliary Task

- If you do not have much labeled training data, one last option is to train a first neural network on an auxiliary task for which you can easily obtain or generate labeled training data, then reuse the lower layers of that network for your actual task.
- The first neural network's lower layers will learn feature detectors that will likely be reusable by the second neural network.
- For example, if you want to build a system to recognize faces, you may only have a few pictures of each individual - clearly not enough to train a good classifier.
- But gathering hundreds of pictures of each person would not be practical. Instead, you could gather a lot of pictures of random people on the web and train a first neural network to detect whether or not two different pictures features the same person.
- Such network would learn good feature detectors for faces, so reusing it lower layers would allow you to train a good face classifier that uses little training data.
- For natural language processing (NLP) applications, you can download a corpus of millions of text documents and auto generate labeled data from it.
- For example, you could randomly mask out some words and train a model to predict the missing words are (e.g., it should predict what the missing word in the sentence "What ___ you saying?" is probably "are" or "were").
- If you can train a model to reach good performance on this task, then it will already quite a lot about language, and you can certainly reuse it for your actual task and fine-tune it on your labeled data.
- *Self-supervised learning* is when you automatically generate the labels form the data itself, as in the text-masking example, then you train the model on the resulting "labeled" dataset using supervised learning technique.

# Faster Optimizers

- Training a very deep neural network can be painfully slow.
- So far, we have four ways of speed up training (and reach a better solution):
    - Using a good activation function
    - Applying a good initialization strategy for the connection weights.
    - Using batch normalization.
    - Reusing part of a pretrained network (possibly built for an auxiliary task or using unsupervised learning).
- Another huge boost comes from using a faster optimizer than the regular gradient descent optimizer.

## Momentum

- Recall the ball analogy from the gradient descent section: Imagining the ball is now rolling on a smooth surface. It will start out slowly, but quickly gain momentum until eventually reaches terminal velocity (if there's friction or air resistance).
- That's the core idea after *momentum optimization*, proposed by [Boris Polyak in 1964](https://www.researchgate.net/publication/243648538_Some_methods_of_speeding_up_the_convergence_of_iteration_methods).
- In contrast, regular gradient descent will take small steps when the slope is gentle and big steps when the slope is steep, but it will never pick up speed.
- As a result, regular gradient descent is generally much slower to reach the minimum than momentum optimization.
- Recall that regular gradient descent updates the weight $\theta by directly subtracting the gradient of th cost function $J(\theta)$ with regard to the weights $(\nabla_\theta J(\theta))$ multiplied by the learning rate $\eta$:
    $$\theta \leftarrow \theta - \eta \nabla_\theta J(\theta)$$
- It does not care about what the earlier gradients were. If the local gradient is tiny, then it goes slowly.
- Momentum optimization cares a great deal about what previous gradients were: at each iteration, it subtracts the local gradient from the *momentum vector* **m**  (multiplied by the learning rate $\eta$), and it updates by adding this momentum vector:
    1. $\textbf{m} \leftarrow \beta \textbf{m} - \eta \nabla_\theta J(\theta)$
    2. $\theta \leftarrow \theta + \textbf{m}$
- In other words, the gradient is used as an acceleration, not as a speed.
- To stimulate some sort of friction mechanism and prevent the momentum from growing too large, the algorithm introduces a new hyperparameter $\beta$, called the *momentum*, which must be set between 0 (high friction) and 1 (no friction). A typical momentum value is 0.9.
- You can do the math to verify that if the gradient remains constant, the terminal velocity (i.e., the maximum size of the weight updates) is equal to that gradient multiplied by the learning rate $\eta$ multiplied by $1 / (1 - \beta)$ (ignoring the sign).
- For example, if $\beta=0.9$, then the terminal velocity is equal to 10 times the learning rate times the gradients, so momentum gradient ends up 10 times faster than gradient descent in this case.
- This allows momentum optimization to space from the plateau much faster than regular gradient descent.
- We saw in chapter 4 that when the inputs have very different scales, the cost function will look like an elongated bowl.
- Gradient descent goes down the steep slope quite fast, but then it takes a long time to travel down the valley. 
- In contrast, momentum optimization will roll down the valley faster and faster until it reaches the the bottom (the optimum).
- In deep neural networks that don't use batch normalization, the upper layers will often having inputs with very different scales, so using momentum optimization helps a lot.
- In can also help roll past the local minima.
- Due to the momentum, the optimizer may overshoot a bit, then come back, overshoot again, and oscillate like this many times before stabilizing at the minimum. This is one of the reasons it's good to have a bit of friction on the system: it get rids of these oscillations and thus speeds up convergence.
- Implementing momentum optimization in Keras is simple: just use the `SGD` optimizer and specify its `momentum` hyperparameter.
- The one drawback of momentum optimization is that it adds yet another hyperparameter to tune. However, the default value of 0.9 is quite good in practice and almost always goes faster than regular gradient descent.

## Nesterov Accelerated Gradient

- One small variant to momentum optimization is proposed by [Yurii Nesterov in 1983](https://scholar.google.com/scholar?q=A+method+for+solving+the+convex+programming+problem+with+convergence+rate+author%3Anesterov), is almost always faster than the regular momentum optimization.
- The *Nesterov acceleration gradient* (NAG) method, also known as *Nesterov momentum optimization*, measures the gradient of the cost function not at the local position $\theta$ but slightly ahead in the direction of the momentum, at $\theta + \beta \textbf{m}$:
    1. $\textbf{m} \leftarrow \beta \textbf{m} - \eta \nabla_\theta J(\theta + \beta \textbf{m})$
    2. $\theta \leftarrow \theta + \textbf{m}$
- This small tweak works because in general, the momentum vector will be pointing in the right direction (i.e., toward the optimum), so it will be slightly more accurate to use the gradient measured a bit farther in that direction rather than the gradient at the original position.
- As you can see in the book, the Nesterov update ends up closer to the minimum. After a while, these small improvements add up and NAG neds up being significantly faster than regular momentum optimization.
- Moreover, note that when the momentum pushed the weights across a valley, $\nabla_1$ continues to push the weights across the valley, while $\nabla_2$ pushes back toward the bottom of the valley (where $\nabla_1$ represents the gradient of the cost function measured at the starting point $\theta$, and $\nabla_2$ represents the gradient at the point located at $\theta+ \beta \textbf{m}$).
- This helps reduce oscillations and thus NAG converges faster.

## AdaGrad

- Consider the elongated bowl again: gradient descent starts by quickly going the steepest slope, which does not point straight toward the global optimum, then it very slowly goes down to the bottom of the valley.
- It would be nice if the algorithm can correct its direction earlier to point a bit more toward the global optimum.

- The [AdaGrad algorithm](https://jmlr.org/papers/v12/duchi11a.html) achieves this correction by scaling down the gradient vector along the steepest dimensions:
    1. $s \leftarrow s + \nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta)$
    2. $\theta \leftarrow \theta - \eta \nabla_\theta J(\theta) \oslash \sqrt{s + \varepsilon}$
- The first step accumulates the square of the gradients into the vector $s$ (recall that the $\otimes$ represents the element-wise multiplication).
- This vectorized form is equivalent to computing $s_i \leftarrow s_i + (\partial J(\theta) / \partial \theta_i)^2$ for each element $s_i$ of the vector $s$. In other words, each $s_i$ accumulates the squares of the partial derivate of the cost function, with regard to parameter $\theta_i$.
- If the cost function is steep along the $i^{th}$ dimension, then $s_i$ will get larger and larger at each iteration.
- The second step is almost identical to gradient descent, but with one important difference: the gradient vector is scaled down by a factor of $\sqrt{s + \varepsilon}$ (the $\oslash$ symbol represents the element-wise division, and $\varepsilon$ is a smoothing term to avoid division by zero, typically set to $10^{-10}$).
- This vectorized form is equivalently to simultaneously computing 
    $$\theta_i \leftarrow \theta_i - \eta \frac{\partial J(\theta)}{\partial \theta_i}.\frac{1}{\sqrt{s_i + \varepsilon}}$$
     for all parameters $\theta_i$.
- In short, this algorithm decays the learning rate, but it does so faster for steep dimensions than for dimensions with gentler slopes. This is called an *adaptive learning rate*.
- It helps pointing the resulting updates more directly toward the global optimum.
- One additional benefit is that it requires much less tuning of the learning rate hyperparameter $\eta$.
- AdaGrad frequently performs well for simple quadratic problems, but it often stops too early when training neural networks: the learning rate gets scaled so much that the algorithm ends up stopping entirely before reaching the optimal optimum.
- **Note**: Even though Keras does have an `AdaGrad` optimizer, you should not use it to train deep neural networks (it may be efficient for simpler tasks, such as linear regression, though).
- However, understanding AdaGrad is helpful to comprehend the other adaptative learning rate algorithms.

## RMSProp

- As we've seen, AdaGrad runs the risk of slowing down a bit too fast and never converging to the global optimum.
- The *RMSProp* algorithm was created by Geoffrey Hinton and Tijmen Tieleman in 2012 and presented by Geoffrey Hinton in his Coursera class on neural networks ([slides](https://homl.info/57); [video](https://homl.info/58)). Amusingly, since the authors did not write a paper to describe the algorithm, researchers often cite “slide29 in lecture 6e” in their papers.
- It fixes this by accumulating only the gradients from the most recent iterations, as opposed to all the gradients since the beginning of training.
- It does so by using exponential decay in the first step:
    1. $s \leftarrow \rho s + (1-\rho)\nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta)$
    2. $\theta \leftarrow \theta - \eta \nabla_\theta J(\theta) \oslash \sqrt{s + \varepsilon}$
- The decay rate $\rho$ is typically set to 0.9. Yes, it is once again a new hyperparameter, but this default value often works well, so you may not need to tune it at all.
- As you expect, Keras has an `RMSProp` optimizer.
- Expect on very simple problems, this optimizer almost always performs much better than AdaGrad. 
- In fact, it was the preferred optimization algorithm of many researchers until Adam optimization showed up.

## Adam

- [Adam](https://arxiv.org/pdf/1412.6980.pdf), which stands for *adaptive moment estimation*, combines the ideas of momentum optimization and RMSProp:
    - Just like momentum optimization, it keeps track of an exponentially decaying average of past gradients.
    - Just like RMSProp, it keeps track of an exponentially decaying average of past squared gradients.
- These estimations of the mean and (unentered) variance of the gradients.
- The mean is often called the *first moment*, while the variance is often called the *second moment*, hence the name of the algorithm.
- Here are the equations describe the process of Adam:
    1. $m \leftarrow \beta_1 m - (1-\beta_1)\nabla_\theta J(\theta)$
    2. $s \leftarrow \beta_2 s + (1-\beta_2)\nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta)$
    3. $\hat{m} \leftarrow \displaystyle\frac{m}{1-\beta_1^t}$
    4. $\hat{s} \leftarrow \displaystyle\frac{s}{1-\beta_2^t}$
    5. $\theta \leftarrow \theta + \eta \hat{m} \oslash \sqrt{\hat{s} + \varepsilon}$
- In this equation, $t$ represents the iteration number (starting at 1).
- If you just look at steps 1, 2, and 5, you will notice Adam's close similarity to both momentum optimization and RMSProp: $\beta_1$ corresponds to $\beta$ in momentum optimization, and $\beta_2$ corresponds to $\rho$ in RMSProp.
- The only difference is that step 1 computes an exponentially decaying average rather than an exponentially decaying sum, but these are actually equivalent expect for a constant factor (the decaying average is just $1-\beta_1$ times the decaying sum).
- Steps 3 and 4 are somewhat of a technical detail: 
    - Since $m$ and $s$ are initialized at 0, they will be biased toward 0 at the beginning of the training.
    - But we want they to be more aggressive at the start of the training, where the optimization is trivial and should be done fast.
    - Since $\beta_1$ $\beta_2$ are both closer than 1, $m$ and $s$ will be multiplied multiple times, which help boosting them at the beginning of training. 
    - After some iterations, since $\beta_1$ $\beta_2$ are both smaller than 1, they will converge to 0, leaves the $m$ and $s$ nearly unchanged.
- The momentum decay hyperparameter $\beta_1$ is typically initialized to 0.9, while the scaling decay hyperparameter is often initialized to 0.999. As earlier, the smoothing term $\varepsilon$ is usually initialized to a tiny number such as $10^{-7}$. These are default values for the $Adam$ class.
- Since Adam is an adaptive learning rate algorithm, like AdaGrad and RMSProp, it requires less tuning of the learning rate hyperparameter. You can often use the default value $\eta = 0.001$, making Adam even easier to use than gradient descent.

### AdaMax

- The Adam paper also introduced AdaMax.
- Notice that step 2 of the above process, Adam accumulates the squares of the gradients in $s$ (with a greater weight for more recent gradients).
- In step 5, if we ignore $\varepsilon$ and steps 3 and 4 (which are technical detail anyway), Adam scales down the parameter by the square root of $s$.
- In short, Adam scales down the parameter updates by the $\ell_2$ norm of the time-decayed gradients (recall that the $\ell_2$ is the square root of the sum of squares).
- AdaMax replaces the $\ell_2$ norm with the $\ell_\infty$ norm (a fancy way of saying the max norm).
- Specifically, it replaces step 2 with:
    $$s \leftarrow \max(\beta_2 s, |\nabla_\theta J(\theta)|)$$
    drops step 4, and in step 5 it scales down the gradients updates by a factor of $s$, which is the max of the absolute value of the time-decayed gradients.
- In practice, this can make AdaMax more stable than Adam, but it really depends on the dataset, and in general Adam performs better.

### Nadam

- Nadam optimization is Adam optimization plus the Nesterov trick, so it will often converge slightly faster than Adam. 
- [In his report introducing this technique](https://cs229.stanford.edu/proj2015/054_report.pdf), the researcher Timothy Dozat compares many different optimizers on various tasks and finds that Nadam generally outperforms Adam but is sometimes outperformed by RMSProp.

### AdamW 

- [AdamW]() is a variant of Adam that integrates a regularization technique called *weight decay*.
- Weigh decay reduces the size of the model's weights at each training iteration by multiplying them by a decay factor, such as 0.99.
- This may remind you of $\ell_2$ regularization (introduced in chapter 4),which also aims to keep the weights small. In fact, it can be shown mathematically that $\ell_2$ regularization is equivalent to weight decay when using SGD.
- However, when using Adam or its variants, $\ell_2$ regularization and weight decay are *not* equivalent: in practice, combining Adam with $\ell_2$ regularization results in models that often don't generalize as well as those produced by SGD.
- AdamW fixes this issue by properly combining Adam with weight decay.
- Adaptive optimization methods (including RMSProp, Adam, AdaMax, Nadam, and AdamW optimization) are often great, converging fast to a good solution.
- However, [a 2017 paper](https://arxiv.org/abs/1705.08292) by Ashia C. Wilson et al. showed that they can lead to solutions that generalize poorly on some datasets.
- So when are disappointed by your model's performance, try using NAG instead: your dataset may just not fitted for adaptive gradients.
- Also check out the latest research, because it's moving fast.
- All the optimization techniques discussed so far only rely on the *first-order partial derivates (Jacobians)*.
- The optimization literature also contains amazing algorithms based on the *second-order partial derivates* (the *Hessian*, which are the partial derivates of the Jacobians).
- Unfortunately, these algorithms are very hard to apply to deep neural networks because there are $n^2$ Hessian per output (where n is the number fo parameters), as opposed to just n Jacobians per output.
- Since DNNs typically have tens of thousands of parameters or more, the second-order optimization algorithms often don't even fit in the memory, and even when they do, computing the Hessians matrix is just too slow.
- The following table compares all the optimizers we discussed so far:

| Class                              | Convergence speed | Convergence Quality |
|------------------------------------|------------------|--------------------|
| `SGD`                              | *                | ***                |
| `SGD(momentum=...)`                | **               | ***                |
| `SGD(momentum=..., nesterov=True)` | **               | ***                |
| `Adagrad`                          | ***              | *(stops too early) |
| `RMSProp`                            | ***              | ** or ***          |
| `Adam`                               | ***              | ** or ***          |
| `AdaMax`                             | ***              | ** or ***          |
| `Nadam`                              | ***              | ** or ***          |
| `AdamW`                              | ***              | ** or ***          |

## Training Sparse Models

- All the optimization algorithms we just discussed produce dense models, meaning that most parameters wil be nonzero. If you need a blazingly fast model at runtime, or if you need it to take up less memory, you may prefer to end up with a sparse model instead.
- One way to achieve this is to train the model as usual, then get rid of the tiny weights (set them to zero).
- However, this will typically not lead to a very sparse model, and it may degrade the model's performance.
- A better option is to apply strong $\ell_1$ regularization during training (you'll se how later in this chapter), as it pushes the optimizer to zero out as many weights as possible (as discussed in Lasso Regression in chapter 4).
- If these techniques remains insufficient, check out the [TensorFlow Model Optimization Toolkit (TF_MOT)](https://www.tensorflow.org/model_optimization/), which provides a pruning API capable of iteratively removing connections during training based on their magnitude.

## Learning Rate Scheduling

- Finding a good learning rate is important:
    - If you set it too much high, training may diverge (as discussed in Gradient Descent chapter 4).
    - If you set it too low, training will eventually converge to the optimum, but it will take a very long time.
    - If you set it slightly too high, it will make progress very quickly at first, but it will end up dancing around the optimum and never really stelling down. If you have limited computing budget, you may have to interrupt training before it has converged properly, yielding a suboptimal solution.
- As discussed in chapter 10, you can find a good learning rate by training the model for a few hundred iterations, exponentially increasing the learning rate from a very small value to a very large value, and then looking at the learning curve and picking a learning rate slightly lower than the one at which the learning curve starts shooting back up. You can then reinitialize your model and train it with that learning rate.
- But you can do better than a constant learning rate: If you start with a large learning rate and then reduce it once training stops making fast progress, you can reach a good solution faster than the optimal learning rate.
- There are many different strategies to reduce the learning rate during training.
- It can also be beneficial to start with a low learning rate, increase it, then drop it again.
- These strategies are called *learning schedules* (which were briefly introduced in chapter 4).
- Here are we list some commonly used learning schedules.

### Power scheduling
- Set the learning rate to a function of the iteration number *t*: 
    $$\eta(t) = \eta_0 / (1 + t / s)^c$$
- The initial learning rate $\eta_0$, the power c (typically set to 1), and the steps s are hyperparameters.
- The learning rate drops at each step.
- After s steps, the learning rate is down to $\eta_0 / 2$. After s more steps, it is down to $\eta_0/ 3$, then goes down to $\eta_0 / 4$, then $\eta_0 / 5$, and so on.
- As you can see, this schedule first drops quickly, then ore and more slowly.
- Of course, power scheduling requires tuning $\eta_0$ and $s$, and possibly $c$.
- In Keras, you should use the `InverseTimeDecay` scheduler to implement power scheduling. The way to use it can be seen in the learning notebook.

### Exponential Scheduling

- Set the learning rate to:
    $$\eta(t) = \eta_0 0.1^{t/s}$$
- The learning rate will gradually drops by a factor of 10 every s steps.
- While power scheduling reduces the learning rate more and more slowly, exponential scheduling keeps slashing it by a factor of 10 every s steps.
- In Keras, similar to power scheduling, you can use the `ExponentialDecay` scheduler to implement exponential scheduling.
- This will update the optimizer's learning rate at the beginning of each epoch.
-If you want to update the learning rate per step, then you need to define a function that takes the current epoch and returns the learning rate. 
- Then you need to create a `LearningRateScheduler` callback, giving it the schedule function, and pass this callback to the `fit()` method. 
- After training, `history.history["lr"]` give you access to the list of learning rates used during training.
- The schedule function can optionally take the current learning rate as a second argument.
- Then the function will rely on the optimizer's initial learning rate (contrary to the previous implementation), hence you need to set it properly.
- When you save a model, the optimizer and its learning rate get saved along with it. This means that with this new schedule function, you could just load a trained model and continue training where it left off.
- However, things are not so simple if you use the `epoch` argument: the epoch does not get saved, and it resets to 0 every time you call the `fit()` method.
- If you were to continue training a model where it left off, this could lead to a very large learning rate,w which would likely destroy your model's weights.
- One solution is to manually set the `fit()` method's `initial_epoch` argument so that the `epoch` starts at the right value.

### Piecewise constant scheduling

- Use a constant learning rate for a number of epochs (e.g., $\eta_0 = 0.1$ for 5 epochs), then a smaller learning rate for another number of epochs (e.g., $\eta_1 = 0.001$ for 50 epochs), and so on.
- Although this solution can work really well, it requires fiddling around to find the optimal sequence of learning rates and how long to use each of them.
- Similar to exponential scheduling, you can use a schedule function, then creates a `LearningRateScheduler` callback with this function and pass it to the `fit()` method.

### Performance Scheduling

- Measure the validation error every N steps (just like for early stopping), and reduce the learning rate by a factor if $\gamma$ when the error stops dropping.
- In Keras, you use the `ReduceLROnPlateau` callback.
- For example, if you pass the following callback to the `fit()` method, it will multiply the learning rate by 0.5 whenever the best validation loss does not improve for five consecutive epochs:
`lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)`
- Alternatively, Keras offers another way to implement learning rate scheduling: You can defined a scheduled learning rate using one of the classes available in `tf.keras.optimizers.schedules`, then pass it to the `learning_rate` argument of any optimizer.
- This approach updates the learning rate at each step instead of each epoch.

### 1cycle Scheduling

- 1cycle was introduced in a [2018 paper](https://arxiv.org/abs/1803.09820) by Leslie Smith.
- Contrary to the other approaches, it starts by increasing the initial learning rate $\eta_0$, growing linearly up to $\eta_1$ halfway through training.
- Then it decreases the learning rate linearly down to $\eta_0$ again during the second half of training, finishing the last few epochs by dropping the rate down by several orders of magnitude (still linearly).
- The maximum learning rate $\eta_1$ is chosen using the same approach we used to find the optimal learning rate, and the initial learning rate $\eta_0$ is usually 10 times lower.
- When using a momentum, we start with a high momentum first (e.g., 0.95), then drop it down to a lower momentum during the first half of training (e.g., down to 0.85, linearly), then bring it back up to the maximum value (e.g., 0.95) during the second half of training, finishing the last few epochs with that maximum value.
- Smith did many experiments showing that this approach was often able to speed up training considerably and reach better performance.
- A [2013 paper]() by Andrew Senior et al. compared the performance of some of the most popular learning schedules when using momentum optimization to train deep neural network for speech recognition.
- The authors concluded that:
    - In this setting, both performance scheduling and exponential scheduling performed well. 
    - They favored exponential scheduling because it was easy to tune and it converged slightly faster to the optimal solution.
    - That said, the 1cycle approach seems to performs even better.
- As for 1cycle, Keras does not support it, but you can implement it yourself by creating a callback that modifies the learning rate at each iteration. See the learning notebook for an example.

# Avoiding Overfitting Through Regularization

- DNNs typically have tens of thousands of parameters, sometimes even millions. 
- This give them an incredible amount of freedom and means that they can fit a huge a variety of complex datasets.
- But this great flexibility comes at a cost: the networks is so prone to overfitting the training set.
- Regularization is often needed to prevent this.
- We already implemented one of the best regularization techniques in chapter 10: Early stopping.
- Moreover, even though designed to solve the unstable gradients problems, batch normalization also acts like a pretty good regularizer.

## $\ell_1$ and $\ell_2$ Regularization

- Just like we did for in chapter 4 for simple linear models, you can use $\ell_2$ regularization to constrain a neural network's connection weights, and/or $\ell_1$ regularization if you want a sparse model (with many weights equal to 0).
- You can pass `tf.keras.regularizers.l2()` function to the `kernel_regularizer` argument of a layer to apply $\ell_2$ regularization.
- The `l2()` function returns a regularizer that will be called at each step during training to compute the regularization loss. This is then added to the total loss.
- If you want to use $\ell_1$, use the `tf.keras.regularizers.l1()` function; if you want both, then use `tf.keras.regularizers.l1_l2()` (which specifying both regularization factors).
- Since you will typically want to apply the same regularizer to all layers in your network, as well as using the same activation function and the same initialization strategy in all hidden layers, you may find yourself repeat the same arguments over and over again. This makes the code ugly and error-prone.
- To avoid this, you can refactor using loops.
- Another option is to use Python's `functools.partial()` function, which let you create a thin wrapper for any callable, with some default argument values.
- As we saw earlier, $\ell_2$ regularization is fine when using SGD, momentum optimization, and Nesterov momentum optimization, but not with Adam and its variants.
- If you want to use Adam with weight decay, use AdamW instead.

## Dropout

- *Dropout* is one of the most popular regularization techniques for deep neural networks.
- It was proposed in a [2012 paper](https://arxiv.org/abs/1207.0580) by Geoffrey Hinton et al. and further detailed in a [2014 paper](https://jmlr.org/papers/v15/srivastava14a.html) by Nitish Srivastava et al.
- It has proven to be highly successful: many state-of-the-art neural networks use dropout, as it gives them a 1%-2% accuracy boost.
- This may not sound like a lot, but when a model already has 95% accuracy, getting a 2% accuracy boost means dropping the error rate by almost 40% (going from 5% error to roughly 3%).
- It is a fairly simple algorithm: at every training step, every neuron (including the input neurons, but always excluding the output neurons) has a probability *p* of being temporarily "dropped out", meaning it will be entirely ignored during this training step, but it may be active during the next step.
- The hyperparameter *p* is called the *dropout rate*, and it is typically set between 10% and 50%: closer to 20%-30% in recurrent neural networks, and closer to 40%-50% in convolutional neural networks.
- After training, neurons don't get dropped anymore.
- Neurons trained with dropout cannot co-adapt on their neighboring neurons; they must be as useful as possible on their own.
- They also cannot rely excessively just a few input neurons; they must pay attention to each of their input neurons. They end up being less sensitive to slight changes in the inputs.
- In the end, you get a more robust network that generalizes better.
- Another way to understand the power of dropout is to realize hat a unique neural network is generated at each training step. 
- Since each neuron can be either present or absent, there are a total of $2^N$ possible neural network (where N is the number fo droppable neurons). This is such a huge number that it is virtually impossible for the same neural network to be sampled twice.
- Once you have run 10,000 training steps, you have effectively trained 10,000 different neural networks, each with just one training instance.
- These neural networks are obviously not independent, as they share many of their weights, but they are nevertheless all different.
- The resulting neural network can be seen as an averaging ensemble of all these smaller neural networks.
- In practice, you can usually apply dropout only to the neurons in the top one to three layers (excluding the output layer).
- There is one small but important technical detail: Suppose $ p =75%$, then on average only 25% of all neurons are activate at each step during training.
- This means after training, a neuron would be connected to four times as many inputs neurons as it would be during training.
- To compensate for this fact, we need to multiply each neuron's input connections weights by four during training.
- If we don't, the neural network will not perform as well, as it will see very different data during and after training.
- More generally, we need to divide the connection weights by the *keep probability* (1 - *p*) during training.
- To implement dropout using Keras, you can use the `tf.keras.layers.Dropout` layer.
- During training, it randomly drops some inputs (setting them to 0), and divide the remaining inputs by the keep probability.
- After training, it does nothing: it just pass the inputs to the next layer.
- Since dropout is only active during training, comparing the training loss and the validation loss can be misleading.
- In particular, a model may be overfitting the training set, yet have similar training and validation losses.
- So, make sure to evaluate the training loss without dropout (e.g., after training).
- If you observe that the model is overfitting, increase the dropout rate. Conversely, try decreasing the dropout rate if the model is underfitting the training set.
- It can also help to increase the dropout rate for large layers, and reduce it for small ones.
- Moreover, many state-of-the-art architectures only use dropout after the last hidden layer (the one next to the output layer), so you may want to try that as well if full dropout is too strong.
- Dropout does tend to significantly slow down converge, but it often results in a better model when tuned properly. So, it is generally worth the extra time and effort, especially for large models.
- If you want to regularize a self-normalizing network based on the SELU activation function (as discussed earlier), you should use *alpha dropout*: this is a variant of dropout that preserves the mean and standard deviation of its inputs. It was introduced in the same paper of SELU, as regular dropout would break self-normalization.

## Monte Carlo (MC) Dropout

- In a [2016 paper]() by Yarin Gal and Zoubin Ghahramani added a few good reasons to use dropout:
    - First, the paper established a profound connection between dropout networks (i.e., neural networks containing `Dropout` layers) and approximate Bayesian inference: they shows that training a dropout network is mathematically equivalent to approximate Bayesian inference in a specific type of probabilistic model called a *deep Gaussian process*. This gives dropout a solid mathematical justification.
    - Second, the authors introduced a powerful technique called *MC dropout*, which can boost the performance of any trained dropout model without having to retrain it or even modify it at all. It also provides a much better measure of the model's uncertainty, and it can be implement in just a few lines of code.
- The code in the learning notebook does the following:
    - `model(X)` is similar to `model.predict(X)`, expect it returns a tensor rather than a NumPy array, and it supports the `training` argument.
    - Setting `training=True` ensures that the `Dropout` layer remains active, so all predictions will be a bit different.
    - We just make 100 predictions over the test set, and we compute their average.
    - More specifically, each call to the model returns a matrix with one row per instance and one column per class.
    - Because there are 10,000 instances in the test set and 10 classes, this is a matrix of shape [10000, 10].
    - We stack 100 such matrices, and ends up with a 3D array of shape [100, 10000, 10].
    - Once we average over the first dimension (`axis=0`), we get an array of shape [10000, 10], like we would get with a single prediction.
- Averaging over multiple predictions with dropped out turned on gives us a Monte Carlo estimation that is generally more reliable than the result of a single prediction with dropout turned off.
- MC dropout tends to improve the reliability of the model's probability estimates.
- This means it's less likely to be confident but wrong, which can be dangerous: imagine a self-driving car confidently ignoring a pedestrian crossing the street.
- It's also useful to know exactly which other classes are most likely. Additionally, you can have a look at the standard deviation of the probability estimates.
- The number of Monte Carlo samples you use (100 in this example) is a hyperparameter you can tweak:
    - The higher it is, the more accurate the predictions and their uncertainty estimates will be.
    - However, if you double it, inference time will also be doubled.
    - Moreover, above a certain number of samples, you'll notice little improvement.
    - Your job is to find the right trade-off between latency and accuracy, depending on your application.
- If your model contains other layers that behave in a special way during training (such as `BatchNormalization` layers), then you should not force training mode like we just did.
- Instead, you should replace the `Dropout` layers with the `MCDropout` class (the detail is in the learning notebook).
- We do that by subclass the `Dropout` layer and override the `call()` method to force it `training` argument to `True`.
- Similarly, you could define an `MCAlphaDropout` class by subclassing `AlphaDropout` instead.
- If you create a model from starch, it's just a matter of using `MCDropout` instead of `Dropout`.
- But if you have a model that was already trained using `Dropout`, you need to create a new model that's identical to the existing model, expect with `MCDropout` instead of `Dropout`, then copy the existing model's weights to your new model.
- In short, MC dropout is a great technique that boost dropout models and provides better uncertainty estimates.
- Of course, since it is just regular dropout during training, it also acts like a regularizer.
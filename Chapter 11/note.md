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
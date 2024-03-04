1. The problem that Glorot initialization and He initialization aim to fix is the unstable gradient problem:
- They try to make the output layer's standard deviation is as close as possible to the input layer's standard deviation, at least at the beginning of training (where the neural net suffer the unstable problem the most).
2. No, it is not OK to initialize all the weights to the same value, whatever that value is come from:
- Each weight must be sampled independently, and they should not have same initial value.
- We have already explained why in this chapter 10: the whole layer will act as it has only one node, just much slower. This configuration is almost guarantee to not converge to a good solution.
3. It's totally fine to initialize the bias term to 0. Some people initialize them just like weights, and that's OK, too: it does not make much a difference.
4. Here are the cases we want to use each activation functions we discussed in this chapter:
- ReLU is usually a good default for the hidden layers, as it is fast and yields good results. Its ability to output precisely zero can also be useful in some cases (see chapter 17). Moreover, it can sometimes benefit from optimized implementations as well as hardware acceleration.
- The Leaky ReLU variant can improve the model's quality without hindering its speed too much compared to ReLU.
- For large neural nets and more complex problems, GELU, Swish and Mish cna give you a slightly higher quality model, but they come at a cost of computation.
- The hyperbolic tangent (tanh) can be useful in the output layer if you need the output is in a fixed range (by default between -1 and 1), but nowadays, it is not used much in hidden layers, expect for recurrent nets.
- The sigmoid is also useful in the output layer when you need to estimate a probability (e.g., for binary classification), but it is rarely used in hidden layers (there are exceptions, for example, for the coding layers of variational autoencoders; see chapter 17).
- The softplus activation function is useful in the output layer when you need to ensure that the output will always be positive.
- The softmax activation function is useful in the output layer to estimate probabilities for mutually exclusive classes, but it is rarely (if ever) used in hidden layers.
5. This will happen if you set the `momentum` hyperparameter too close to 1 (e.g., 0.99999) when using an `SGD` optimizer:
- Too high momentum means too low friction, which means the model will pick up a lot of speed.
- This will make it roll down toward the minimum very fast, but its momentum will carry it pass the minimum.
- Then it will come back, accelerate again, overshoot again.
- It will oscillate this way many times before converging, so overall, it takes much more times to converge than with a slightly smaller value.
6. Three way to produce a sparse model:
- Train the model normally, then zero out tiny weights.
- For more sparsity, apply $\ell_1$ regularization during training, which pushes the optimizer toward sparsity.
- Another option is to use TensorFlow Model Optimization Toolkit.
7. 
- Yes, dropout does slow down training, in general roughly by a factor of two.
- Dropout doesn't slow down inference, as it's only turned on during training.
- MC dropout is exactly like dropout during training.
- But it's also turned on during inference, so each inference is slightly slowed down.
- More importantly, when using MC dropout, you want to run 10 times or more to get a better result.
- This means MC dropout is slower by a factor of the number of times you run (10 or more in this case).
## Linear model
- Linear regression is the simplest form of regression.
- Linear regression is a model-based approach, with the model is the equation: 
$$\hat{y} = \theta_0 + \theta_1x_1+\theta_2x_2+\dots+\theta_nx_n$$
with:
- $\hat{y}$ is the predicted value
- n is the number of features
- $x_i$ is the i-th feature value
- $\theta_i$ is the hyperparameter of the model. $\theta_0$ is the bias and $\theta_1, \theta_2, \dots, \theta_n$ are the weights <br>

This equation can be rewrite as:
$$\hat{y}=h_\theta(x)=\theta \cdot x$$ 
In this equation:
- $\theta$ is the parameter vector
- $x$ is the instance's feature vector, with $x_0$ always equals to 1.
- $\theta \cdot x$ is the dot product of $\theta$ and x, which of course is $\theta_0 + \theta_1x_1+\theta_2x_2+\dots+\theta_nx_n$
<br>

In this specific notebook, we will use MSE (mean squared error), and if we use linear regression, MSE is actually uses frequently.
<br>
$$MSE(X, h_\theta) = \frac{1}{m}\sum_{i=1}^m(\theta^Tx^{(i)}-y^{(i)})^2 $$ 
where m is the number of instances<br>
There exists a closed-form equation (a mathematical equation that gives the result directly). This is called the Normal equation: 
$$\hat{\theta}=(X^TX)^{-1}X^Ty$$
In this equation:
- $\hat{\theta}$ is the value of $\theta$ that minimizes the cost function.
- y is the target values, contains from $y^{(1)}$ to $y^{(m)}$

## Computational complexity
- In regard of the number of features (we call it n), the computational complexity is 
    > from $O(n^{2.4})$ to $O(n^3)$

    which means it scales poorly with the number of features.
- In regard of the number of instances (we call it m), the computational complexity is
    > O(m)

    which means it can deal with large training set efficiently, if they can fit in the memory.
- Time to make predictions
    > O(mn)

    which means the time to make a new prediction scales linearly with both the number of instances and the number of features.

## Gradient descent

- The idea of gradient descent is we want to minimize the cost function iteratively. 
- Imagine you stuck in a valley and the weather is extremely foggy. How can you go down to the very bottom of the valley? A good option is choose whichever direction is steepest and is down.
- In our situation:
    - The direction is the Jacobian matrix of the cost function 
    - The size of step is a constant, which we call learning rate. 
- The only hyperparameter is the learning rate:
    - If it's too small, then the model will take a very time to converge.
    - If it's too big, the model will oscillate back and forth way above the minium, and never converge within our desire range.
- Now we calculate the partial derivate of the cost function with respect to each $\theta_j$:
$$\frac{\partial}{\partial \theta_j}MSE(\theta)=\frac{2}{m}\sum_{i=1}^m(\theta^Tx^{(i)}-y^{(i)}).x_j^{(i)}$$
- In conclusion, the vector we want to move is the gradient vector of the cost function, or
$$
\begin{equation}
\Delta_\theta MSE(\theta) = \begin{pmatrix}
\frac{\partial}{\partial \theta_0} MSE(\theta) \\
\frac{\partial}{\partial \theta_1} MSE(\theta) \\
\vdots \\
\frac{\partial}{\partial \theta_n} MSE(\theta) \\
\end{pmatrix}
= \frac{2}{m} X^T(X\theta - y)
\end{equation}
$$
- The reason we have $X\theta$ and $X^T$ is we actually calculate 
$((\theta^TX^T - y^T)X)^T$ but
$$((\theta^TX^T - y^T)X)^T = X^T(\theta^TX^T - y^T)^T=X^T(X\theta - y)$$
- Opposite to SVD and Normal equation, Gradient descent scales well with the number of features, but scales badly with the number of instances. The reason it scales badly with number of instances is we feed it with the whole training set at once.
- Because the vector is uphill, we multiplies it by a negative number to make it go downhill. This is where the learning rate hyperparameter come into play: we multiply the gradient vector by $\eta$ to determine the size of step.
$$\theta^{(\text{next step})} = \theta - \eta \Delta_\theta MSE(\theta)$$
- 
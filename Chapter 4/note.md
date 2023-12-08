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
- In the plot at the end of Batch Gradient Descent, we have some following comments:
    - On the left, the learning rate is too low, so the algorithm will take a long time to reach the minium.
    - In the middle, the learning rate is pretty good. The algorithm converge rapidly to the optimal solution.
    - On the right, the learning rate is too high. The algorithm diverges, jump around over the place and actually getting further and further with each step.
- You can help the algorithm by scale every feature to the same range beforehand. 
- Depends on the plot, we should think about how to choose the learning rate. One strategy is define a tiny number $\epsilon$, called tolerance, so when the gradient vector is too small, or in other word, the norm of it is too small, we will stop.
- However, this time, time to train the model will be $O(\frac{1}{\epsilon})$, so for example, you want the tolerance to be 10x smaller to have a more precise solution, then it will take 10x time to train.

## Stochastic Gradient Descent

- At the opposite extreme, stochastic gradient descent picks a random instance in training set, as opposed to the whole training set in batch gradient descent, and calculate gradient based only on that instance.
- The advantage is it works on one instance at one time, so it fits nicely in the memory, and scales well with the number of instances. Because of these reasons, Stochastic Gradient Descent can be used as an out-of-core algorithm.
- The disadvantage is its stochastic (random) nature. Instead of converging gently to the minimum, the cost function bounces up and down, but decreases on average. Over time, it will end up very close to minimum, but the point is once it gets there, it will continue to bounce around. It can reach a very good, but not a optimal solution.
- But the random nature can be a good thing. Randomness help the algorithm jumps out of local minima, so stochastic gradient decent has better chance to find the global minima than batch gradient descent.
- Randomness is good is because it helps escape the local minima, but bad because the model can never settle at the minimum. One solution is to gradually decrease the learning rate. This process is called *simulated annealing*.
- The algorithm to determine learning rate is called learning schedule.
- If the learning rate decreases too quickly, you may get stuck in a local minima, or worse is frozen halfway to the minimum.
- If the learning rate decreases too slowly, you will bounce around the minimum and can end up with a suboptimal solution if you halt training too early.
- When using SGD, you need to satisfy 2 requirements:
    - Independent
    - Identically distributed

    which means you need to ensure every set of same size will have same probability of being chosen.
- One way to achieve it is to shuffle the instances while training:
    - Pick each instance randomly
    - Shuffle the train set at the beginning of each epoch
- If you don't shuffle the instances - for example, if the train set is sorted by label, then the model will start to optimize for the first label, then the second, until the very end, which has the risk of forgetting the first label when it reaches the end of training.

## Mini-batch Gradient Descent

- Instead of staying in the extreme, we can live in the middle. Mini-batch Gradient Descent trains on small random sets of instances called mini-batches.
- The main advantage of it over SGD is you can get a performance boost form hardware optimization of matrix operations, especially when using GPU.
- The algorithm's progress in less erratic than SGD. As a result, mini-batch GD will end up closer to the minimum that SGD, but harder for it to escape the local minima.
- In summary:
    - Batch GD will end up at the minimum, while SGD and mini-batch GD will bounce round.
    - However, SGD and mini-batch GD can reach the minimum if you have a good learning schedule.
    - Batch is more prone to stuck at local minima, while SGD and mini-batch GD can escape the local.

## The comparison table

Remind that:
- m is the number of instances
- n is the number of features

| Algorithm          | Large m | Out-of-core support | Large n | Hyperparameter | Scaling required | Scikit-learn       |
|--------------------|---------|---------------------|---------|----------------|------------------|--------------------|
| Normal equation    | Fast    | No                  | Slow    | 0              | No               | N/A                |
| SVD                | Fast    | No                  | Slow    | 0              | No               | LinearRegression   |
| Batch GD           | Slow    | No                  | Slow    | 2              | Yes              | N/A                |
| Stochastic GD      | Fast    | Yes                 | Fast    | $\geq$2        | Yes              | SGDRegressor       |
| Mini-batch GD      | Fast    | Yes                 | Fast    | $\geq$2        | Yes              | N/A                |

## Polynomial Regression

- What if the underlying model is more complex than a straight line? Surprisingly, you can use a linear model to fti nonlinear data.
- A simple way to do it is to add the power of each features, consider them to be the new features, and apply linear model on this extended set of features.
- This technique is called Polynomial Regression.
- For example, if the degree is 3 and there are 2 features, then the equation is:
$$\theta_0+\theta_1.a+\theta_2.b+\theta_3.a^2+\theta_4.ab+\theta_5.b^2+\theta_6.a^3+\theta_7.a^2.b+\theta_a.b^2+\theta_8b^3$$
- Polynomial Regression with degree=d transform an array with n features into a new array with 
$$\frac{(n+d)!}{n!d!}=C^k_{k+n}=C^n_{k+n}$$
which means the number of features explodes very fast (hyper-polynomial complexity, or higher that polynomial complexity).
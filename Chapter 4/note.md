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
\nabla_\theta MSE(\theta) = \begin{pmatrix}
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
$$\theta^{(\text{next step})} = \theta - \eta \nabla_\theta MSE(\theta)$$
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

## Learning curves

- If you decide to select a high degree, for example d=300, the model wll likely fit the training data much better than linear regression.
- The first plot drawn in the learning notebook illustrates 3 distinct model, all try to estimate the given function:
    - A linear model
    - A quadratic model
    - A 300-degrees polynomial model
- Because the underlying function is indeed a quadratic function, the 300-degrees model is severely overfitting the training data, while the linear one is unbefitting it. 
- The model is of course generalizes best with the quadratic model in this case, but in reality, when we don't know what is the underlying function, how can we know if a model is overfit or underfit?
- We learned a way to deal with it: Cross-validation:
    - If the model performs well on training data but badly on validation data, then it is overfitting
    - If the model performs badly on both the training data and the validation data, then it is underfitting.
- Another way is looking at *learning curves*, which are plots of the model performance on the training set and the validation set against the size of training set (or the training iteration).
- Note that if the model can't be trained incrementally, then you need to retrain it several times, each time on gradually larger train set.
- Conveniently, scikit-learn has learning_curve() function to help with this: It automatically trains and evaluates the model using cross-validation. 
- It trains the model on growing subsets of the training set by default, but if the model support incrementally learning, then you can set exploit_incremental_learning=True when calling learning_curve() and it will train the model incrementally instead.
- The function returns the training set size at which it evaluated the model, the training and validation score for each size and for each cross-validation fold.  

### Comment on 2 plots in the learning notebook
- In the first plot:
    - The training error at first is zero, because there are only 1 or 2 instances, which can be fitted easily.
    - Then, when the number of training instances increases, the model can no longer fit the data perfectly, both because the data itself is not linear and there are noises in the training data.
    - After that, the error continues to go up until it reaches a plateau, at which point adding more instances don't necessary make it better or worse.
    - Now consider the validation error, when there are few training instances, the model gereneralizes very badly.
    - After adding more instances, the validation error slowly decreases.
    - But again, a linear model can't fit this data well, so the validation error ends up in a plateau, near the other curve.
- In the second plot:
    - The error on the training set is much lower than before.
    - There is a big gap between the curves. Which means the model is performs significantly better on the training set than the validation set, which is the hallmark of an overfitting model.
    - Although there is a gap, the learning curves will continue to get closer if you have a much larger training set.

In short:
- An underfitting model has both curves reach a plateau, they are both close and very high. Let's explain why:
    - Both curves reach a plateau, means the model don't increase its performance when adding more instances.
    - However, the errors are high, which means it can't even fit the training well, let alone new, unseen data.
    - You need to use a more complex model or come up with better features.
- An overfitting model has a big gap between the curves and the learning curve is much lower than that of an underfitting model. Here's the explanation:
    - The learning curve is much lower, means the model is actually performs very well on train data.
    - However, there is a gap between the curves, which means the performances on the validation is low.
    - A way to improve an overfitting model is to keep feeding it more and more data.

### The bias/variance trade-off

An important theoretical results of statistics and machine learning is the fact that a model's generalization error can be expressed as the sum of three very different errors:
- Bias: This part of generalization error is due to wrong assumption, such as assuming that the data is linear instead of quadratic. A model with high bias is more likely to underfit the data. Note that this notion of bias is different from the bias of linear models.
- Variance: This part is due to the model's excessiveness sensitivity to small variations in the training data. A model with many degrees of freedom (such as high degree polynomial model) is likely to have high variance and thus overfit the training data.
- Irreducible error: This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to lean up the data. For example, fix the data sources, such as broken sensors, or detect and remove outliers.

## Regularized linear model

- A good way to reduce overfitting is to regularize the model (i.e., constrain it): The fewer degree of freedom is has, the harder it is to overfit the data.
- A simple way to regularize a polynomial model is to reduce the number of polynomial degrees.
- For a linear model, regularization is typically achieved by constraining the weights of the model. 
- There are 3 ways to constraining te weights of the linear model:
    - Ridge regression
    - Lasso regression
    - Elastic net regression

### Ridge regression (Tikhonov regularization)

- Is a regularized version of linear regression: A regression term equal to 
$$\frac{\alpha}{m}\sum_{i=1}^n\theta_i^2$$ 
is added to the MSE.
- This forces the model to not only fit the training data but also keep the weights as small as possible. 
- Note that the regularized term should only be added during training. Once the model is trained, you want to use the unregularized MSE or RMSE to evaluate the model's performance.
- The hyperparameter $\alpha$ let you control how much you regularize the model:
    - If $\alpha=0$ then ridge regression is just linear regression.
    - If $\alpha$ is very large, then the result is a flat line going through the data's mean.
- This is the ridge's cost function. Note that $J(\theta)$ is commonly used for the cost functions that don't have a short name.
    $$J(\theta) = MSE(\theta) + \frac{\alpha}{m}\sum_{i=1}^n\theta_i^2$$
- Note that the bias term $\theta_0$ is not regularized.
- If we define w is the vector of the features weights (from $\theta_1$ to $\theta_n$), then the above equation can be simplified as:
    $$J(\theta) = MSE(\theta) + \frac{\alpha}{m}(\|w\|_2)^2$$
which means we are considering the $\ell_2$, stands for 2D Euclidean Norm.
- If you are using Batch GD, then just add 
    $$2\frac{\alpha w}{m}$$
to the gradient vector that corresponds to the features weight, while add nothing to the bias term.
> It's important to scale the data before performing ridge regression, as it is sensitive to the scale of the input features. This is true for most regularized models.
- Look at the plot in the learning notebook, we have some comments:
    - Increase the $\alpha$ leads to flatter (i.e. less extreme, more reasonable) predictions.
    - So increase $\alpha$ leads to increasing bias and decreasing variance.
- As with linear regression, we can perform ridge regression using either a closed-form equation or by performing a gradient descent.
- The pros and cons are the same.
- The closed-form equation of ridge regression is:
    $$\hat{\theta} = (X^TX+\alpha A)^{-1} X^Ty$$
where A is the n-1 dimensions identity matrix, expect with a in the top-left cell, corresponding to the bias term.


# Linear model
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

# Computational complexity
- In regard of the number of features (we call it n), the computational complexity is 
    > from $O(n^{2.4})$ to $O(n^3)$

    which means it scales poorly with the number of features.
- In regard of the number of instances (we call it m), the computational complexity is
    > O(m)

    which means it can deal with large training set efficiently, if they can fit in the memory.
- Time to make predictions
    > O(mn)

    which means the time to make a new prediction scales linearly with both the number of instances and the number of features.

# Gradient descent

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

# Stochastic Gradient Descent

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

# Mini-batch Gradient Descent

- Instead of staying in the extreme, we can live in the middle. Mini-batch Gradient Descent trains on small random sets of instances called mini-batches.
- The main advantage of it over SGD is you can get a performance boost form hardware optimization of matrix operations, especially when using GPU.
- The algorithm's progress in less erratic than SGD. As a result, mini-batch GD will end up closer to the minimum that SGD, but harder for it to escape the local minima.
- In summary:
    - Batch GD will end up at the minimum, while SGD and mini-batch GD will bounce round.
    - However, SGD and mini-batch GD can reach the minimum if you have a good learning schedule.
    - Batch is more prone to stuck at local minima, while SGD and mini-batch GD can escape the local.

# The comparison table

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

# Polynomial Regression

- What if the underlying model is more complex than a straight line? Surprisingly, you can use a linear model to fti nonlinear data.
- A simple way to do it is to add the power of each features, consider them to be the new features, and apply linear model on this extended set of features.
- This technique is called Polynomial Regression.
- For example, if the degree is 3 and there are 2 features, then the equation is:
$$\theta_0+\theta_1.a+\theta_2.b+\theta_3.a^2+\theta_4.ab+\theta_5.b^2+\theta_6.a^3+\theta_7.a^2.b+\theta_a.b^2+\theta_8b^3$$
- Polynomial Regression with degree=d transform an array with n features into a new array with 
$$\frac{(n+d)!}{n!d!}=C^k_{k+n}=C^n_{k+n}$$
which means the number of features explodes very fast (hyper-polynomial complexity, or higher that polynomial complexity).

# Learning curves

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

## Comment on 2 plots in the learning notebook
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

## The bias/variance trade-off

An important theoretical results of statistics and machine learning is the fact that a model's generalization error can be expressed as the sum of three very different errors:
- Bias: This part of generalization error is due to wrong assumption, such as assuming that the data is linear instead of quadratic. A model with high bias is more likely to underfit the data. Note that this notion of bias is different from the bias of linear models.
- Variance: This part is due to the model's excessiveness sensitivity to small variations in the training data. A model with many degrees of freedom (such as high degree polynomial model) is likely to have high variance and thus overfit the training data.
- Irreducible error: This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to lean up the data. For example, fix the data sources, such as broken sensors, or detect and remove outliers.

# Regularized linear model

- A good way to reduce overfitting is to regularize the model (i.e., constrain it): The fewer degree of freedom is has, the harder it is to overfit the data.
- A simple way to regularize a polynomial model is to reduce the number of polynomial degrees.
- For a linear model, regularization is typically achieved by constraining the weights of the model. 
- There are 3 ways to constraining te weights of the linear model:
    - Ridge regression
    - Lasso regression
    - Elastic net regression

## Ridge regression (Tikhonov regularization)

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

## Lasso regression

- Least absolute shrinkage and selection operator regression, usually called lasso regression, is another regularized version of linear regression.
- Instead of using $\ell_2$ norm, lasso uses $\ell_1$ norm. This is its equation:
    $$J(\theta) = MSE(\theta) + 2\alpha \sum_{i=1}^n|\theta_i|$$
- An important characteristic of lasso is it tends to eliminate the weights of the least important features.
- In other words, lasso regression automatically performs feature selection and output a sparse model with few nonzero feature weights and output a sparse model (i.e. with few nonzero weights).
- If we assume that the derivate of $|x|$ is $sign(x)$, then the formula of gradient vector in case of Lasso regression is:
    $$\begin{equation}
    g(\theta, J) = \nabla_{\theta}MSE(\theta) + \alpha
        \begin{pmatrix}
        0 \\
        sign(\theta_0) \\
        sign(\theta_1) \\
        \vdots \\
        sign(\theta_n)
        \end{pmatrix} 
        \text{where sign(x)} = 
        \begin{cases}
        -1 \text{  if } x < 0 \\
        0 \text{  if } x = 0 \\
        1 \text{  if } x > 0 \\
        \end{cases}
    \end{equation}$$

## Comment on plots:
Now, let's consider the 4 plot in the learning notebook, note that we initialize the model parameters to $\theta_1=2$ and $\theta_2=0.5$.
- The two upper plot demonstrates Lasso's cost function:
    - The top-left plot represents the $\ell_1$ norm. Running GD decrease both of them equally, so $\theta_2$ reaches 0 first, as it is smaller. After that, GD will push $\theta_1$ until $\theta_1=0$.
    - The top-right plot, the contour represents the Lasso's cost function. The small white circles shows the path GD takes to reach the optimize the model parameters, which were initialized at $\theta_1=0.25$ and $\theta_2=-1$. Notice how the model reaches $\theta_2=0$ rapidly, then follow the gutter to ends up bouncing around the global optimum, i.e. the red square.
    - The bouncing pattern occurs because the gradient vector of $\ell_1$ is not continuous, it discontinues as 0. 
    - If we increase $\alpha$, the optimum will slide along the left of the dashed yellow line, while if we decreases $\alpha$, the optimum will go right.
- The two lower plot illustrates Ridge's cost function:
    - In the bottom-left plot, $\ell_2$ is proportional to the distance to the origin. So GD just takes a straight line towards that point.
    - In the bottom-right plot, the contours represent the Ridge Regression's cost function.
    - There are 2 main differences: The gradients get smaller, so GD naturally slows down, which help converges (there is no bouncing around) and the optimum value get closer to the origin as $\alpha$ get larger when you increases $\alpha$, but they never get eliminated entirely.

## Elastic Net

- Elastic Net is a middle ground between Lasso Regression and Ridge Regression.
- The regularization term is a mix of both Lasso and Ridge's regularization terms, and you can control the mix ratio r.
- The formula is:
    $$J(\theta) = MSE(\theta) + r2\alpha\sum_{i=1}^n|\theta_i|+(1-r)\frac{\alpha}{m}\sum_{i=1}^n\theta_i^2$$
- When $r=0$, elastic net is equivalent to ridge regression, when $r=1$, it is equivalent to lasso regression.
- So when you should use elastic net, ridge, lasso or plain old linear regression? Here are some good rule of thumb:
    - It is almost preferable to have at least some little bit of regularization. So in general, you should avoid plain linear regression.
    - Ridge is a good default, but if you suspect that just a few features is important, then you should prefer elastic net or lasso because they tend to reduce the useless features' weights down to zero.
    - In general, elastic net is more prefer than lasso, because lasso may behave erratically when the number of features is more than the number of instances or when some features are strongly correlated.

## Early Stopping

- A very different approach to regularize iterative learning algorithm such as GD is to stop training as soon as the validation error reaches a minimum. This is called early stopping.
- As you can see from the plot in the learn notebook, first the valid error go down and then start increasing. The valid error increases is an indicator that the model is overfitting the train data.
- Using early stopping, you stop the training as soon as the validation reach the minimum.
- Early stopping is such an easy and efficient regularization technique that Geoffrey Hinton called it a "beautiful free lunch".
- However, there are some notes:
    - Because the stochastic nature of SGD, it may be the case that the validation error can decrease later, i.e. the minimum discussed above could be local minimum.
    - On the other hand, also because the stochastic nature of SGD, the valid error will fluctuate near around a value, not necessary reach a plateau, so it could be a waste of time to wait for SGD.
    - To solve the first problem, we set a tolerance (tol) parameter that we set a new lower score as the new best valid score only if the that new lower score if smaller than the current best score - tolerance:
    ```[python]
    if valid_error < best_valid_error - tol:
        best_valid_error = valid_error
    ```
    - To solve the second problem, if after n_iter_no_change iterations, we don't change the best valid score, then we quit training:
    ```[python]
    n_iter_ = 0
    if n_iter_ < n_iter_no_change:
        n_iter_ += 1
    else:
        # quit training
    if valid_error < best_valid_error - tol:
        best_valid_error = valid_error
        n_iter_ = 0
    ```
# Logistic Regression

- As discussed in chapter 1, some regression algorithm can be used for classification tasks, and vice versa.
- Logistic regression (also called logit regression) is commonly used to estimate the probability that an instance belong to a particular class.
- If the estimated possibility is greater than a threshold, then the model predicts that the instance belongs to that class (called the *positive class*, label "1"), and otherwise it predicts that the instance does not (i.e. it belongs to the negative class, label "0").
- So this is a binary classifier.

## Estimating Probabilities 

- The way logistic regression work is instead of output the the weighted sum of the input features (plus the bias term) like Linear Regression, it output the logistic of this result.
- The estimated probability of logistic regression model is:
    $$\hat{p} = h_{\theta}(x) = \sigma(\theta^Tx)$$
- The logistic, noted $\theta$, is a *sigmoid function* that output number between 0 and 1:
    $$\sigma(x)=\frac{1}{1+e^{-x}}=\frac{e^x}{1+e^x}$$
- Anh here is how we predict:
$$
\hat{y} = \begin{cases} 
           0 \text{ if } \hat{p} < 0.5 \\
           1 \text{ if } \hat{p} \geq 0.5 \\
          \end{cases}
$$
- So a logistic regression model using the default threshold of 50% probability will predict 1 if $\theta^Tx$ is positive and 0 if it is negative.
- The score t is often called the logit. The name comes from the fact the logit function, defined as:
    $$logit(p) = ln\left(\frac{p}{1-p}\right)$$
    is the inverse function of the logistic function. Indeed, if you compute the logit of the estimated probability p, you will get the score t.
- The logit also called the log-odds, because it is the log of the ratio between the estimated probability of the positive class and the estimated probability of the negative class.

## Training and Cost Function

- The goal of training is to set the parameter vector $\theta$ so that the model estimated high possibilities for positive instance (y=1) and low possibility for negative instance (y=0).
- This idea is captured in the following cost function for a single training instance:
    $$c(\theta) = \begin{cases}
                  -ln(\hat{p}) \text{, if } y = 1 \\
                  -ln(1 - \hat{p}) \text{, if } y = 0 \\  
                  \end{cases}
    $$
- This cost function makes sense because $0 \leq \hat{p} \leq 1$ so when t approaches 0, $-ln(t)$ grows very large and when t approaches 1, $-ln(t)$ approaches 0.
- If the model estimates a probability closed to 0, then the cost will be large for a positive instance and small for a negative instance.
- On the other hand, if the model estimates a probability closed to 1, then the cost will be small for a positive instance and large for a negative instance.
- The cost function over the whole training set is:
    $$J(\theta) = -\frac{1}{m}\sum_{i=1}^m y^{(i)}ln(p^{(i)}) + (1-y^{(i)})ln(1 - p^{(i)})$$
- Think about the cost function this way: We can rewrite the $c(\theta)$ as below:
    $$c(\theta) = \begin{cases}
                  -ln(\hat{p}) \text{, if } y = 1 \text{ and } 1 - y= 0 \\
                  -ln(1 - \hat{p}) \text{, if } 1 - y = 1 \text{ and } y = 0\\  
                  \end{cases}
    $$
    - So if y=1 then 1-y=0, then the term it the sum is indeed $c(\theta)$. Similar to y=0.
    - A way to remember the cost function: In the rewrite $c(\theta)$, take the equation, multiply by the first condition, add them together.
    - Another way is think about the cost function as we want to calculate the estimated value, which is the value in each case, multiply with the probability that case happen, just slightly different that we multiply with ln. So in each case, we have the value is $y^{(i)}$, the ln of probability is $ln(p^{(i)})$.
    - In both way, because in logistic regression, there are only 2 classes, so we have $1-y$ and $1-\hat{p}$.
    - In both way, don't forget the minus sign.
- This log loss has its reasons. It can be shown mathematically (using Bayesian inference) that minimizing the cost will result in the *maximum likelihood* of the model to be predict optimally, assuming that the instances follow a Gaussian distribution around their means. When you use the log loss, that is the implicit assumption you are making. The more wrong this assumption is, the more biased the model will be.
- Similarly, when you use MSE to train linear regression, you are making the implicit assumption that the data is purely linear, plus som Gaussian noise. So if the data is not linear (e.g. it is quadratic), or the noise is not Gaussian (e.g. if the outliers is not exponentially rare), the model will be biased.
- The bad new is that there is no known closed-form equation to compute the $\theta$ that minimizes the cost function directly.
- The good new is that the cost function is convex, so Gradient Descent (or any other optimization algorithms) is guaranteed to find the global minimum.
- The partial derivate of the cost function with regard to each $\theta_j$ is:
    $$\frac{\partial}{\partial\theta_j}J(\theta)=\frac{1}{m}\sum_{i=1}^m (\sigma(\theta^Tx^{(i)}) - y^{(i)})x_j^{(i)}$$
- This equation is very similar to the partial derivate of the linear regression's cost function: We first compute the prediction error, then we multiply that different by the j-th feature value, finally it computes the average over the whole training instances. 
- The rest is similar to linear regression: If you want to use batch gradient descent, you compute the gradient vector as above and then train it, same with SGD and mini-batch.

## Decision boundaries

- The petal width of the Iris virginica flowers (represented by green triangles) ranges from 1.4 cm to 2.5 cm, while the other iris flowers (represented by blue triangles) ranges from 0.1 cm to 1.8 cm. Clearly, there is a bit of overlap.
- Above 2 cm, the model highly confident that the instance is *Iris virginica*, while below 1 cm, it highly confident that the instance is not *Iris virginica*. However, between there extremes, the model is not sure.
- If you ask it to predict the class (using predict() instead of predict_proba()), it will return whichever class has higher probability.
- Therefore, there exists a decision boundary at around 1.65 cm, where both probability is equal to 50%: If the petal width is higher than 1.65 cm, the classifier predict that the flower is *Iris virginica*, otherwise it predicts that the instance is not (albeit not very confident).
- The second plot shows the same dataset, but this time displaying two features: petal width and length. The model uses these 2 features to estimate the probability a new flower is an *Iris virginica*.    
- The dashed line is where the model estimates a 50% probability: This is the model's decision boundary. Note that this is a linear boundary.
- Each color-coded parallel line illustrates the points where the model predicts a specific probability, ranges from 15% (bottom left) to 90% (top right). All the flowers above the most top-right has over 90% chance to be a *Iris virginica*, according to the model.
- Just like Linear Regression, Logistic Regression can also be regularized using $\ell_1$ and $\ell_2$ penalities. Scikit-learn actually uses $\ell_2$ by default.
- The hyperparameter controlling the regularization length of a Scikit-learn's Logistic Regression is not the alpha, but its inverse C. The higher the value of C, the less the model is regularized.

## Softmax Regression

- The logistic regression model can be generalized to classify multiclass directly, without having to train and combine multiple binary classifiers, as discussed in chapter 3. That model is called Softmax Regression, or multinomial logistic regression.
- The idea is simple: When given an instance x, the Softmax Regression model first a score $s_k(x)$ for each class k, the estimates the probability for each class by using the *softmax function* (also called the *normalized exponential*) to the scores.
- The equation to compute $s_k(x)$ is actually the same as Linear Regression:
    $$s_k(x) = (\theta^{(k)})^T.x$$
- After computing the score of every class for the instance x, now you can estimate the probability
    $$\hat{p_k}=\sigma(s(x))_k=\frac{exp(s_k(x))}{\sum_{j=1}^k exp(s_j(x))}$$
- In this equation:
    - k is the number of classes.
    - s(x) is a vector containing the scores of each class for the instance x.
    - $\sigma$ is the softmax function.
    - $\sigma(s(x))_k$ is the estimated probability that the instance x belongs to class k, given the scores of each class for that instance.
- Like the Logistic Regression classifier, Softmax Regression classifier predicts the class with the highest estimated possibility:
    $$\hat{y} = \underset{k}{\mathrm{argmax}} \ \hat{p_k} = \underset{k}{\mathrm{argmax}} \ \sigma(s(x))_k = \underset{k}{\mathrm{argmax}} \  s_k(x)= \underset{k}{\mathrm{argmax}} \ \left( (\theta^{(k)})^T.x\right)$$
- The softmax function predicts one class at a time, so it is multiclass, not multioutput. So it should be use to predict mutually exclusive classes, such as different species or plants. You cannot use it to predict people in a image, as many people can be in the same picture.
- The objective is to have a model that estimates a high probability for the target class (and consequently low probability for other classes). 
- Minimizing the cross entropy cost function should lead to this objective, because it penalizes the model if it estimates a low probability for the target class:
    $$J(\theta) = -\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)}ln(\hat{p_k}^{(i)})$$
- In the equation:
    - $y_k^{(i)}$ is the target probability that the instance belong to class k. In general, it is either 1 or 0, depending on the instance belong to the class or not.
    - $\hat{p_k}^{(i)}$ is the estimated probability that the instance belong to class k.
    - K is the number of classes.
- The trick to remember this cost function is the same as the Logistic Regression's cost function. In fact, the Logistic Regression's cost function is just a special case of the cross entropy cost function (K = 2).
- The gradient vector of this cost function with regard to $\theta^{(k)}$ is given by:
    $$\nabla_{\theta^{(k)}}J(\Theta) = \frac{1}{m}\sum_{i=1}^m\left(\hat{p_k}^{(i)}-y_k^{(i)}\right)x^{(i)}$$
- Now you can apply gradient descent (or any other optimization algorithms) to find the parameter matrix $\Theta$ that minimizes the cost function.

### Comment on the plot:

- Notice that the decision boundaries between any two classes is a straight line.
- The figure also shows the estimated probability for the *Iris versicolor* class, represented by the curve lines.
- Note that the model can predict a class that has an estimated probability below 50%. For example, at the point where all three decision boundaries meet, the estimated probability for all classes is all equal to roughly 33% (i.e. 1/3).

### Extra topic: Dive into Cross entropy

- Cross entropy originated from Claude Shannon's information theory.
- Suppose you want to transmit information about the weather everyday. If there are eight options (sunny, rainy, etc), you could encode each options using 3 bits, because $2^3 = 8$. However, if you think it will be sunny almost everyday, you can take advantage of it and encode 0 to be 'sunny', using just 1 bit, and the other 7 options using 4 bits (starting with a 1). This is much more efficient, given your assumption is correct.
- Cross entropy measures the average of bits you actually send per option. Academically, cross-entropy is the average number of bits needed to encode data from a source of distribution p when we use model q.
- If your assumption about the weather is perfect, then cross entropy will be the same as the entropy of the weather (i.e. its intrinsic unpredictability). But if you assumption is wrong (i.e. if it rains often), cross entropy will be higher by an amount called the *Kullback-Leiber (KL) divergence*.
- The cross entropy between two probability distributions p and q is defined as:
    $$H_p(p, q) = E_p(-ln(q(x))) = -\sum_{x \in \Omega} p(x)ln(q(x))$$
- For more detail, check out [this video](https://www.youtube.com/watch?v=ErfnhcEV1O8).
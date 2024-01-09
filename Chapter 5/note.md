# Support Vector Machine

- SVM (abbreviate for Support Vector Machine) is a powerful and versatile Machine Learning model, which can be used for classification, regression and even outlier detection.
- SVM is based on the idea that we can separate two part of data using a line, or a plane, or a hyperplane (a plane in high dimension).
- SVM is just trying to draw a plane to separate two part between training set, which means trying to find a weight vector **w** such that:
    $$w_1x_1+\dots+w_nx_n$$
    drawing a hyperplane that separates well enough, with n is the input dimension.
- In other words, we trying to find a vector of n-dimension to optimize a function go from $\mathbb{R}^n$ to $\mathbb{R}$.
- The name of that line is boundary decision.
- In classification tasks, we want that line not only separate 2 part of the training set, but also has the widest possible width. Hence, the name of it is *large marin classification*.
- A good analogy is thinking of SVC as trying to fitting the widest street possible between the classes.
- Notice that adding more instances "off the street" does not change the boundary decision: It is fully determined (or "supported") by the instances located at the edge of the street. These instances are called "support vectors".

## Comment on plots
This topic is best learned using images:
- In the first images:
    - On the left plot, the green line can't even separate 2 part of training set. The magenta and red line can separate the training set, but we don't think they can generalize well.
    - On the right plot, The black line is the line separate the training set and can generalize well because this line stay as far as possible from both part of training set. The 2 marked data points are the data points closest to the boundary decision, so they are support vectors. 
- In the second images: 
    - On the left plot, because y axis has a much larger scale, it results in a nearly flat line. But as you can see, it won't generalize so well.
    - On the second plot, the line can generalize better.
    - In conclusion, SVMs are sensitive to the scales of features, so remember to scale the features before apply SVM. 

# Soft Margin Classification

- If we apply the rule that every instance must be off the street and on the right side, then it's called *hard margin classification*.
- The problem with hard margin classification is it only works with linearly separable data and extremely sensitive to outliers.
- In contrast with hard margin classification, support vectors in soft margin classification consist of instances lie inside the margin, on the edge of margin and on the wrong side of the model.

## Comment on plots
- Look at the first two plots in the learning notebook:
    - In the first plot, you can't find a hard margin.
    - In the second plot, because of a single outlier, the decision boundary ends up very different from the previous plot, so it won't generalize well.
- To avoid this issue, just use a more flexible model. The goal is to find a good balance between keeping the street as large as possible and limiting the *margin violations* (i.e. instances that ends up in the middle of the street or on the wrong side). This is called *soft margin classification*.
-  When using Scikit-learn to create an SVM model, we have some hyperparameters to tweak. One of them is C, the regularization hyperparameter. 
- Consider the next two plots in the learning notebook:
    - The left plot has a smaller C, which means it is less regularized.
    - Reducing C makes the street larger, but it also leads to more margin violations.
    - In other words, reducing C results in more instance support the street, so there is less risk of overfitting. Just look at the previous plot, you can see if you consider one more yellow circle, the model is less prone to the outlier (i.e. less overfitting).
    - But if you reduce it too far, you end up with an underfitting model, which is likely our case: The model on the right seems to be generalized better than the left one.
    - A good rule of thumb: Larger C, more risk of overfitting.

## Comments on code
- In the code generating the later plots, here are some clarifies:
    - We want to calculate the weights and biases of the unscaled version of SVM, given the weights and biases of the scaled SVM and the parameter of the scaling function.
    - The biases are calculated by using the decision boundary on the kernel of the scaling function.
    - The weights are computed by dividing the original weights by the the scale parameter.
- Unlike LogisticRegression, LinearSVC doesn't have a predict_proba() method to estimated the classes probabilities. 
- Which means if you uses SVC class instead of LinearSVC and set the *probability* hyperparameter to True, then the model will fit an extra model at the end to estimates probabilities.
- Under the hood, the model using 5-fold cross-validation to generate an out-of-sample prediction for every instance in the training set, then trains a *LogisticRegression* model, so it will slow down the training process considerably. After that, the predict_proba() and predict_log_proba() will be available.

# Nonlinear SVM Classification

- Even though linear SVMs work efficient and well in many cases, many datasets don't even close to being linearly separable.
- A good approach is too add more features, specially polynomial features (discussed in chapter 3).
- Consider 2 plots in the learning notebook. The data is completely unable to be linearly separable. However, when we add a new feature $x_2 = x_1^2$, then data now can be separate by a line, as you can see in the second plot.
- You can use this idea by adding a PolynomialFeatures transformer (discussed in chapter 3) in Scikit-learn.

## Polynomial Kernel

- Adding more polynomial work great with all models (not only SVMs).
- However, there are a downside: If the degree is too low, then the model can't work with very complex dataset, if the degree is too high, then the number of features will be exponentially large, lead to very slow training.
- Fortunately, in SMVs context specifically, we have a nearly miraculous technique from the mathematical world, name **"the kernel trick"**. We explain it later.
- The kernel trick let us calculate the result as if we had added the features, without actually having to add them. 
- Of course, when the model is overfitting, you should decrease the degree of the model. Conversely, if the model is underfitting, you should increase it.
- The hyperparameter coef0 controls how much the model is influenced by high-degree polynomial versus low-degree polynomial. The higher it is, the more influenced the high-degree polynomial.

## Similarity Features

- Another approach is to add features computed using a *similarity function*, which measures how much each instance closes (based a certain metric) to a particular *landmark*.
- Here, we use Gaussian RBF (Radial Basis Function) as the similarity function:
    $$\phi_\gamma=\exp(-\gamma\|x-\ell\|^2)$$ 
    Radial basis means a function only depend on the distance from that point to the origin.
- This function is a bell-shaped function, with the parameters:
    - The expected value (or mean): $\ell$
    - The derivation: $\frac{2}{\sqrt{\gamma}}$. This is the reason why $\gamma$ must be strictly positive.
- Look at the two plots in the learning notebook:
    - On the left plot, you can see the instance $x_1=-1$ locates a distance of 1 from the first landmark and 2 from the second landmark. Therefore, its new features are $x_2=\exp(-0.3 \times 1^2) \approx 0.74$ and $x_3=\exp(-0.3 \times 2^2) \approx 0.3$.
    - The right plot shows the features (dropping the original one). Now the dataset is linearly separable.
- Now, how can you choose the landmarks? The simplest way is to create a landmark at the location at the every single data point. Doing this creates lots of additional dimensions, thus increases the chance we could separate the newly generated dataset by a line. 
- However, doing so create many more features. For example, if you have m instances and n features, then doing this way will create a new dataset consists of m instances and m features (assuming you drop the original features). If the number of instances is very large, then the number of features in the new dataset will also be very large, leads to very slow training time.

## Gaussian RBF Kernel

- Similar to the polynomial features method, we can apply similarity features to any ML algorithms.
- However, as discussed above, this can leads to the explosion of the number of features, especially on large training set.
- But here again, in SMVs context specifically, we can apply the kernel trick to have the results without actually have to adding many more similarity features.
- Look at the plots in the learning notebook:
    - Increasing gamma ($\gamma$) leads to smaller invariance, so the bell-shaped curve is narrower. As a result, the influence range of each instances ends up smaller, leads to more irregular decision boundary: It wiggle around each data point. In contrast, a small gamma will make the bell shape curve wider, leads to a smoother curve around the dataset. 
    - In short, $\gamma$ is similar to a regularization hyperparameter. The higher it is, the more prone the model is to overfitting (similar to C).

## Other kernels

- There are other kernels, but they are used much more rarely.
- Some kernels are specialized for some data structures.
- String kernels are sometimes used when classifying text documents or DNA sequences. For example:
    - String subsequence kernel: Measures the similarity of 2 strings based on the presences of common subsequence.
    - The Levenshtein distance: Also known as the edit distance, is a metric measures the minimum amount of edit operations (insert, delete, replace).
 ## How to choose the kernel

 - You should try the linear kernel first, especially when the training dataset is very large or has a huge amount of features. Note that LinearSVC is faster than SVC(kernel='linear).
 - If you have plenty of time left, yo can try the Gaussian RBF kernel, as it works well in most cases.
 - If you still have spare time computing power, you can use a grid search to try a few more kernels.
 - Remember to consider a kernel specialized for your training set's data structure.

 ## Computational Complexity

 - The LinearSVC class is based on liblinear library, which implements an optimized algorithm for calculating linear SMVs. It does not support the kernel trick, however it scales very well with the number of instance, nearly in linear time complexity. The training complexity is roughly $O(m \times n)$. 
 - This algorithm takes more time as you require more precision. This precision is controlled by the tolerance hyperparameter $\epsilon$. However, the default value works well in most case.
 - The SVC class is based on the libsvm library, which implements an algorithm that support the kernel trick. The training time complexity is usually between $O(m^2 \times n)$ and $O(m^3 \times n)$. 
 - This means the algorithm is very slow when the number of instances is very large, so this algorithm is better suited for small or medium-sized training set. It scales well with the number of features, especially with sparse features (i.e. when each instance has few nonzero features). In this case, the algorithm scale roughly with the number of the average of nonzero features per instance. 
 - The SGDClassifier class also performs large margin classification by default, and its hyperparameters, especially the regularization hyperparameters (alpha and penalty) and the learning_rate, can be adjusted to achieve the similar results as the Linear SVMs. 
 - For training it uses stochastic gradient descent (discussed in chapter 4), which allow incremental learning and requires little memory, so we can use it on a large model that does not fit in RAM (i.e. out-of-core learning). Moreover, it scales very well, as its training time complexity is $O(m \times n)$. 

| Class          | Time Complexity                  | Out-of-core learning | Scaling Required | Kernel trick |
|----------------|----------------------------------|---------------------|------------------|--------------|
| LinearSVC      | $O(m \times n)$                  | No                  | Yes              | No           |
| SVC            | $O(m^2 \times n)$ to $O(m^3 \times n)$ | No            | Yes              | No           |
| SGDClassifier  | $O(m \times n)$                  | Yes                 | Yes              | No           |


# SVM Regression

- We can uses SVMs for regression tasks. 
- All we need to do is flip the objective around: Instead of trying to fit the largest possible street between two classes and trying to limit margin violations, we try to fit as many instances as possible while trying to limit margin violations (i.e. instances *off* the street). The width of the street is controlled by the hyperparameter $\epsilon$.
- Look at the 2 plots in the learning notebook to keep up with the following statements:
    - Reducing $\epsilon$ increases the number of support vectors, which regularizes the model. If you don't understand, tweak the epsilon to 5, you will end up with a terribly overfitting model.
    - Moreover, if you add more training instances within the margin, it will not affect the model's prediction. That's the reason why the model is said to be *$\epsilon$-insensitive*.
- Similar to classification tasks, you can apply the kernelized model to solve nonlinear regression tasks.
- And as before, the higher the C value, the more regularized the model.
- The SVR class is the regression equivalent of SVC class, and the LinearSVR is the regression equivalent of LinearSVC class. 
- The LinearSVR scales linearly with the number of training set (just like the LinearSVC class) , while the SVR class will be much more slower when the number of training instances gets large (just like the SVC class).
- SVMs can also be used for novelty detection (will be discussed in chapter 9).

# Under the Hood of Linear SVM Classifiers

- Up until now, we have used the convention of putting all the hyperparameters in a vector $\theta$, which includes the bias term $\theta_0$ and the weight terms from $\theta_1$ to $\theta_n$, and adding a bias input $x_0=1$ to all instances. From this chapter onward, we will use a more common convention: We separate the bias term *b* (equals to $\theta_0$) and the weights vector $\textbf{w}$ (containing $\theta_1$ to $\theta_n$). Then no bias feature is needed in the input features vector and the linear SVM's decision function is become to $\textbf{w}^Tx+b=w_1x_1+\dots+w_nx_n+b$.

## Decision Function and Predictions

- A linear SVM classifier predict a class for a new instance x by first computing the decision function 
    $$\textbf{w}^Tx+b=w_1x_1+\dots+w_nx_n+b$$
    If the result is nonnegative, then the model predicts the new instance is belong to the positive class (1), otherwise, it is predicted to be belong to the negative class (0).
- Making predictions is straightforward, but how about training? We need to find the weights vector $\textbf{w}$ and bias term b so that the street is the widest possible while limiting margin violations. 
- Let us start with the width of the street: To make it larger, you need to make w smaller.
- Let's define the borders of the street is the points where the decision function is equals to 1 or -1:
    - In the left plot, the weight $w_1$ is 1, so the points at which $w_1x_1=1$ or $+1$ are $x_1 = 1$ and $x_1 = -1$. Therefore, the margin's size is 2.
    - In the right plot, the weight is 0.5, so the points at which $w_1x_1=1$ or $+1$ are $x_1 = 2$ and $x_1 = -2$. Therefore, the margin's size is 4.
- So the size of margin is inversely proportional to the weights. So our goal becomes making the weights as small as possible. Note that the bias term has no influence on the size of the margin, as tweaking the bias only shift the margin around without affecting its size.
- Our second goal is limiting the margin violations. So we need to make the decision function to be greater than 1 for all the positive training instances and be lower than -1 for all the negative training instances.
- If we define $t^{(i)}=1$ for positive instances ($y^{(i)}=1$) and $t^{(i)}=-1$ for positive instances ($y^{(i)}=0$), then this constraint can be written as:
    $$t^{(i)}(\textbf{w}x^{(i)}+b) \geq 1$$
    for all training instances.
- Therefore, we can express the hard margin linear SVM classifier as an constrained optimization problem:
    $$\underset{w, b}{\text{minimize}}\frac{1}{2}\textbf{w}^T\textbf{w}$$
    subject to $t^{(i)}(\textbf{w}x^{(i)}+b) \geq 1$ for $i = 1, 2, \dots, m$.
- The reason why we minimize $\frac{1}{2}\textbf{w}^T\textbf{w}$, which is equal to $\frac{1}{2}\|\textbf{w}\|^2$, instead of $\|\textbf{w}\|$ is because $\frac{1}{2}\|\textbf{w}\|$ has a nice derivate, which is $\|\textbf{w}\|$, while $\|\textbf{w}\|$ is not differentiable at 0. Optimization problems are easier to work with differentiable functions.
- To deal with soft margin objective, we need to introduce a *slack variable* $\zeta^{(i)} \geq 0$ for each instance: $\zeta^{(i)}$ measures how much the i-th instance is allowed to violate the margin:
    $$\zeta^{(i)} = \begin{cases} 
    0 \text{, if is classified correctly} \\
    \|\textbf{w}^Tx_i+b-y_i\| \text{, if it is classified incorrectly}
    \end{cases}$$
- A more detail explanation about soft margin classification can be found in [this blog](https://machinelearningcoban.com/2017/04/13/softmarginsmv/).
- Now we have two conflicting objective: Make the slack variables as small as possible to reduce the margin violations and make $\frac{1}{2}\textbf{w}^T\textbf{w}$ as small as possible to have a wider margin. This is where the C hyperparameter come from: It let us have a trade-off between these two objectives. 
- Now we have the constrained optimization problem for the soft margin classification:
    $$\underset{w, b, \zeta}{\text{minimize}}\frac{1}{2}\textbf{w}^T\textbf{w} + C \sum_{i=1}^m \zeta^{(i)}$$
    subject to $t^{(i)}(\textbf{w}x^(i)+b) \geq 1 - \zeta^{(i)}$ and $\zeta^{(i)} \geq 0$ for $i = 1, 2, \dots, m$.
- The hard margin and soft margin problems are both convex quadratic optimization problem with linear constraints. Hence they belong to *quadratic programming* (QG) problem class. 
- There are many solver to solve quadratic programming problems, any further information can be found in 
- Using a QP solver is a way to solve the problem (using closed-form equation, as discussed in chapter 4), but you can use gradient descent to minimize the *hinge loss* or the *squared hinge loss*.
- Look at the 2 plots in the learning notebook to see the graphs of these 2 loss functions.
- Given an instance belong to the positive class, so t = 1, then the loss function is 0 if and only if $s = \textbf{w}^Tx + b \geq 1$, or in other words, the instance is off the street and on the positive side.
- Given an instance belong to the negative class, so t = -1, then the loss function is 0 if and only if $s = \textbf{w}^Tx + b \leq -1$, or in other words, the instance is off the street and on the negative side.
- The further away an instance is from its correct side of the margin, the more higher the loss: The loss grows linearly for the hinge loss, and quadratically for the squared hinge loss.
- It makes the squared hinge loss more sensitive to outliers but squared hinge loss tends to converge faster when the dataset is clean.
- By default, `Linear SVC` use squared hinge loss and `SGDClassifier` use hinge loss. You can change the loss function by setting the `loss` hyperparameter to either squared_hinge or hinge.
- `SVC` class's algorithm optimization function find a similar to minimizing the hinge loss function.

## The Dual Problem

- Given a constrained optimization problem, known as the *primal problem*, it is possible to view that same problem in other perspective, the problem expressed in that other perspective named *dual problem*.
- The solution to the dual problem typically lower than the solution to the dual problem, but under some constrains, The solution to the dual problem can be equal to the solution to the dual problem. One of them is the objective function is convex and the inequality constrains are continuously differentiable and convex functions. 
- Luckily, the SVM problem meets that constrain, so you can solve either the primal problem or the dual problem.
- The dual form of the linear soft margin SVM classifier is:
    $$\underset{\alpha}{minimize}\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha^{(i)}\alpha^{(j)}t^{(i)}t^{(i)}x^{(i)^T}x^{(j)} - \sum_{i=1}^m \alpha^{(i)}$$
    subject to $C \geq \alpha^{(i)} \geq 0$ for all $i = 1, 2, \dots, m$ and $\sum_{n=1}^m\alpha^{(i)}t^{(i)} = 0$
- Once you find the vector $\hat{\alpha}$ that minimizes the equation (using a QP solver), then you can find hyperparameters of the primal problem:
    $$\hat{\textbf{w}} = \sum_{i=1, \hat{\alpha}^{(i)} > 0}^m \hat{\alpha}^{(i)}t^{(i)}x^{(i)}$$
    $$\hat{b}=\frac{1}{n_s}\sum_{i=1, C>a^{(i)}>0}^m \left(t^{(i)}-\hat{\textbf{w}}^Tx^{(i)}\right) $$
    where $n_s$ is the number of instances that lie exactly on the border of the margin.
- The dual problem is easier to solve than the primal one when the number of features is larger than the number of training instances. More importantly, the dual problem makes the kernel trick possible, while the primal doesn't.

## Kernelized SVMs

- Suppose you wan to apply a second-degree polynomial transformation to a two-dimensional training set (such as the moons training we have created in the learning notebook), then train a linear SVM on that transformed training set.
- This is the second-degree mapping function that you wan to apply:
    $$\varphi(\textbf{x}) = \varphi\left( \begin{pmatrix} 
            x_1 \\
            x_2 \\
            \end{pmatrix}\right)
            = \begin{pmatrix} 
            x_1^2 \\
            \sqrt{2}x_1x_2 \\
            x_2^2 \\
            \end{pmatrix}$$
- Notice that the transformed vector is 3D instead of 2D. Now let's see how it unfolds if we apply this second-degree mapping to two 2D vectors $\textbf{a}$ and $\textbf{b}$ and calculate the dot product of the transformed vectors.
    $$\varphi(\textbf{a})^T \varphi(\textbf{b})= \begin{pmatrix} 
            a_1^2 \\
            \sqrt{2}a_1a_2 \\
            a_2^2 \\
            \end{pmatrix} \begin{pmatrix} 
            b_1^2 \\
            \sqrt{2}b_1b_2 \\
            b_2^2 \\
            \end{pmatrix} = a_1^2b_1^2+2a_1a_2b_1b_2+b_1^2b_2^2\\=(a_1b_1+a_2b_2)^2=\left(\begin{pmatrix} 
            a_1 \\
            a_2 \\
            \end{pmatrix}\begin{pmatrix} 
            b_1 \\
            b_2 \\
            \end{pmatrix}\right)^2=(a^Tb)^2$$
- So in the end, the dot product the transformed is equals to the square of the dot product of the original vectors.
- But for what? If you look back at the dual problem, you need to know all the parameters beforehand to apply the QP solver, however, how haven't yet had the dot product of all the vectors after we transform the dataset. 
-But if we do transomed the dataset, then it will need a lot of resources to calculate and store each transformed vector, but we can directly compute the dot product without transforming the dataset, which speed up training significantly.
- The function $K(\textbf{a}, \textbf{b})=(a^Tb)^2$ is a second-degree polynomial function. In ML, a *kernel* is a function capable of computing the dot product $\varphi(\textbf{a})^T \varphi(\textbf{b})$ based only on the original vectors $\textbf{a}$ and $\textbf{b}$, without having to perform (or even know about) the transformation $\varphi$. 
- Here are some examples of commonly used kernels:
    - Linear: $K(\textbf{a}, \textbf{b})=a^Tb$
    - Polynomial: $K(\textbf{a}, \textbf{b})=(\gamma a^Tb + r)^d$
    - Gaussian RBF: $K(\textbf{a}, \textbf{b})=\exp(-\gamma\|\textbf{a}-\textbf{b}\|^2)$
    - Sigmoid: $K(\textbf{a}, \textbf{b}) = \tanh(\gamma\textbf{a}^T\textbf{b}+r)$

### Mercer's theorem

- According to *Mercer's theorem*, a function $K(\textbf{a}, \textbf{b})$ if follow a few conditions called the *Mercer's conditions* (e.g. K must be continuos and symmetric in its arguments so that $K(\textbf{a}, \textbf{b})=K(\textbf{b}, \textbf{a})$, etc.), then there exists a function $\varphi$ that maps $\textbf{a}$ and $\textbf{b}$ into another spaces (possibly with much higher dimensions) such that $K(\textbf{a}, \textbf{b})=\varphi(\textbf{a})^T\varphi(\textbf{b})$.
- You can then use K as a kernel, because you know $\varphi$ exists, even if you don't know what $\varphi$ is. In the case of Gaussian RBF kernel, it can be shown that $\varphi$ maps each training instances into an infinite-dimensional space, so it's great knowing you don't have to perform the mapping.
- Note that some frequently used kernels (such as the sigmoid kernel), don't respect all the Mercer's conditions, but they work generally well in practice.

### Back to Kernelized SVMs

- However, now there is a problem. Now, you know to go from the dual problem to the primal one, but you need to have the value of each $\varphi(x^{(i)})$ in order to calculate $\textbf{w}$. In fact, $\textbf{w}$ has the same number of dimension as $\varphi({x^{(i)}})$, so its dimension can be very large or even infinite, so you can't compute it.
- But the point of calculate $\textbf{w}$ is just to compute the predictions, so you can once again, not having to directly compute $\textbf{w}$, but plug the formula of $\textbf{w}$ into the decision function for a new instance $\textbf{x}^{(n)}$, and you get an equation with only dot product between transformed vectors, which we now know how to deal with using the kernel trick:
    $$\begin{align*}
        h_{\hat{\textbf{w}}, \hat{b}}(\varphi(\hat{\textbf{x}}^{(n)})) 
        &= \hat{\textbf{w}}^T\varphi(\hat{\textbf{x}}^{(n)})+\hat{b} \\
        &=\left(\sum_{i=1, \hat{\alpha}^{(i)}>0}^m\hat{a}^{(i)}t^{(i)}\varphi(\hat{\textbf{x}}^{(i)})\right)^T\varphi(\hat{\textbf{x}}^{(n)}) + b \\
        &=\sum_{i=1, \hat{\alpha}^{(i)}>0}^m\hat{a}^{(i)}t^{(i)}\left(\varphi(\hat{\textbf{x}}^{(i)})^T\varphi(\hat{\textbf{x}}^{(n)})\right) + b \\
        &= \sum_{i=1, \hat{\alpha}^{(i)}>0}^m\hat{a}^{(i)}t^{(i)}K(\textbf{x}^{(i)}, \textbf{x}^{(n)}) + b \\
    \end{align*}$$
- Note that $ \hat{\alpha}^{(i)} > 0$ only for support vectors (i.e. instances that lie on the border of margin, in the margin or the wrong side of the margin), making predictions with new instance only involve computing the dot product of the new input vector $\textbf{x}^{(i)}$ with support vectors, not the whole training set. 
- Of course, you can use the same technique to compute the bias term $\hat{b}$:
    $$\begin{align*}
    \hat{b} &= \frac{1}{n_s}\sum_{i=1, C>a^{(i)}>0}^m \left(t^{(i)}-\hat{\textbf{w}}^T\varphi(x^{(i)})\right) \\
            &= \frac{1}{n_s}\sum_{i=1, C>a^{(i)}>0}^m \left(t^{(i)}-\left(\sum_{i=1, \hat{\alpha}^{(i)}>0}^m\hat{a}^{(i)}t^{(i)}\varphi(\hat{\textbf{x}}^{(i)})\right)^T\varphi(x^{(i)})\right) \\
            &= \frac{1}{n_s}\sum_{i=1, C>a^{(i)}>0}^m \left(t^{(i)}-\sum_{i=1, \hat{\alpha}^{(i)}>0}^m\hat{a}^{(i)}t^{(i)}K(\textbf{x}^{(i)}, \textbf{x}^{(n)})\right) \\
    \end{align*}$$
- Many machine learning algorithm involve thousands or even millions of features for each training instance. This not only slows down the training process a lot, but also makes finding a good solution much harder. This is called the *curse of dimensionality*.
- Lucky for us, there are techniques to reduce the number of features considerably, turn an intractable problem into a tractable one.
- For example, consider the MNIST problem: 
    - We can ignore the pixels in the corner and on the edge completely, as they are almost always white. 
    - We did see in the previous chapter (in the Random Forests part) that these pixels is barely important.
    - So we can drop these, without worrying about losing much information.
    - Additionally, two neighboring pixels is highly correlated: If you merge them into a single pixel (e.g. takes the mean of the two pixel intensities).
- Reducing dimensionality does cause some information loss, so even it speeds up the training time, it may make your system performs slightly worse. 
- It also makes your system more complex and thus harder to maintain.
- The author recommends to try to train the system on the full original training set before considering dimensionality reduction.
- Sometimes, reducing dimensions can filter out some noises and unnecessary detail and thus increase the performance of the model, but in general, it will just speeds up training.
- Apart form speed up training, dimension reduction can help visualize data. Reducing the number of dimensions down to only two (or three) can allow us to plot a condensed view of high-dimensional data in a single graph and allow us to gain some visual insights by visually detecting patterns, such as clusters.
- Moreover, data visualization is ver important when explain your conclusions to people who are not data scientist - in particular, decision makers who will use your results.

# The Curse of Dimensionality

- We human all live in a three dimensions reality (4 if you consider time, a little more if you are a string theorist) that our intuition tricks us when we try to imagine a high-dimensional space. Even a 4D cube is stupidly hard to imagine in our mind, let alone a 1,000-dimensional ellipsoid bends in  a 10,000-dimensional hyperplane.
- It turn outs there are many counterintuitive things happen when we talk about high-dimensional space. Here are some of them:
    - If you pick a random point in a unit square, then there's only 0.4% chance it will be located less than 0.001 form a border. But in a 10000-dimensional unit hypercube, the chance rises to 99.999999%. Most points in a high-dimensional hypercube are very close to the borders.
    - Say you pick to 2 random points in a unit square, then the average distance of them is about 0.52. If you pick in a 3D cube, then the number is roughly 0.66. But if you consider an 1,000,000-dimensional unit hypercube, then the average distance is about 408.25 (roughly $\sqrt{\frac{10000}{6}}$)!
    - This means high-dimensional dataset is at risk of being very sparse, as most training instances will be so far away from each other.
    - This also means a new instance will likely be far away from any training instances, thus making predictions less reliable on lower dimensions, as they will be based on much larger extrapolations.
    - In short,the more dimensions the data is, the higher the risk of overfitting.
- In theory, we could tackle the curse of dimensionality by adding more training data instances to reach a sufficient density of training instances. Unfortunately, the number of training instances required increase exponentially with the number of dimensions. With just 100 features - far lower than the MNIST problem - all ranging from 0 to 1, you need more training instances ($10^100$) than the number of atoms in observable universe ($10^78$ to $10^82$) to ensure that there's always an instance within 0.1 from any instance, assuming they are uniformly distributed across all dimensions.

# Main approaches for Dimensionality Reduction

- There are two main approaches to reducing dimensionality: Projection and manifold learning.

## Projection

- In most real-world problems, training instances does not spread ou uniformly across all dimensions. Many features are almost constant, while many others are highly correlated (as discussed earlier in MNIST). 
- As a result, most instances lie within a much lower-dimensional subspace of the original high-dimensional space.
- You can look at the first plot in the learning notebook to get a grasp of this idea:
    - Training instances are small spheres.
    - You can notice that all of them lie near a plane: This is a lower-dimensional (2D) subspace of a higher-dimensional (3D) space. If you project every instance perpendicularly onto this subspace, which are represented as red dashed lines connect the instances to the plane, we get a new training dataset, plotted in the next plot.
- Congrats, we have just reduce the dataset's dimensionality from 3D to 2D.
- Note how the axes now turn into $z_1$ and $z_2$: they are the coordinates of the projections on the plane.

## Manifold learning

- However, projections is not an one-size-fit-all solution to all dimensional reduction problems. In many cases. the subspace may twist and turn, such the famous Swiss roll toy dataset, represented in the first image.
- Simply projecting the whole dataset onto a plane (e.g. by dropping $x_3$) would squash different layers of the Swiss roll together, result in a really hard to make sense of graph in the left plot. What we want is to unroll it instead and obtain the 2D dataset in the right plot.
- The Swiss roll is an example of a manifold. Put simply, a 2D manifold is a 2D shape that can be bent and twisted in a higher-dimensional space.
- More generally, a d-dimensional manifold is a part of an n-dimensional space (where d < n) and resembles a a-dimensional space locally. In the case of Swiss roll, d = 2 and n = 3: it resembles a 2D space, but rolled in a 3D space.
- Many manifold reduction algorithms work by modeling the manifold which training instances lies on, this is called *manifold learning*. It is backed by the *manifold assumption*, which is also called the *manifold hypothesis*, which holds that most real-world high-dimensional dataset lie close to a much smaller-dimensional manifold. This assumption is very often empirically (saw a lot in practice) observed.
- Once again, take MNIST as an example. There are lots of similarities between all handwritten digit images: They are connected by a couple of lines, the borders are white and the images are more or less centered. If you create images by randomly pick the intensities of each pixel, there is a ridiculously low chance you will obtain a recognizable handwritten digit.
- In other words, the degree of freedom you have when creating a digit image is dramatically lower than the degrees of freedom available to you when you create any image you like.
- These constraint tend to squeeze the training dataset into a much smaller-dimensional manifold.
- The manifold assumption is often accompanied by another implicit assumption: The task at hand (e.g. classification or regression) will be simpler if expressed in a lower-dimensional manifold.
- For example, look at the next plot in the learning notebook. The Swiss roll is now split into 2 classes, but if you consider the 3D plot, then the decision boundary is fairly complex, but in the 2D unrolled manifold space, the decision boundary is straight forward (it's just a straight line).
- However, this implicit assumption does not always hold. For example, consider the next image:
    - The decision boundary is located at $x_1=5$. This decision boundary is very simple in 3D space, which is just a plane.
    - But if you unrolled the dataset, then you the decision boundary is much more complex in the manifold. In fact, you need 4 consecutive lines to correctly divided the dataset into 2 classes.
- In short, reducing the dimensionality of your dataset before training will usually speed up training, but it does not necessarily lead to a better or simpler solution; it all depends on the dataset.

# PCA

- *Principal component analysis* (PCA) is by far the most popular dimensionality reduction algorithm. First, it defines the hyperplane that lies closest to the data, then it projects the data onto that hyperplane, as we did earlier.

## Preserving the Variance

- Before you project the training set into a hyperplane, you need to have the right hyperplane first.
- Look at the image in the learning notebook:
    - On the left is the dataset and 3 different axes.
    - On the right is the projection of the dataset on each axes.
    - As you can see, the projection onto the solid line preserves the highest variance.
    - The projection onto the dotted preserves the lowest variance.
    - And the projection onto the dashed line preserves an intermedia amount of variance.
- It is reasonable to choose the axes that preserves the maximum amount of variance, as it most likely will lose the least amount of information.
- Another way to think about it is we want to find the axis that minimizes the mean squared distance between the original data and its projection onto that axis.
- That is the idea behind PCA: We want to reduce the dimensionality of the dataset, while retaining as much variation from the original dataset as possible.

# Principal Components

- PCA finds the axis that has largest amount of variance in the training set. In the image in the previous part, it is the solid line.
- Next, PCA finds a second axis, orthogonal to the first axis, and accounts for the most amount of remaining variance. In our example, we have no other choice: It's the dotted line.
- However, if it is a higher-dimensional dataset, then PCA will continue to find a third axis, orthogonal to both previous axis and accounts for the most amount of remaining variance, then a fourth, a fifth, and so on - as many axes as the number of dimensions in the dataset.
- The $i^{th}$ axis is called the *$i^{th}$ principal component* (PC) of the dataset.
- In the **Preserving the Variance** part, the first PC is the axis on which **$c_1$** lies, and the second PC is the axis on which **$c_2$** lies.
- In the **Projection** part, the first two PCs are on the projection plane, while the third PC is the axis orthogonal to the plane. After the projection, the first PC corresponds to the $z_1$ axis and the second PC corresponds to the $z_2$ axis.
- For each principal component, PCA finds a zero-centered unit vector pointing in the direction of the PC. 
- Since two opposing unit vectors lie on the same axis, the direction of the unit vector is not stable: If you change the data ever so slightly and run PCA again, the unit vector may point in the opposite direction compared to the original vector. However, they will generally end up in the same axis.
- In some cases, a pair of unit vectors may even rotate or swap (if the variance along these two axes are very close), but in general the plane they define will remain the same.
- How do we find the principal component of a dataset? Fortunately, our old friend *singular value decomposition* (SVD), which is a standard matrix factorization technique, can help us decompose the training set matrix X into the matrix multiplication of three matrices $\textbf{U}, \Sigma, \textbf{V}^T$, where **$V$** is the vector contains the unit vectors that define all the principal components that we want to find:
    $$\textbf{V} = (c_1^T, c_2^T, \dots, c_n^T)$$
- PCA assumes that the dataset in centered around the origin. Scikit-learn's PCA automatically takes care of this for us, but if you implement PCA yourself, or use other libraries, don't forget to center the data first.

## Projecting Down to d Dimensions

- Once you have defined all the principal components, you can reduce the dimensionality of the dataset down to d dimensions by projecting the dataset onto the hyperplane defined by the first d principal components.
- Selecting this hyperplane ensures that the projection preserves the maximum amount of variance.
- In the **Projection** part, the 3D dataset is projected down onto the 2D plane defined by the first two principal components, preserving a lot of the dataset's variance. As a result, the 2D projection looks very similar to the original 3D dataset.
- To project the training set onto the hyperplane and obtain a reduced dataset $\textbf{X}_{d-\text{proj}}$ of dimensionality $d$, simply apply the matrix multiplication of the training set matrix $\textbf{X}$ by the matrix $\textbf{W}_d$, defined as the matrix containing the first $d$ columns of $\textbf{V}$:
    $$\textbf{X}_{d-\text{proj}} = \textbf{X}\textbf{W}_d$$

## Using Scikit-learn

- Scikit-learn's `PCA` class uses SVD to implement PCA, juts like we did earlier.
- After fitting the `PCA` transformer to the dataset, its `component_` attribute holds the transpose of $\textbf{W}_d$: It contains one row for each of the first d principal components.

## Explained Variance Ratio

- 
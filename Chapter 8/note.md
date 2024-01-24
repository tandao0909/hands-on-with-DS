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

- There are teo main approaches to reducing dimensionality: Projection and manifold learning.

## Projection

- In most real-world problems, training instances does not spread ou uniformly across all dimensions. Many features are almost constant, while many others are highly correlated (as discussed earlier in MNIST). 
- As a result, most instances lie within a much lower-dimensional subspace of the original high-dimensional space.
- You can look at the first plot in the learning notebook to get a grasp of this idea:
    - Training instances are small spheres.
    - You can notice that all of them lie near a plane: This is a lower-dimensional (2D) subspace of a higher-dimensional (3D) space. If you project every instance perpendicularly onto this subspace, which are represented as red dashed lines connect the instances to the plane, we get a new training dataset, plotted in the next plot.
- Congrats, we have just reduce the dataset's dimensionality from 3D to 2D.
-Note how the axes now turn into $z_1$ and $z_2$: they are the coordinates of the projections on the plane.
    
1. The main motivations are:
- Help the model learns faster and maybe remove noise form the data, hence makes the model performs better.
- To visualize the data and gains some insight about it. Also helps explain the insights to your stakeholder.
- To compress the data.
The main drawbacks are:
- Some data is lost, which can hurt the model's performance.
- This adds complexity to your model's structure.
- It can be hard to interpret.
- It can be computationally expensive.
2. The curse of dimensionality is high-dimensional objects behave very different from our intuitive thinking in low-dimensional space. One of them is instances lie very far apart in high-dimensional space, makes our model hard to find any patterns.
3. Once a dataset's dimensionality has been reduced, it mostly impossible to convert the dataset back to its original form. This is because some information is lost during the compression. Moreover, even some algorithms has a reverse function to recover the dataset to a similar form of its original one, most of them  don't.
4. PCA can be used to reduce the dimensionality of a nonlinear dataset, it will get rid of unnecessary features. But if it turns out that every features is important, for example the Swiss roll dataset, then PCA won't help us at all. You want to unroll, not squash it onto a plane.
5. The answer is it depends on the dataset. If the data is perfectly aligned in an axis, then PCA will reduce the dataset down to that one axis, remove all other 999 dimensions. But if the data is uniformly distributed across all dimensions, then PCA will keeps $95% \times 1000 = 950$ dimensions and get rid of 50 dimensions. So the answer is it depends.
6. Here are the desired situation for each of them:
    - Regular PCA: This is the default. Regular PCA will find the correct global optimal solution, but it has 2 downsides. First, it has to fit the whole dataset into memory, so if your dataset requires a lot of memory, regular PCA is not a choice. Second, it has a very large computational complexity, $O(n^3) + O(m \times n^2)$, which means it will be very slow if the data has too many features.
    - Randomized PCA: If you just want to reduce the dataset's dimensionality considerably (you don't need the exact answer, precisely correct is fine) and the dataset fit into the memory, then you should use randomized PCA, as it's much faster than regular PCA. Its computational complexity is $O(d^3) + O(m \times d^2)$, where d is the number of principal components we want to find.
    - Incremental PCA: This tackles the first problem in regular PCA by feeding the model batches of instances, instead of the whole datasets at once. However, note that it's slower than regular PCA, so if the dataset fit in the memory, you should prefer regular PCA.
    - Randomized projection: This is great for very high-dimensional datasets.
7. 
- We can say a dimensionality reduction algorithm is good if it eliminates a lot of dimensions from the dataset without losing too much information. 
- One way to measure this is to apply the inverse transformation and measure the reconstruction error. 
- However, not every dimensionality reduction algorithm has a reverse transformation.
- Alternatively, if you use dimensionality reduction as an preprocessing step before another ML algorithm (e.g. a Linear Regression), then you can simply compare the performance of that algorithm: If dimensionality doesn't lose so much information, then the algorithm will perform just as well as being trained on the full dataset.
8.
- Yes, it makes sense to chain two different dimensionality reduction.
- For example, you can use PCA or Random projection to get rid of a large number of useless dimensions, then applying a much slower (but more precise) dimensionality reduction algorithm, such as LLE. This two-step approach will have an answer close to using LLE only, but in just a fraction of time.
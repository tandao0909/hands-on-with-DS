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
6. 
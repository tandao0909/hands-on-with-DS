1. Because Decision tree is approximately balanced binary tree, so there depth is roughly $\log_2(m) =  \log_2(10^9) \approx 20$. We need to approximate up, because Decision Trees are not perfectly balanced binary tree, so the answer is about 21.
2. 
- A node's Gini impurity is generally lower than its parents. This is because the fact that the CART training algorithm tries to split each node in a way that minimize the weighted sum of all its child's impurity. 
- However, this is not guarantee that a node's impurity is higher than every child's impurity. We can have a node which has impurity higher than its parent, as long as this increase results in a lower overall in the weighted sum of all child's impurity.
- For example, consider a node containing 4 instances of class A and 1 instance of class B. Suppose the data is one-dimensional and the data is lined up in this order: A, A, A, B, A. Its Gini impurity is $1-(1/5)^2-(4/5)^2=0.32$ You can work through the algorithm yourself to see it will split the node in such a way that one node consists of three instances: A, A, A and the other node consists of 2 instances: B, A. The latter node has the impurity is $1-\left(1/2\right)^2-\left(1/2\right)^2 = 0.5$, which is higher than the parent's Gini impurity. However, the former node is pure, so the weighted sum of Gini impurity in total is $2/5\times 0.5+3/5 \times 0=0.2$, which is lower than the parent's Gini impurity.
3. If a decision tree is overfitting the training set, it will be a good idea to reduce to `max_depth` hyperparameter. As it will regularize the model.
4. If the decision tree is underfitting the training set, it is not a good idea to scale the features. The reason is scaling the model does not affect the decision tree at all, so it is pointless to do so.
5. 
- The training time complexity of Decision tree is $O(n\times m\log_2(m))$, with $m=10^6$. 
- We have $n\times m\log_2(m)=1$ and want to calculate $n \times 10m \log_2(10 m)$. So we have:
    $$n \times 10m \log_2(10m) = 10 \times n \times m \log_2(m) \times \frac{\log_2(10m)}{\log_2(m)} = 10 \times \frac{\log_2(10m)}{\log_2(m)}$$
- Note that $m=10^6$, so $\frac{\log_2(10m)}{\log_2(m)} = \frac{7}{6}$$, so in the end, the approximation of training time will be $10 \times \frac{7}{6} \approx 11.67$ hours.
6. 
- The training time complexity of Decision tree is $O(n\times m\log_2(m))$. 
- So if we double the number of features, it will take twice amount of time to complete training.
- So in the end, the approximation of training time if we double the number of features will be 2 hours.

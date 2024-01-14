- Decision tree are versatile algorithm that can work on both classification and regression tasks, even with multioutput tasks.
- They are powerful algorithms, capable of fitting very complex dataset. For example, in chapter 2, we trained a Decision Tree model to fit the California housing dataset and fit it perfectly (actually, overfit it).
- Decision Trees are also the fundamental components of Random Forests (will discuss in Chapter 7), which are among the most powerful machine learning algorithms available today.

# Making Predictions

- The way make predictions is similar to walk through several if statement. Suppose you find an iris flower and want to classify it:
    - You start at the root node: This node ask whether the petal length is smaller or equal to 2.45 cm. If yes, you move to the root's left node (depth 1, left). In this case, this is a leaf node (i.e. it has no child nodes), so it does not ask any question: Just look at the predicted class of that node and the decision tree predicts that your flower is an *Iris Setosa* (class=setosa).
    - If no, then you move down the right node (depth 1, right). This is not a leaf node, but a spilt node, so it asks another question: Is the petal width smaller or equal to 1.75 cm? If yes, then the model thinks the flower most likely is an *Iris Versicolor*. If not, then the model thinks the flower is most likely to be an *Iris Virginica*.
- One of the advantages of decision trees is they require very little data preparation. In fact, they don't need to center or scale the features at all.
- A node's samples attribute counts how many training instances it applies to. For example, there are 100 training instances have petal width > 2.45 cm (depth 1, right), anh of these 100, there are 46 training instances have petal width > 1.75 cm (depth 2, right).
- A node's value attribute tell us how many training instances in each class it applies to. For example, the bottom-left node (depth 2 , left) applies to 0 Iris Setosa, 49 Iris Versicolor, 5 Iris Virginica.
- A node's gini attribute measure its *Gini impurity*: A node is *pure* (gini=0) if and only if all training instances it applies to is belong to the same class. It talks about the probability you classify a point incorrectly. For example, the first left node (depth 1, left) applies only to Iris Setosa class, make its gini=0. 
- The formula for computing the Gini impurity $G_i$ of i-th node  is:
    $$G_i = 1 - \sum_{k=1}^n \rho_{i,k}^2$$
    where:
    - $G_i$ is the Gini impurity value of i-th node.
    - n is the number of classes.
    - $\rho_{i, k}$ is the ratio of instances belongs to class k, among all the training instances in the i-th node.
- For example, in bottom-right node, there are 0 Iris Setosa, 1 Iris Versicolor, 45 Iris Virginica and 46 training instances in total. So its Gini impurity is:
    $$1 - \frac{0}{46} - \frac{1}{46} - \frac{45}{46} \approx 0.043$$
- Scikit-learn uses the CART algorithm, which only produces binary tree. It means the split nodes always have exactly 2 children nodes. However, there are other algorithms, such as ID3, can produce decision trees with nodes that have more than 2 children.
- The plots in the learning notebook show the decision tree's decision boundaries:
    - The thick vertical line represents the decision boundary of the root node (depth-0): petal length = 2.45 cm. Since the left are is pure, it can't be split further. 
    - However, the right area is impure, so the depth-1 right node splits it at petal width = 1.75 cm (represented by the horizontal dashed line)
    - Because we set `max_depth=2`, the algorithm stops right there. If you set `max_depth=3` instead, then the model will add another decision boundaries (represented by two vertical dotted lines).
- The tree structure, including all the information shown in the .dot file, can be accessed via the classifier's `tree_` attribute. See the notebook for more detail.

## Model Interpretation: White Box Versus Black Box

- Decision trees are intuitive, and their result are easy to interpret. We can follow along its algorithm and understand how each feature impacts each other and how they influence the final result. 
- Such models are called *white box models*.
- In contrast, random forests and neural networks are generally considered *black box models*. They make great decisions, and you can verify each calculation that they perform to come to the result. Nevertheless, it is usually hard to understand why that calculation were made. 
- For example, you train a neural network to recognize people in images. Say it recognizes a particular person in an image, how does it know that blob of color, which is to computer is just a blob of number, is that person? Did it recognize the eyes? The nose? The mouth? Or perhaps even the couch they are sitting on?
- Conversely, decision trees have nice, simple classification rules that can be used to learn about the data (e.g. for flower classification).
- The field of *interpretable ML* aims at creating ML systems that can be explained in a way that human can understand. This is important in many domains, for example, to ensure the system makes fair decisions.

# Estimating Class Probabilities

- A decision tree can also estimates the probability that an instance belongs to a particular class k.
- First, it traverses the tree to find the node the instance belongs to, then it returns the ratio of class k in that node.
- For example, your instance is a flower whose petals 5 cm long and 1.5 cm wide. Its node is the left node, depth 2, so the Decision Tree will output the following probabilities: 0% for Iris Setosa (0/54), 90.7% for Iris Versicolor (49/54) and 9.3% for Iris Virginica (5/54). 
- If you ask the model to predict the class, it will output class 1 (Iris Versicolor) because it has the highest possibility.
- Notice that the estimated probabilities will be identical everywhere in the sample region. For example, If you ask the model to predict a new flower whose petals 6 cm long and 1.5 cm wide, then the estimated probabilities and therefore, the predicted class will be the same as our previous instance (even though it more likely to be an *Iris virginia*) in this case.

# The CART Training Algorithm

- Scikit-learn uses the *Classification and Regression Tree* (CART) to train Decision Trees (also called "growing" tree).
- The algorithm first splits the training set into two subsets using a single feature k and a threshold $t_k$ (e.g. "petal length $\leq$ 2.45 cm"). How does it choose k and $t_k$? It searches for the pair (k, $t_k$) that produces the purest subsets.
- This is the cost function the model try to minimize:
    $$J(k, t_k) = \frac{m_{\text{left}}}{m}G_{\text{left}}+\frac{m_{\text{right}}}{m}G_{\text{right}}$$
    where $G_{\text{left/right}}$ measures the impurity of the left/right subset and $m_{\text{left/right}}$ is the number of instances in the left/right subset.
- After the CART algorithm splits the training set into 2 subsets, it further splits the training subsets using the sam logic, and the sub-subsets, and so on recursively. It stops recursing once it reaches the maximum depth (define by the `max_depth` hyperparameter), or it cannot find a split that reduces the impurity. 
- There are a few hyperparameters you can use to control additional stopping conditions: `min_sample_split`, `min_sample_leaf`, `min_weight__fraction_leaf`, `max_leaf_nodes`.
- The CART is a greedy algorithm: It greedily searches for the most effective split and the top, and continues the process for all the children nodes. It does not check whether or not the spilt will lead to the lowest impurity several levels down. A greedy algorithm will find the suboptimal solution, not necessary the optimal one. However, sometimes suboptimal is reasonably good.
- Unfortunately, finding the optimal decision tree is known to be an NP-complete problem: It requires $O(\exp(m))$ time, making the problem intractable, even for small training sets. That is why we must settle for just a "good enough" solution.

# Computational Complexity

- Making predictions requires traversing the Decision Tree from the root to a leaf. Decision Trees are generally approximated balanced, so traversing it requires going through roughly the depth of Decision Tree. Because each node only check for a feature of the instance, going through each level effectively divide the training set by 2. So in general, the prediction complexity is $O(\log_2(m))=O(\ln(m))$, independent of the number of features. That is the reason why predictions are very fast, even with large training sets.
- The training algorithm compares all features (or less if `max_features` is set) on all the training instances at each nodes. This results in a training complexity time of $O(n \times m \log_2(m))$. Here is a breakdown why:
    - We will sort the training data, which takes $O(n \log_n(n))$ time (Tim sort don't work here because data can be floating point number).
    - After sorting, we need to traverse the sorted data to find the right threshold. This takes $O(n)$ time.
    - We repeat the above process for all n dimensions. So in total, the time complexity is $O(n \times (m\log_2(m) + m)) = O(n \times m\log_2(m))$
- Space complexity is O(nodes), because we need to store the whole tree in the memory.
- For small training set, you can speed up the process by presorting the data, but doing that slow down training considerably for large training sets.

# Gini impurity or Entropy?

- By default, Scikit-learn use the Gini impurity measure, but you can change to entropy by setting the `criterion` hyperparameter to "entropy". 
- The concept of entropy comes from thermodynamic, as a measure of disorder: entropy approaches zero if the element are grouped together and stay still.
- Other domains uses this term too, for example in Shannon's information theory, where it measures the average information content in a message, as we discussed in chapter 4. Entropy is zero when all the messages are identical.
- In machine learning, entropy is frequently used as an impurity measure: a set's entropy is zero if all element in that set is in the same class.
- The formula of entropy is:
    $$H_i = - \sum_{k=1, \rho_{i, k}\neq 0}^n \rho_{i, k} \log_2(\rho_{i, k})$$
- For example, in left node, depth 2, the entropy is $-49/54\log_2(49/54) - 5/54\log_2(5/54) \approx 0.445$.
- So in the end, what should you choose? The truth is, they lead to similar trees. Gini impurity tends to compute faster, so it is the default.
- However if they differ, Gini impurity tends to isolate the most frequent class in its own branch, while entropy tends to produce slightly more balanced tree.

# Regularization Hyperparameters

- Decision Trees make very few assumptions about the data (as opposed to linear model, which assume that the data is linear, for example). 
- If left unconstrained, the tree structure will adapt itself to the training data, fitting it very closely. So close, to the point that it usually overfit the training data.
- Such a model is often called a *nonparametric model*, not because it does not have any parameters (it usually has a lot) but because the number of parameters is not determined prior to training (i.e. we can have as many more parameters as we want). Some examples are Decision Tree, K-nearest Neighbors, Kernel Density Estimation.
- In contrast, a *parametric model*, for example a linear model, has predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting (but increasing the risk of underfitting).
- To avoid overfitting the data, you need to restrict the decision tree's freedom during training. In other word, regularizing it. 
- The regularization hyperparameters depend on the used algorithms, but in general, you can restrict the maximum depth of the decision tree. In Scikit-learn, you control it via the `max_depth` hyperparameter. The default is None, which means unlimited. Reducing `max_depth` will regularize the model, thus reduce the risk of overfitting.
- The `DecisionTreeClassifier` has a few more hyperparameters to restrict the shape of the decision tree:
    - `max_features`: Maximum number of features to evaluate when splitting at each node.
    - `max_leaf_nodes`: Maximum number of leaf nodes.
    - `min_samples_split`: Minimum number of samples a node must have before it can be split
    - `min_sample_leaf`: Minimum number of samples a leaf must have to be created.
    - `min_weight_fraction_leaf`: Same as `min_sample_leaf` but expressed as fraction of the total number of weighted instance. Weighted instances are instances that has a weight assign to it to make it more important. For example, if your dataset has a minority class, you can assign higher weights to make the model pay more attention to it. In our context, this means that for a leaf node to be created, the sum of the weights of its samples must be at least 20% of the total sum of all sample weights in the entire dataset.
    - `min_impurity_decrease`: Minimum amount of decreasing impurity guaranteed if splitting a node.
- Increasing `min_*` or decreasing `max_*` hyperparameters will regularize the model. 
- Other algorithms work by allowing the decision tree as it want, then *pruning* (deleting) unnecessary nodes.
- A node whose children are all leaf nodes is considered unnecessary if the purity improvement it provides is not statically significant. Standard statistical tests, such the $\chi^2$ test (chi-squared tests), are used to estimate the probability that the improvement is just by chance (which is call *the null hypothesis*). If this probability, called the *p-value*, is not higher than a given threshold (typically 5%, controlled by a hyperparameter), then the node is considered unnecessary and all of its children are deleted. The pruning process continues until all the unnecessary nodes have been pruned.

# Regression

- Decision tree can also perform regression tasks. 
- The only different between it and its classification version is instead of predict a class, it predicts a value.
- Look at the tree in the training notebook. It looks very similar to the classification tree you saw earlier.
- Suppose you want to make prediction about a new instance with $x_1=0.2$:
    - You start at the root node and ask whether $x_1<=-0.303$. It's not, so you go to its right child node.
    - You continue with the node, and continue to ask whether $x_1<=0.272$. It is, so you go its left child node.
    - This is the leaf node, so you stop and the model predicts the value is 0.028.
    - This prediction value is the average target value of the 110 training instances associated with this leaf node, and it results in a mean squared error of 0.028 over these 110 training instances.
- Look at the first image in the learning notebook (I don't count on the image of tree). Notice how the predicted value for each region is always the average value of all the instances in that region. The algorithm tries to split each region in a way that make most training instances as close as possible to that predicted value.
- The CART algorithm works the same as earlier, expect that instead of minimizing the Gini impurity, it now try to minimize the MSE:
    $$J(k, t_k) = \frac{m_{\text{left}}}{m}\text{MSE}_{\text{left}} + \frac{m_{\text{right}}}{m}\text{MSE}_{\text{right}}$$
    where:
    - k is the index of the feature
    - $t_k$ is the threshold associated with the feature k.
    - $\text{MSE}_{\text{node}} = \sum\limits_{i \in \text{node}}\left(\hat{y}_{\text{node}}-y^{(i)}\right)^2$
    - $\hat{y}_{\text{node}}=\frac{\sum\limits_{i \in \text{node}}y^{(i)}}{m_\text{node}}$
- Similar to classification, decision trees are also prone to overfitting in regression tasks, too.
- Look at the second image in the learning notebook:
    - In the left plot is when we train a decision tree without any restrictions. Its prediction is badly overfitting the training dataset.
    - The right plot represents a decision tree when we add just a regularization `min_samples_leaf=10`. Clearly, now the model can generalize better.

# Sensitivity to Axis Orientation

- Decision trees have many upsides: They ar easy to use, to understand and to interpret, also powerful and versatile.
- However, decision trees have many downside need to be consider when training it, as well.
- You may have noticed that decision trees love orthogonal decision boundaries (i.e. all splits are perpendicular to an axis), which makes them sensitive to dataset's orientation.
- Look at the first image in the learning notebook. We work on a simple linearly separable dataset:
    - In the left plot, the decision tree separates the training set with ease.
    - However, when we rotate the dataset by $45^\circ$, now the decision tree can still separate the training set, but it looks unnecessary convoluted (unnecessary complicated).
- A way to help model tackling this limit is to scale the data, then applying a principal component analysis transformation. We will talk about PCA later in chapter 8, for now, you just need to know that it rotates the dataset in a way that reduces the corelation between features, which often (not always) make training easier for decision trees.

# Decision Trees Have a High Variance

- More generally, the problem with decision trees is they have a quite high variance: Small changes to the hyperparameters may leads to very different models.
- In fact, since the training algorithm used by Scikit-learn is stochastic-it randomly selects a feature to evaluate at eac node-even if you retrain in the same model, you can end up with very different model(expect you set the `random_state` hyperparameter).
- Look at the image in the learning notebook, it looks very different from the decision tree we trained in the *Making Predictions* chapter.
- Luckily, you can improve the performance of decision trees significantly by training multiple of them and averaging out the predictions over all of them. This method is an *ensemble* of decision trees, which named a *random forest*, one of the most powerful types of models available today, which we will talk about in the next chapter.
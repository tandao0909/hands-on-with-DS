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
    $$1 - \frac{0}{46} - \frac{1}{46} - frac{45}{46} \approx 0.043$$
- Scikit-learn uses the CART algorithm, which only produces binary tree. It means the split nodes always have exactly 2 children nodes. However, there are other algorithms, such as ID3, can produce decision trees with nodes that have more than 2 children.
- The plots in the learning notebook show the decision tree's decision boundaries:
    - The thick vertical line represents the decision boundary of the root node (depth-0): petal length = 2.45 cm. Since the left are is pure, it can't be split further. 
    - However, the right area is impure, so the depth-1 right node splits it at petal width = 1.75 cm (represented by the horizontal dashed line)
    - Because we set `max_depth=2`, the algorithm stops right there. If you set `max_depth=3` instead, then the model will add another decision boundaries (represented by two vertical dotted lines).
- The tree structure, including all the information shown in the .dot file, can be accessed via the classifier's `tree_` attribute. See the notebook for more detail.

## Model Interpretation: White Box Versus Black Box

- Decision trees are intuitive, and their result are easy to interpret. We can follow along its algorithm and understand how each feature impacts each other and how they influence the final result. 
- Such models are called *white box m\mathbf{a}^T\mathbf{x} \leq bodels*.
- In contrast, random forests and neural networks are generally considered *black box models*. They make great decisions, and you can verify each calculation that they perform to come to the result. Nevertheless, it is usually hard to understand why that calculation were made. 
- For example, you train a neural network to recognize people in images. Say it recognizes a particular person in an image, how does it know that blob of color, which is to computer is just a blob of number, is that person? Did it recognize the eyes? The nose? The mouth? Or perhaps even the couch they are sitting on?
- Conversely, decision trees have nice, simple classification rules that can be used to learn about the data (e.g. for flower classification).
- The field of *interpretable ML* aims at creating ML systems that can be explained in a way that human can understand. This is important in many domains, for example, to ensure the system makes fair decisions.

# Estimating Class Probabilities

- A decision tree can also estimates the probability that an instance belongs to a particular class k.
- First, it traverses the tree to find the node the instance belongs to, then it returns the ratio of class k in that node.
- For example, your instance is a flower whose petals 5 cm long and 1.5 cm wide. Its node is the left node, depth 2, so the Decision Tree will output the following probabilities: 0% for Iris Setosa (0/54), 90.7% for Iris Versicolor (49/54) and 9.3% for Iris Virginica (5/54). 
- If you ask the model to predict the class, it will output class 1 (Iris Versicolor) because it has the highest possibility.

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

- 
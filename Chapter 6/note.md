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
- A node's gini attribute measure its *Gini impurity*: A node is *pure* (gini=0) if and only if all training instances it applies to is belong to the same class. For example, the first left node (depth 1, left) applies only to Iris Setosa class, make its gini=0. 
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
- Suppose you ask a hard question to thousands of random people, then aggregate their answers. 
- In many cases, you will find that this aggregated answer is better than an expert's answer. This is called *the wisdom of the crowd*.
- Similarly, if you aggregate the predictions of a group of predictors (such as classifiers or regressors), then the answer will be better than the best individual predictor.
- A group of predictors is called an *ensemble*; thus, this technique is called *ensemble learning* and an ensemble learning algorithm is called an *ensemble method*.
- An example of an ensemble method is training a group of decision tree classifiers, each of them is trained on a different subset of training set. You then obtain the predictions of all the individual trees, and the class that gets the most votes is the ensemble's prediction (we did this in exercise 11 chapter 6).
- This ensemble method named *random forest*, and despite its simplicity, this is one of the most powerful ML algorithms nowadays.
- As discussed in chapter 2, you will often use ensemble near the end of the project, once you have already built a few good predictors, and combine them into an even better predictor. 
- In fact, the winning solution in most ML competition involve several ensemble methods.
- In this chapter, we will discuss most popular ensemble methods, including *bagging*, *boosting* and *stacking*. We will also talk about Random Forest.

# Voting Classifiers

- Suppose you have trained a few classifiers, each of them achieves about 80% accuracy. You may have a logistic regression classifier, a support vector machine classifier, a random forest classifier, etc.
- A simple way to create an even better model is to aggregate the predictions of each classifier: the class that gets the most votes is the ensemble's prediction. This majority-vote classifier is called a *hard voting* classifier.
- Surprisingly, this voting classifier often has a higher accuracy than the best classifier in the ensemble. 
- In fact, even if each classifier is a *weak learner* (i.e. it only does slightly better than random guessing), te ensemble can still be a *strong learner* (i.e. has a high accuracy), provided there are sufficient number of weak learners and they are sufficiently diverse.
- Look at the plot in the learning notebook:
    - Suppose we have an unfair coin, which has 51% of coming up as heads and 49% coming up as tails.
    - If you toss the coin 1000 times, then the probability to obtain the majority of heads is about 75%.
    - If you toss the coin 10000 times, then the probability to obtain the majority of heads is about 97%.
    - If you toss the coin a very large amount of time, then the ratio we obtain head in total will converge to 51%. This is the *law of large numbers*. Hence, the more you toss the coin, the higher the probability you obtain a majority of heads.
    - The plot shows 10 series of biased coin tosses, each series consists of 10000 tosses. 
    - You can see that as the number of tosses increases, the ratio of heads approaches 51%.
    - Eventually, after 10000 tosses, all 10 series ends up so close to 51% that they stay consistently above the 50% line.
- Similarly, suppose you train 1000 classifiers, each of them predicts correct only 51%. If you predict the majority voted class, you can hope for up to 75% accuracy. 
- However, this is only true if all classifiers are perfectly independent, making uncorrelated errors, which is clearly not the case because they all learn from the same dataset. They are likely to make them same types of mistakes, so there will be many majority votes for the wrong class, reducing the ensemble's accuracy. 
- Ensemble methods work best when the predictors are as independent from one another as possible. One way to achieve this is to train them using very different algorithms. This increases the chance that they will make very different types of errors, thus increases the ensemble's accuracy.
- If each classifier can estimate the probability of each class, then you can tell Scikit-lean to predict the class with the highest probability, which is calculated by averaging over all the individual estimators. This is called *soft voting*.
- This usually performs better than hard voting, because it gives more weight to highly confident votes.

# Bagging and Pasting

- One way to get a diverse set of predictors is to use very different training algorithms.
- Another way is to use the same algorithm but train it on different subset of the training dataset.
- There are two ways of choosing the subsets from the training dataset:
    - Bagging (short for *bootstrap aggregating*): You choose a card from a deck, write that card down, PUT BACK the card to the deck and draw another card. This is called sampling with replacement.
    - Pasting: You choose a card from a deck, write that card down, THROW AWAY that card and draw another card. This is called sampling without replacement.
- Both bagging and sampling allowed training to be sampled several times across multiple predictors, but only bagging allow a predictor see an instance multiple times.
- Once all predictors are trained, the ensemble can make a predictions using a *statistical mode* for classification (i.e. the most frequent prediction, just like hard voting) or the average for regression.
- Each individual has a higher bias (because it's trained on a smaller subset of the training set). However, aggregating reduces both bias and variance.
- In the end, the result is an ensemble that has similar bias but lower variance than a single predictor train on the whole training set.
- After the bootstrapping step, we can train predictors in parallel, via different CPU cores, GPUs or even different servers. Similarly, predictions can be performed parallel. That's one of the reasons bagging and pasting are popular: They scale very well.

# Out-of-Bag Evaluation

- With bagging, some training instances may be sampled several times for any given predictor, while other training instances may not be sampled at all.
- By default, `BaggingClassifier` samples m training instances with replacement (`bootstrap=True`), where m is the size of the training set.
- We can show mathematically that as m increases, only about 63% of the training are sampled on average for each predictor:
    > Every instance has the same chance of being sampled. So each instance has a chance of $1/m$ of being chosen. So each instance has a chance of $1 - 1/m$ of not being chosen. We sample m training instances, or we choose from the training set m times, so the chance of an instance did not be sampled is $\left(1-\frac{1}{m}\right)^m$. As m approach infinity, the chance of an instance did not be sampled approaches $1/e$. So in average, when m increases, there is about $1-1/e \approx 63.2\%$ of the training set being sampled in total.
- The remaining 37% of training instances that are not sampled are called *out-of-bag* (OOB) instances. 
- Note that this 37% is not the same across all predictors.
- A bagging ensemble can be evaluated using OOB instances, without needing a separate validation set.
- In fact, if there are enough predictors, then each instances is likely to be an OOB instance of several predictors, so we can use these predictors to make an ensemble prediction for this instance.

## Random Patches and Random Subspaces

- The `BaggingClassifier` also support sample the features as well.
- Features sampling is controlled by two hyperparameters: `max_features` and `bootstrap_features`. They work the same way as `max_samples` and `bootstrap`, just sampling features instead of sampling instances.
- Then, each predictor will be trained on a random subset of the input features.
- This technique is very useful when you have to deal with high-dimensional data, such as images, as it an speed up training considerably.
- There are two ways Scikit-learn allow you to sample features:
    - Sampling both features and instances, called ***random patches* method**.
    - Sampling features but keeping all the instances, called ***random subspaces* method**: Setting `bootstrap=False` and `max_samples=1.0` but sampling features by setting `bootstrap_features=True` and/or `max_features` to a value smaller than 1.
- Sampling features results in a more predictor diversity, by trading a bit more bias for a lower variance.
- The reason is the data is slightly skewed, so we end up a bit more bias, but the training is more differ across the predictors, so we have a lower variance. Now the estimators are more diverse, so it will generalize better.

## Random Forests

- As we have discussed, random forests is just an ensemble of decision trees, generally trained via the bagging method (or sometimes pasting), usually with `max_samples=1.0`, which means training with the size of training set.
- Instead of using `Bagging Classifier` with `DecisionTreeClassifier` as the base estimator, you can use `RandomForestClassifier`, which is more convenient and optimized for decision trees (there also is `RandomForestRegressor` for regression tasks.)
- The random forest algorithm introduces more randomness to the decision trees: Instead of searching for the very best feature when splitting a node (see chapter 6), we select the best feature from a random subset of features. By default, it samples $\sqrt{n}$ features, where n is the number of features.
- This algorithm results in an even more tree diversity, which (again) trading a bit more bias for a lower variance.

## Extra-Trees

- When you grow a random forest, at each node only a random subset of features is considered when splitting a node.
- You can make it even more random, by instead of trying to find the optimal threshold when splitting, you select a random threshold for each feature. This is called ***extremely randomized trees***, or extra-trees for short.
- Once again, this technique further trade more bias for lower variance.
- This also makes extra-tree classifiers much faster to train than random forest, because selecting the best threshold is usually one of the bottlenecks when training.
- Extra-trees is not guaranteed to perform better than random forest. The only way to know is to try both and compare them using cross-validation.

## Feature Importance

- Another upside of random forest is that we can measure the relative importance of each features.
- Scikit-learn do this by looking at how much the tree nodes which use that features reduce Gini impurity on average.
- Furthermore, this is a weighted average, where each node's weight is the number of training instances associated to it (see chapter 6).
- This is done automatically by Scikit-learn for each features after training, then it scales the features importance such that the sum of them is 1.
- You can access them by using the `feature_importances_` attribute. 
- Random forest is good if you want to gain a quick understanding of how every features actually matters, especially if you want to perform feature selection.

# Boosting

- *Boosting* (originally called *hypothesis boosting*) refers to any ensemble method that can combine several weak learners into a strong learner.
-The way it differ from Bagging and Pasting is instead of training in a parallel manner, we train predictors sequentially in boosting, each step trying to correct its predecessor model. 
- There are many boosting method, but we only focus on *AdaBoost* (short for *adaptive boosting*) and *gradient boosting*. 

## AdaBoost

- One way for a new predictor to correct its predecessor is to pay a bit more attention to the training instances that predecessor underfit.
- This results in new predictors focusing more and more on the hard cases.
- When training an AdaBoost classifier, the algorithm first train a base classifier, then use it to make predictions on the training set. The algorithm then increase the relative weights of misclassified training instances. Then it trains a second classifier using the weighted dataset, then again make predictions on the training set, update the instances weights, and so on.
- This process looks very similarly to gradient descent, but instead of trying to minimize the cost function, AdaBoost adds more predictors to ensemble, gradually makes it better.
- Once the training process is complete, the ensemble makes prediction very similar to bagging or pasting, expect that predictors have different weights, depending on their overall accuracy on the weighted training set.
- However, there is a clear downside of AdaBoost: Because you need to train previous predictors before training any specific predictor, training can be parallelized. As a consequence, AdaBoost does not scale as well as bagging and pasting. 
- This is the break down of the AdaBoost algorithm:
    - Each instance weight is initialized at 1/m.
    - A first predictor is trained, and its weighted error $r_j$ of the j-th predictor is computed by:
    $$r_j = \sum_{i=1, \hat{y_j}^{(i)} \neq y^{(i)}}^m w^{(i)}$$ 
    where $y_j^{(i)}$ is the j-th predictor's prediction for the i-th instance.
    - The predictor's weight $\alpha_j$ is then computed, with $\eta$ is the learning hyperparameter (default is 1). Note that the original AdaBoost does not have a learning rate hyperparameter:
    $$\alpha_j = \eta \log \frac{1-r_j}{r_j}$$
    - The more accurate the predictor is, the higher its weight will be. If it just guess randomly, it weight will be close to 0. However, if it mostly wrong, then its weight will be negative.
    - Next, the AdaBoost updates the instance weights, which boost the weights of misclassified instances.
    $$\forall i = \overline{1, m}, w^{(i)} \leftarrow \begin{cases}
    w^{(i)} \text{, if } \hat{y_j}^{(i)} = y^{(i)} \\
    w^{(i)}\exp(\alpha_j) \text{, if } \hat{y_j}^{(i)} \neq y^{(i)} \\
    \end{cases}$$
    where we calculate based only on j-th predictor, which is the latest predictor we've considered so far.
    - Then all the instance weights are normalized (i.e., divided by $\sum\limits_{i=1}^mw^{(i)}$).
    - Finally, a new predictor is trained using the updated weights, and the whole process repeated: The new predictor's weight is computed, the instance weights are updated, then another predictor is trained, so on.
    - The algorithm stops when the desired number of predictors is reached, or when a perfect predictor is found.
- To make predictions, AdaBoost simply computes the predictions of all predictors and weight them using the predictor weights $\alpha_j$. The predict class is the one that receives the majority of weighted votes:
    $$\hat{y}(\textbf{x}) = \underset{k}{\text{argmax}} \sum_{j=1, \hat{y_j}(\textbf{x})=k}^N \alpha_j$$
    where N is the number of predictors.
- Scikit-learn uses a multiclass version of AdaBoost called SAMME (which stands for *Stagewise Additive Modeling using a Multiclass Exponential loss function*).
- When there are just two classes, SAMME is equivalent to AdaBoost.
- If the predictors can estimate class probabilities (i.e. if they have `predict_proba()` method), Scikit-learn can use a variant of SAMME called SAMME.R (R stands for "Real"), which relies on class probabilities rather than predictions rather than predictions and therefore generally performs better.
- If your AdaBoost ensemble is overfitting, you can try to reduce the number of predictors or more regularized the base predictor.

## Gradient Boosting

- Another popular boosting algorithm is *gradient boosting*.
- Just like AdaBoost, gradient boosting works by sequentially adding predictors to the ensemble, each one try to correct its predecessor.
- However, instead of tweaking the instance weights to make the instance more noticeable, this method try to fit new predictor to the *residual errors* made by previous predictor.
- The residual error can be understand as the delta error, i.e. the different between predicted value, made by the sum of all predicted values from all predictors, and true value.
- In the learning notebook, consider the image with 6 plots:
    - The left column is the predictions of three trees on the residual errors, while the right column represents the ensemble's predictions on the training set.
    - In the first row, there is only one tree in the ensemble, so the ensemble's prediction os the same as the first tree.
    - In the second row, a new tree is trained on the residual errors of the first tree. On the right, you can see that the predictions of the ensemble is now the sum of the first two trees.
    - Similarly, a new tree is trained on the residual errors in the third row. 
    - You can observe that the accuracy of the ensemble is slowly increase as trees are added to the ensemble.
- You can use `GradientBoostingRegressor` class form Scikit-learn to have a more easy time training GBRT ensembles.
- Similar to `RandomForestRegressor` class, it has hyperparameters to control the growth of decision trees (e.g. `min_samples_split`, `max_depth`), as well as hyperparameters to control the ensemble itself, such as the number of trees (`n_estimators`).
- The `learning_rate` hyperparameter scales down the impact of all trees added to the ensemble. When the tree is added to the ensemble, the predictions it made will be multiplied by a factor of `learning_rate`. 
- If you set it to a low value, such as 0.05, then you will need more tree to fit the training set, but the ensemble usually performs better, because it is forced to ignore the "hard-to-learn" instance, which may just be the noise.
- The next image shows two GBRT ensembles trained with different hyperparameters: 
    - The one on the left is our previous model, which does not have enough tree to fit the training set.
    - The one on the right has just right amount of tree and can generalize better. Adding more can lead to overfitting.
- To find the optimal number of trees, you can perform cross-validation, either by using `GridSearchCV` or `RandomizedSearchCV` as usual.
- But there is a more simpler way: You can set the `n_iter_no_change` to an positive integer value, say 5, then if the `GradientBoostingRegressor` find the last 5 added trees don't improve performance at all, it will stop the training. This is our old friend *early stopping*, just a bit more patience: It allows the model for having no progress for a few instances before stopping.
- If you set `n_iter_no_change` too low, it can stop too soon, hence raises the risk of underfitting. If you set it too high, it can learn as many as it want, which has the risk of overfitting.
- In our code, we set the learning rate fairly low and the number of estimators quite high, but because of early stopping, the actual number of estimators is actually much lower.
- When `n_iter_no_change` is set, then the `fit()` method automatically splits the training set into a smaller training set and a validation set. The size of the validation set is controlled by the `validation_fraction` hyperparameter, which is 10% by default. 
- The `tol` hyperparameter describers the tolerance of early stopping, which is default at 0.0001. When the loss is not improving by at least `tol` for `n_iter_no_change` iterations (if set to a number), the training stops.
- The `GradientBoostingRegressor` also supports a `subsample` hyperparameter, which specifies the fraction of training samples to be used for training each tree. For example, if `subsample=0.25`, then each tree is trained on 25% of training set, selected randomly. 
- This technique, once again, trades a higher bias for a lower variance. It also speeds up training noticeably. This is called *stochastic gradient boosting*.

## Histogram-Based Gradient Boosting

- *Histogram-based gradient boosting* works by binning the feature, replacing them with integers. The number of bins if controlled by the `max_bins` hyperparameter, which defaults to 255 and cannot be set higher than that.
- Binning can greatly reduce the number of possible thresholds that the algorithm needs to evaluate. Moreover, working with integers allows the model to use faster and more data-efficient data structure.
- The ways the bins are built also remove the needs to sorting the features before training each tree.
- As the result, This implementation has the training time complexity $O(b\times m)$ instead of $O(n \times m\log(m))$, where b is the number of bins, n is the number of features, m is the number of instances. In practice, this means HGB can be trained hundreds of times faster than regular GBRT on large dataset.
- However, binning also acts as an regularizer: depending ont the dataset, this may help reduce overfitting, or it may cause underfitting.
- Scikit-learn has two class dedicated for HGB: `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier`. They are quite similar to `GradientBoostingRegressor` and `GradientBoostingClassifier` with a few worth noticing differences:
    - Early stopping is automatically activated if the number of instances is greater than 10000 instances. You can turn early stopping always on or always off by setting the `early_stopping` hyperparameter either to `True` or `False`.
    - Subsampling is not supported.
    - `n_estimators` is renamed to `max_iter`
    - The only decision trees's hyperparameters that can be tweaked are `max_leaf_nodes`, `max_depth` and `min-samples_leaf`.
- The HGB classes also have 2 nice features: they support both categorical features and missing values. However, the categorical must be represented as integers ranging from 0 to a number lower than `max_bins`.
You can use a `OrdinalEncoder` for that.
- There are several other optimized implementation of gradient boosting. For example, XGBoost, CatBoost and LightBGM. They are very specialized for gradient boosting, their APIs is very similar to Scikit-learn and they provide many additional features, including GPU acceleration.
- Moreover, the Tensorflow Random Forests library provides optimized implementations of various random forests algorithms, including plain random forests, extra-trees, GBRT and some more.

# Stacking

- The last ensemble we will talk about is *stacking* (short for *stacked generalization*).
- It is based on a simple idea: Instead of using trivial strategy (such as hard voting) to aggregate the predictions of all predictors in ensemble, why don't we train a model to predict it?
- Each of the base predictor will make a prediction and based on these predictions, the final predictor (called a *blender*, or a *meta learner*) will make the final prediction.
- To train the blender, you need to build a training set. You can use `cross_val_predict()` on every predictor in the ensemble to get out-of-samples predictions for each instance in the original training set and use the as the input features to train the blender; and the target and simply be copied from the original training set.
- Regardless of the number of features in the training set, the blending predictor will have one feature per predictor. 
- Once the blender is trained, all the base predictors will be trained again on the whole training set.
- It is actually possible to train several different blenders this way (e.g., one using linear regression, one using random forests regression) to have a whole layer of blenders, then add a new blender on top of that layer to predict the final prediction.
- You may be able to squeeze out some drops of performance using this technique, but it will cost you in both training time and system complexity.
- Scikit-learn provides two classes for stacking ensembles: `StackingClassifier` and `StackingRegressor`.
- To sum up, ensemble methods are versatile, powerful and fairly simple to use. Random forests, AdaBoost and GBRT are among the most first model you should test for most machine learning tasks,a nd they are particularly shine with heterogeneous tabular data. Moreover, as they require very little preprocessing, they're great for having a prototype up and running quickly.
- Lastly, ensemble methods like voting classifiers and stacking classifiers can help increase the performance of your system.
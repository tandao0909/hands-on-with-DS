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
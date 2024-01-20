1. 
- If you have trained five different models on the exact same training dataset, and they all achieve 95% precision, then you can combine them to achieve a higher accuracy to get a better accuracy. 
- You can set a hard voting strategy or soft voting if these models can predict probability.
- The more different these models, the better the performance of the voting estimator.
- It will even be better if they are trained on different subsets of the training dataset, which is the whole point of bagging and pasting.
- But if not, the ensemble model will also effective, as long as the base models are very different.
2. The difference between hard voting and soft voting is:
    - Hard voting is when we estimate the class. Which class get the most votes is the predicted class of the ensemble.
    - Soft voting is when we estimate the class probability. Which class get the highest total probability among all the base predictors is the predicted class of the ensemble.
    - Soft voting allow vote with high confidence has more weight and generally perform better, but requires each base model must be able to predict class probabilities (e.g. the class `SVC` must set `probability=True`).
3. 
- Yes, you can speed up training of a bagging ensemble by distributing it across multiple servers.
- You can also do that for pasting ensembles, because it is very similar to bagging.
- Random forests is just a bagging ensemble of decisions trees, so of course so can do that, too.
- However, boosting ensembles requires the previous model to be trained completely, so it must be trained sequentially. Hence, boosting can't be trained parallelly.
- Stacking ensemble is a bit difficult to talk about. All the predictor in the same layer is completely independent of each others, so we can trained them distributively. However, the predictors in one layer can only be trained after all the predictor in the previous layer have all been trained.
4. 
- Out-of-bag evaluation allows us to gain all the benefits of cross-validation, without having a validation set.
- Out-of-bag evaluates the model on the instances that is not used for training. So we can have an unbiased view about the performance of the model, without having to hold out a validation set.
- Thus, we can have more training instances, and therefore, our models can perform a little bit better.
5. 
- Random forests is trained based on decisions trees, which try to find the best threshold at each node.
- Extra-trees is more random, because we use a random threshold, not the optimal one.
- This random allows extra-trees have a higher bias and a lower variance, allow the base predictors be even more diversity. This help the ensemble model a lot.
- Extra-trees are also faster to train than random forests, because finding the optimal threshold is also one of the bottleneck of the training process.
- However, extra-trees are neither faster or slower than random forests in making predictions.
6. If your AdaBoost underfit the model, then you can tweak some hyperparameters as followed:
    - The learning rate: Change it to be smaller.
    - The number of base predictors: Change it to be bigger.
    - The regularization of base predictors: Change it to be less regularized.
7. If your gradient boosting is overfitting the training dataset, then you should increase the learning rate. You could also use early stopping to reduce the number of base predictors (you probably have too many).
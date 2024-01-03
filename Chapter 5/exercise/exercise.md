1. The fundamental idea behind support vector machine are:
    - We can fit the widest street possible between two classes.
    - In soft margin classification, we try to seek a balance between perfectly separate the training set and have the widest street (i.e. has a few instances on the street).
    - We can apply the kernel to the model to work on nonlinear datasets.
    - The objective of SVMs can be tweak to performs regression tasks and novelty detection.
2. A support vector is an instance that affects the prediction of the SVM. To be more specific:
    - A support vector is an instance that lies on the street, including its border.
    - The decision boundary is determined only by the support vectors.
    - Support vectors are the only instances that matter. You can add, delete, move around any other instances, as long as they stay off the street they don't influence the model at all.
    - Computing the kernelized model only involves support vectors.
3. The reason why we need to scale the inputs is because if we don't scale the input beforehand, the model will try to focus more on the larger scale features and neglect the smaller features.
4. 
- An SVM can output a confidence score by calling its `decision_function()` method. This method calculate based on the distance from the instance to the decision boundary. 
- Originally, SVM can't output a probability, however, you can set `probability=True`, then the methods `predict_proba()` and `predict_log_proba()` will be available.
- Under the hood, SVM will perform a 5-fold cross-validation to train an out-of-sample model. After that, it will try to map the result to a `LogisticRegression`, which can predict the probability. This process will slow down training significantly.
5. Here is how to choose between `LinearSVC`, `SVC` and `SGDClassifier`:
    - `LinearSVC` is overall good and should be try first. Its training time complexity is linear with regard to both the number of instances and the number of features.
    - If you suspect that the dataset is not linearly separable, then you should try `SVC` and apply a kernel to it. However, note that it does not scale well with the number of instance even though scale well with the number of features. Support kernel trick is also a plus.
    - `SGDClassifier` is more suited with too large dataset that cannot fit in the RAM.
    - Because `LinearSVC` performs an optimized algorithm for Linear SVM, it sometimes faster than `SGDClassifier`. Note that `SGDClassifier` is more flexible.
6. When you think the model is underfitting, then you should try to increase both $\gamma$ (gamma) and C, because the higher they are, the more regularized the model.
7. The meaning of a model being *$\epsilon$-insensitive* is as you change the $\epsilon$, the model does not change at all. To be more specific in our SVM Regression context, SVM regression algorithm try to fit as many instances as possible in its margin. If you add more instances within the margin, the model will not change, hence it is sad to be *$\epsilon$-insensitive*.
8. We use the kernel trick to calculate the dot product as if we have applied the kernel, without actually adding it, which speed up training dramatically. 
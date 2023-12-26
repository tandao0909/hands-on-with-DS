# Support Vector Machine

- SVM (abbreviate for Support Vector Machine) is a powerful and versatile Machine Learning model, which can be used for classification, regression and even outlier detection.
- SVM is based on the idea that we can separate two part of data using a line, or a plane, or a hyperplane (a plane in high dimension).
- SVM is just trying to draw a plane to separate two part between training set, which means trying to find a weight vector **w** such that:
    $$w_1x_1+\dots+w_nx_n$$
    drawing a hyperplane that separates well enough, with n is the input dimension.
- In other words, we trying to find a vector of n-dimension to optimize a function go from $\mathbb{R}^n$ to $\mathbb{R}$.
- The name of that line is boundary decision.
- In classification tasks, we want that line not only separate 2 part of the training set, but also has the widest possible width. Hence, the name of it is *large marin classification*.
- A good analogy is thinking of SVC as trying to fitting the widest street possible between the classes.
- Notice that adding more instances "off the street" does not change the boundary decision: It is fully determined (or "supported") by the instances located at the edge of the street. These instances are called "support vectors".

## Comment on plots
This topic is best learned using images:
- In the first images:
    - On the left plot, the green line can't even separate 2 part of training set. The magenta and red line can separate the training set, but we don't think they can generalize well.
    - On the right plot, The black line is the line separate the training set and can generalize well because this line stay as far as possible from both part of training set. The 2 marked data points are the data points closest to the boundary decision, so they are support vectors. 
- In the second images: 
    - On the left plot, because y axis has a much larger scale, it results in a nearly flat line. But as you can see, it won't generalize so well.
    - On the second plot, the line can generalize better.
    - In conclusion, SVMs are sensitive to the scales of features, so remember to scale the features before apply SVM. 

# Soft Margin Classification

- If we apply the rule that every instance must be off the street and on the right side, then it's called *hard margin classification*.
- The problem with hard margin classification is it only works with linearly separable data and extremely sensitive to outliers.


## Comment on plots
- Look at the first two plots in the learning notebook:
    - In the first plot, you can't find a hard margin.
    - In the second plot, because of a single outlier, the decision boundary ends up very different from the previous plot, so it won't generalize well.
- To avoid this issue, just use a more flexible model. The goal is to find a good balance between keeping the street as large as possible and limiting the *margin violations* (i.e. instances that ends up in the middle of the street or on the wrong side). This is called *soft margin classification*.
-  When using Scikit-learn to create an SVM model, we have some hyperparameters to tweak. One of them is C, the regularization hyperparameter. 
- Consider the next two plots in the learning notebook:
    - The left plot has a smaller C, which means it is less regularized.
    - Reducing C makes the street larger, but it also leads to more margin violations.
    - In other words, reducing C results in more instance support the street, so there is less risk of overfitting. Just look at the previous plot, you can see if you consider one more yellow circle, the model is less prone to the outlier (i.e. less overfitting).
    - But if you reduce it too far, you end up with an underfitting model, which is likely our case: The model on the right seems to be generalized better than the left one.
    - A good rule of thumb: Larger C, more risk of overfitting.

## Comments on code
- In the code generating the later plots, here are some clarifies:
    - We want to calculate the weights and biases of the unscaled version of SVM, given the weights and biases of the scaled SVM and the parameter of the scaling function.
    - The biases are calculated by using the decision boundary on the kernel of the scaling function.
    - The weights are computed by dividing the original weights by the the scale parameter.
- Unlike LogisticRegression, LinearSVC doesn't have a predict_proba() method to estimated the classes probabilities. 
- Which means if you uses SVC class instead of LinearSVC and set the *probability* hyperparameter to True, then the model will fit an extra model at the end to estimates probabilities.
- Under the hood, the model using 5-fold cross-validation to generate an out-of-sample prediction for every instance in the training set, then trains a *LogisticRegression* model, so it will slow down the training process considerably. After that, the predict_proba() and predict_log_proba() will be available.
# Nonlinear SVM Classification
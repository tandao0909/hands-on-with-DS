<h1>Confusion matrix<br>(sklearn.metrics.confusion_matrix):</h1>

|       | Predicted Negative | Predicted Positive |
|-------|-------------------|-------------------|
| Actual Negative | TN                | FP                |
| Actual Positive | FN                | TP                |
 
Precision means the ratio of returned objects which is correct. <br>

$$precision = \frac{TP}{TP+FP}$$
<br>

Recall means the ratio of correct objects which are returned over which are should be returned. <br>
Its other name is true positive rate (TPR) or sensitivity. <br>

$$recall = \frac{TP}{TP+FN}$$

Specify means the ratio of negative instances that are correctly classified as negative. <br>
Its other name is true negative rate (TNR). <br>

$$\textit{specify} = \frac{TN}{TN+FN}$$

One way to account both accuracy and recall at the same time is using $F_1$ score. <br>
$F_1$ score is the harmonic mean of the precision and the recall. <br>
$$F_1 = \frac{2}{\frac{1}{precision}+\frac{1}{recall}} = 2 \times \frac{precision \times recall}{precision + recall} = \frac{TP}{TP + \frac{FN + FP}{2}}$$ 

The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR), or recall versus (1 - specify). <br>
Their formula is: 
$$TPR = \frac{TP}{TP + FN}$$
$$FPR = \frac{FP}{FP + TN}$$

There are 2 main ways to train a multiclass classification using binary classification algorithm:
- OvO(One-versus-one): 
    - Train a binary classifier for every pair of class. 
    - In this example: We train one to distinguish 0s and 1s, one to distinguish 0s and 2s, another for 1s and 2s, so on.
    - If there are N classes, we will train N x (N + 1) classifiers in total. 
    - The advantage of this is each classifier only need to train on part of the train set containing the two classes it needs to distinguish.
    - We run an instance through all N x (N + 1) classifiers and select the class win the most duels.
- OvA(One-versus-all): 
    - Train a binary classifier to detect each class, as we have done in the learn notebook to detect class '5'. 
    - In this example: We train one to detect 0s, one to detect 1s and so on.
    - When classify an instance, you select the class with the highest decision function.
<br>
Some algorithms scale poorly with the size of training set. For these algorithm, it is preferred because it is faster to train many classifiers on many small training sets than train few classifiers on large training sets. <br>

Using Confusion Matrix Display, we can see what to improve, for example:
- In my learning notebook, many error is misclassified as 8s, so maybe we should gather more data that looks like 8s (but not 8s) to help the model. 
- Write a preprocessor to help the model (using Scikit-Image, Pillow or OpenCV) to see some patterns, as closed loops, more stand out.
- Engineer more features to help the model, for example: 8s has two loops, other number has only one or no loop.

<br>

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
The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR), or recall versus (1 - specify).
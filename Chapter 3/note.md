<h1>Confusion matrix<br>(sklearn.metrics.confusion_matrix):</h1>

|       ||Predicted|       |
|----   |----|------   |-------|
|       ||Negative |Positive|
|Actual |Negative|TN       |FP     |
|       |Positive |FN       |TP     |
 
Precision means the ratio of returned objects which is correct<br>

precision = $\frac{TP}{TP+FP}$
<br>

Recall means the ratio of correct objects which are returned over which are should be returned. 
Its other name is true positive rate (TPR) or sensitivity.

recall = $\frac{TP}{TP+FN}$

Specify means the ratio of negative instances that are correctly classified as negative.
Its other name is true negative rate (TNR).

specify = $\frac{TN}{TN+FN}$

The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR), or recall versus (1 - specify).
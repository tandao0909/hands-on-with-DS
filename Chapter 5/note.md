# Support Vector Machine

- SVM (abbreviate for Support Vector Machine) is a powerful and versatile Machine Learning model, which can be used for classification, regression and even outlier detection.
- SVM is based on the idea that we can separate two part of data using a line, or a plane, or a hyperplane (a plane in high dimension).
- SVM is just trying to draw a line to separate two part between training set.
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
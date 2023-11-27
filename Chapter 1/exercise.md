1. Machine learning is making a computer better at some task using some data, without explicitly programmed.
2. Four types of applications where ML is a good option:
    - The problem have too many rules and/or too many edge cases. ML can simplify the rule and/or go through the long process of implement the answer.
    - Traditional approach yields no good solution or too complex.
    - Environment has too many fluctuation.
    - Getting insight about complex problems and large amount of data.
3. A labeled training set is a set of data has label for each instance. More detail, the training set includes the solutions.
4. The two most common supervised tasks are regression and classification.
5. Four example for unsupervised tasks:
    - Clustering 
    - Visualization
    - Dimensionality reduction
    - Anomaly detection
6. I would use reinforcement learning to allow a robot to walk in various unknown terrains.
7. There are two possibilities:
    - If there is no label, I would use clustering (unsupervised learning) to segment customer into multiple groups.
    - If there are labels, I would use supervised (classification learning) to classify customer into multiple groups. 
8. I would frame the problem of spam detection as a supervised learning. Because we have two label, either spam or ham.
9. An online learning system is a system where learning is happened incrementally. More specific, the model learns some more little piece of data, without train over again from scratch.
10. Out-of-core learning is train huge datasets which can't fit in one machine's main memory. We load parts of the data, train on that part, then repeat until we've trained all the data.
11. The type of algorithms relies on a similarity measure to make predictions is instance-based learning. 
12. The difference between model hyperparameter and model parameter is:
    - Model hyperparameter: Parameter you defined **before** training. These parameters don't change throughout the training.
    - Model parameter: Parameter you changed **while** training. These parameters is what the algorithms try to tweak to fit your desired outcome.
For example, consider Linear Regression:
    - Hyperparameter: learning rate, regularization parameter.
    - Parameter: Weight and bias
13. - Model-based algorithms search for a model which best describes the data, and try to use it to predict new data.
    - The most common strategy they use to succeed is having a cost function that measures how bad the model is predicting on the training data, plus the penalty for the model for the complexity if regularized.
    - They make predictions by feeding new instances into the model prediction's function, using the parameters found by the learning algorithms.
14. Some of the main challenges in Machine Learning:
    - Lack of data
    - Non-representative data
    - Poor quality data
    - Irrelevant features
    - Too simple model that underfit the data 
    - Too complex model that overfit the data
15. If my model performs great on the training data but generalizes poorly to new instances, it means the model is overfitting.<br>
Three possible solutions:
    - Add more training data.
    - Regularization the model, if using model-based approach.
    - Switch to simpler algorithms. 
16. Test set is the set used to test the model. We use it to see its generalization ability before launching into production.
17. Validation set is the set we hold-out during training to see the performance of current model
18. - Train-dev set is the set that you hold-out from the training data and not train on it (different from validation set, which you can merge back to train set and shuffle).
    - Train-dev is used when there is a risk of mismatch between the train set and the test set. For example, you train on image download from google but test on images taken by your apps.
    - You use it by test the trained model on it. If it preforms poorly, then the reason is bad model; if it perform well on train-dev but badly on test set, then it's data's fault.
19. Then model may overfit the test set, and then would preform poorly on new data in production.
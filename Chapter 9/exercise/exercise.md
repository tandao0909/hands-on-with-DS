1. 
- I would define clustering as a set of instances closed to each other and form a meaningful shape. The "closed" is calculated using some metric, such as the Euclid distance, while "meaningful" may be the global shape of the cluster.
- A few clustering algorithms: K-means, DBSCAN, agglomerative clustering, BIRCH, mean-shift, affinity propagation, spectral clustering, etc.
2. Some of the main applications of clustering algorithms:
- Customer segmentation
- Document segmentation
- Anomaly detection
- Novelty detection
- Semi-supervised learning
- Recommender system
- Feature engineering
- Dimensionality reduction
- Data analysis
3. Three (the exercise only requires two) techniques to select the right number of clusters when using k-means:
- The elbow rule: The inertia is increase as the number of clusters increase. If you plot the inertia versus the number of clusters, it will have the shape of an arm. The position of the elbow is (or closed to) the optimal number fo clusters.
- The silhouette score: You can also plot the silhouette score as a function of the number of clusters. There's usually a peak, and the optimal number of clusters is generally nearby. The silhouette score is the mean of the silhouette coefficient over all the instances. This coefficient varies from 1 for instances that well inside their cluster and far from other clusters, to -1 for instances that are probably assigned to wrong cluster.
- The silhouette diagram: You can also plot the silhouette diagrams and have a more thorough analysis.
4. 
- Label propagation: You cluster a dataset, then in every cluster, most instances has no labels, only a few has. Then you propagate the label of the labelled instances to all the instances belong to the same cluster. That is label propagation.
- You need to do label propagation is because sometimes, while you can collect a large number of samples easily, it is costly and requires lots of times and efforts to manually label all of them.
- One way to do it is to use a clustering algorithm (e.g., k-means) to cluster the dataset, then for each cluster, we propagate the most common label or the the label of the most representative instance (i.e., the closest to the centroid) to all other unlabeled instances in this cluster.
5.
- K-means and BIRCH scale well with the number of instances.
- DBSCAN and mean-shift look for regions of high density.
6.
- Active learning can be useful when you have plenty of unlabeled instances but labeling is expensive.
- In this case (which is quite common), rather than randomly selecting a new instance to label it, we choose a new instance strategically.
- On way to implement it is uncertainty sampling: we choose the instance we uncertain the most, and give it to the experts to manually label it, repeat until the improved accuracy doesn't worth the cost of labeling.
7.
- People use the terms "anomaly detection" and "novelty detection" interchangeably, but they are different from each other.
- Anomaly detection assumes that the training dataset is "unclean", and the algorithm will get rid of all potential outliers in the training set, as well as outliers among new instances.
- Novelty detection assumes the training set is clean, and the algorithm will only get rid of novelty among new instances, without modifying the dataset.
- Some algorithms work best for anomaly detection (e.g., Isolation Forest), while others better suited for novelty detection (e.g., One-class SVM).
8.
- Gaussian mixture is an probabilistic model assumes that the instances is created from a combination of several Gaussian distribution with unknown parameters, and its task is to find all of these parameters.
- In other words, the assumption is the instances are grouped into multiple clusters, each with an ellipsoid shape (but the clusters may have different ellipsoid shapes, sizes, orientations and densities), and we don't know which cluster each instance belongs to.
- Some task we can use Gaussian Mixture for is density estimation, clustering and anomaly detection.
9. Two techniques to find the right number of clusters when using Gaussian mixture model:
- You can use a *theoretical information criterion*, such as *Bayesian information criterion* (BIC) or *Akaike information criterion* (AIC). You plot one of them as a function of the number of clusters, and choose the number that minimizes BIC or AIC.
- You can also choose to use Bayesian Gaussian Mixture model, which will optimize the number fo clusters automatically for you.
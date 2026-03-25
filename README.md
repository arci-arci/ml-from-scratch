A simple repository created with the idea of learning Go and Machine Learning by creating from scrath different type of ML algorithms, rangin from supervised and unsupervised ML.

### Supervised Machine Learning

- Naive Bayes classifier
  - Implemented using [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes) with smoothing priors equal to 1
- KNN
  - Data are stored inside an array
  - A future implementation: [Ball Tree](https://web.archive.org/web/20251219030314/https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=17ac002939f8e950ffb32ec4dc8e86bdd8cb5ff1), [Wikipedia](https://en.wikipedia.org/wiki/Ball_tree)
- Decision Tree
  - Implemented using [ID3 Algorithm](https://en.wikipedia.org/wiki/ID3_algorithm)
- Linear Regression
  - implementation based on [Artificial Intelligence: foundations of computational agents](https://artint.info/3e/html/ArtInt3e.Ch7.S3.html) algorithm;
  - Implemented using stochastic gradient descent;
  - Mean Squared Loss as my loss function;
  - Added a Regularization Term using L2 norm;

### Unsupervised Machine Learning

- K-Means
- DBSCAN

### Dataset

- Dataset used for classification: [Enron Spam](https://www2.aueb.gr/users/ion/data/enron-spam/)
- Dataset used for regression: [Wine recognition dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)

The data used during clustering is the output of the [Run-length encoding algorith](https://en.wikipedia.org/wiki/Run-length_encoding) over the entire Frankenstein book's first chapter.

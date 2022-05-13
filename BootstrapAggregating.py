import numpy as np
from scipy import stats

class BootstrapAggregating(object):
    """
    Bootstrap Aggregating method.
    """

    def __init__(self, learner, kwargs={}, bags=20):
        """
        Constructor method
        """
        self.learner = learner
        self.bags = bags
        self.learners = [learner(**kwargs) for i in range(bags)]

    def fit(self, X, y):
        """
        Add training data to learner

        :param X: A set of feature values used to train the learner
        :type X: numpy.ndarray
        :param y: The value we are attempting to predict given the X data
        :type y: numpy.ndarray
        """
        # Create random samples with replacement, m=data_x.shape[0]
        for lr in self.learners:
            try:
                sample_indices = np.random.choice(a=X.shape[0], size=X.shape[0],
                                            replace=True)
                lr.fit(X[sample_indices], y[sample_indices])

            except Exception as e:
                continue

    def predict(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        # Make predictions with every tree in the forest
        y = np.array([lr.predict(points) for lr in self.learners])

        # Use majority voting for the final prediction
        mode = stats.mode(y, axis=0)
        pred = mode[0][0]
        return pred

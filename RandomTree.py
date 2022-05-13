import numpy as np


class RandomTree(object):
    """
    Random tree implementation
    """
    def __init__(self, leaf_size):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.tree = []

    def fit(self, X, y):
        """
        Add training data to decision tree for training

        :param X: A set of feature values used to train the learner
        :type X: numpy.ndarray
        :param y: The value we are attempting to predict given the X data
        :type y: numpy.ndarray
        """
        # Algorithm from A Cutler 's pseudocode
        def build_tree(train_x, train_y):
            # check if #. of node's left is smaller than leaf's size
            if train_x.shape[0] <= self.leaf_size:
                return np.array([[-1, np.mean(train_y), np.nan, np.nan]])
            # check if all dataY are the same
            elif np.unique(train_y).shape[0] == 1:
                return np.array([[-1, np.mean(train_y), np.nan, np.nan]])
            else:
                # Determine random feature i to split on
                feature_indx = np.random.randint(0, train_x.shape[1])
                split_val = np.median(train_x[:, feature_indx])
                max_val = np.max(train_x[:, feature_indx])

                # If split_val(median) is equal to max, return mean
                if split_val == max_val:
                    return np.array([[-1, np.mean(train_y), np.nan, np.nan]])

                mask_lft = train_x[:, feature_indx] <= split_val
                mask_rt = train_x[:, feature_indx] > split_val
                left_tree = build_tree(train_x[mask_lft], train_y[mask_lft])
                right_tree = build_tree(train_x[mask_rt], train_y[mask_rt])
                root = np.array(
                    [[feature_indx, split_val, 1, left_tree.shape[0] + 1]])
                return np.vstack((root, left_tree, right_tree))

        # Reset
        if len(self.tree) > 0:
            self.tree = []

        self.tree = build_tree(X, y)

    def predict(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        def query_tree(point, row=0):
            node = self.tree[row]
            feature_indx = int(node[0])
            split_val = node[1]

            # if is leaf
            if feature_indx == -1:
                return split_val
            elif point[feature_indx] <= split_val:
                left_indx = int(node[2])
                row += left_indx
                return query_tree(point, row)
            else:
                right_indx = int(node[3])
                row += right_indx
                return query_tree(point, row)

        pred = np.zeros((points.shape[0]))
        for i in range(points.shape[0]):
            pred[i] += query_tree(points[i])
        return pred

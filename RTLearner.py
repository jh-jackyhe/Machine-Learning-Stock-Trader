import numpy as np
from scipy import stats

class RTLearner(object):
    """
    This is a Decision Tree Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size, verbose=True):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        pass  # move along, these aren't the drones you're looking for

    def build_tree(self, data_x, data_y):
        if data_y.shape[0] <= self.leaf_size:
            # print(type(stats.mode(data_y)[0][0]))
            # print("1")
            return np.array([[-1, stats.mode(data_y).mode, np.nan, np.nan]])
        if np.max(data_y) == np.min(data_y):
            # print(type(data_y[0]))
            # print("2")
            return np.array([[-1, data_y[0], np.nan, np.nan]])

        else:
            max_idx = np.random.choice(data_x.shape[1])
            splitval = np.median(data_x[:, max_idx])
            if np.all(data_x[:, max_idx] <= splitval):
                # print(type(stats.mode(data_y)[0][0]))
                # print("3")
                return np.array([[-1, stats.mode(data_y)[0][0], np.nan, np.nan]])

            leftidx = data_x[:, max_idx] <= splitval
            rightidx = ~leftidx
            # print(data_x[leftidx])
            lefttree = self.build_tree(data_x[leftidx], data_y[leftidx])
            righttree = self.build_tree(data_x[rightidx], data_y[rightidx])
            # print(lefttree.shape)
            root = np.array([[max_idx, splitval, 1, lefttree.shape[0]+1]])
            tree = np.vstack((root, lefttree, righttree))
            # print(tree)
            return tree

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # build and save the model
        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            print(self.tree.shape)
            print(self.tree)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        y = []
        for point in points:
            search = True
            node = 0
            while search:
                factor = int(self.tree[node, 0])
                splitval = self.tree[node, 1]
                if factor == -1:
                    y.append(splitval)
                    search = False
                else:
                    if point[factor] <= splitval:
                        node += int(self.tree[node, 2])
                    else:
                        node += int(self.tree[node, 3])
        return np.array(y)

# if __name__ == "__main__":
#     print("the secret clue is 'zzyzx'")
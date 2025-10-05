import numpy as np
import RTLearner as rt
from scipy import stats


class BagLearner(object):
    """
    This is a Decision Tree Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        """
        Constructor method
        """
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "zhe343"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        # build and save the model
        self.learners = []
        for i in range(0, self.bags):
            index = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            learner = self.learner(**self.kwargs)
            learner.add_evidence(data_x[index], data_y[index])
            self.learners.append(learner)

        if self.verbose:
            print(self.learners)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        y = []
        for learner in self.learners:
            y.append(learner.query(points))
        result = stats.mode(y)[0]
        # print(y)
        test = result[0]
        return test


# if __name__ == "__main__":
#     print("the secret clue is 'zzyzx'")
    # print(np.random.choice(5, 5, replace=True))

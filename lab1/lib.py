from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics as stats

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier

def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
    '''
    plt.imshow(np.reshape(X[index], (16,16)), cmap = 'gray')
    plt.title('Digit No '+str(index))

    return



def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''

    # Classify indices according to their digit's label (0 to 9)
    digit_table = [[],[],[],[],[],[],[],[],[],[]]
    for index, digit in enumerate(y):
        digit_table[digit].append(index)

    # randomly select one sample from each class
    digits = []
    for i in range(10):
        digits.append(random.choice(digit_table[i]))

    # plot the 10 chosen samples
    fig, ax = plt.subplots(1, 10, figsize = (20,20))
    for i in range(10):
        ax[i].imshow(np.reshape(X[digits[i]], (16,16)), cmap = 'gray')
        ax[i].set_title('Digit: {}'.format(i))

    return



def digit_mean_at_pixel(X, y, digit, y_dim, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    '''
    pixel_x = pixel[1]
    pixel_y = pixel[0]

    # Classify indices according to their digit's label (0 to 9)
    digit_table = [[],[],[],[],[],[],[],[],[],[]]
    for index, digit_class in enumerate(y):
        digit_table[digit_class].append(index)
 
    pixels = []
    for i in digit_table[digit]:
        pixels.append(X[i][y_dim * pixel_y + pixel_x])

    return stats.mean(pixels)



def digit_variance_at_pixel(X, y, digit, y_dim, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    pixel_x = pixel[1]
    pixel_y = pixel[0]

    # Classify indices according to their digit's label (0 to 9)
    digit_table = [[],[],[],[],[],[],[],[],[],[]]
    for index, digit_class in enumerate(y):
        digit_table[digit_class].append(index)

    # aggregate the (pixel_y, pixel_x) pixels 
    pixels = []
    for i in digit_table[digit]:
        pixels.append(X[i][y_dim * pixel_y + pixel_x])

    return stats.variance(pixels)



def digit_mean(X, y, digit, x_dim, y_dim):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    mean_array = [[digit_mean_at_pixel(X, y, digit, y_dim, pixel=(i,j)) for j in range(x_dim)] for i in range(y_dim)]
    return np.array(mean_array).flatten()



def digit_variance(X, y, digit, x_dim, y_dim):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    var_array = [[digit_variance_at_pixel(X, y, digit, y_dim, pixel=(i,j)) for j in range(x_dim)] for i in range(y_dim)]
    return np.array(var_array)


def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    return np.sqrt(np.sum(np.square(s-m)))



def euclidean_distance_classifier(X, X_mean):
    '''Classifies based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    classification = np.zeros((X.shape[0]), dtype=np.int64)
    for index, sample in enumerate(X):
        dist = np.inf
        classified = -1
        for i in range(10):
            new_dist = euclidean_distance(sample, X_mean[i])
            if dist > new_dist:
                dist = new_dist
                classified = i
        classification[index] = classified
    
    return classification
        


class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""

    def __init__(self, x_dim, y_dim):
        self.X_mean_ = None

        # image dimensions
        self.x_dim = x_dim
        self.y_dim = y_dim


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """

        self.X_mean_ = [digit_mean(X, y, i, self.x_dim, self.y_dim) for i in range(10)]
        self.X_mean_ = np.array(self.X_mean_)
        return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        # given X is a list of 1D arrays (imgs)
        classification = []
        for img in X:
            dist = np.inf
            classified = -1
            for i in range(10):
                new_dist = euclidean_distance(img, self.X_mean_[i])
                if dist > new_dist:
                    dist = new_dist
                    classified = i
            classification.append(classified)

        return np.array(classification)

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        predictions = self.predict(X)
        correct = sum( (predictions == y).astype(np.int) )
        return correct/len(y)


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    scores = cross_val_score(clf, X, y,
                             cv=KFold(n_splits=folds, random_state=42), 
                             scoring="accuracy")
    return scores

    
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    # count number of appearence of each class
    cardinality = [0 for i in range(10)]
    for index, digit_class in enumerate(y):
        cardinality[digit_class] += 1

    return np.array(cardinality)/y.shape[0]



class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""
   
    def __init__(self, x_dim = 256, y_dim = 1, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance
        self.X_mean = None
        self.X_var = None
        self.pC = None

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.smoothing = 1e-9

    def norm (self, x, mean, sd):
        return 1/np.sqrt(2*np.pi*sd**2) * np.exp(-0.5*((x-mean)/sd)**2)
        
    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        self.X_mean = np.array([digit_mean(X, y, i, self.x_dim, self.y_dim) for i in range(10)])
        
        if not self.use_unit_variance:
            self.X_var = np.array([digit_variance(X, y, i, self.x_dim, self.y_dim) for i in range(10)])
            self.X_var += self.smoothing
        else:
            self.X_var = np.ones((10,256))

        self.pC  = calculate_priors(X, y)
        return self


    def predict(self, X):
        """
        Make predictions for X based on max likelihood
        """
        # given X is a list of 1D arrays (imgs)
        classification = []
        for img in X:
            pxC = np.zeros((10, X.shape[1]))
            pCx = np.array(self.pC)
            prob = -np.inf
            classified = -1
            
            for i in range(10):
                pxC[i] = self.norm(img, self.X_mean[i], np.sqrt(self.X_var[i]))
                for j in range(X.shape[1]):
                    pCx[i] *= pxC[i][j]
                if prob < pCx[i]:
                    prob = pCx[i]
                    classified = i
                    
            classification.append(classified)
        return np.array(classification)

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        predictions = self.predict(X)
        correct = sum( (predictions == y).astype(np.int) )
        return correct/len(y)


class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        # WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.
        # TODO: initialize model, criterion and optimizer
        self.model = ...
        self.criterion = ...
        self.optimizer = ...
        raise NotImplementedError

    def fit(self, X, y):
        # TODO: split X, y in train and validation set and wrap in pytorch dataloaders
        train_loader = ...
        val_loader = ...
        # TODO: Train model
        raise NotImplementedError

    def predict(self, X):
        # TODO: wrap X in a test loader and evaluate
        test_loader = ...
        raise NotImplementedError

    def score(self, X, y):
        # Return accuracy score.
        raise NotImplementedError

        
def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    model = SVC(kernel='linear')
    model.fit(X,y)
    
    return stats.mean(evaluate_classifier(model, X, y, folds))

def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    model = SVC(kernel='rbf')
    model.fit(X,y)
    
    return stats.mean(evaluate_classifier(model, X, y, folds))


def evaluate_knn_classifier(X, y, folds=5, neighbors=5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(X,y)
    
    return stats.mean(evaluate_classifier(model, X, y, folds))
    

def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    model = GaussianNB()
    model.fit(X,y)
    
    return stats.mean(evaluate_classifier(model, X, y, folds))
    
    
def evaluate_custom_nb_classifier(X, y, folds=5, use_unit_variance=False):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    model = CustomNBClassifier(use_unit_variance=use_unit_variance)
    model.fit(X,y)
    
    return stats.mean(evaluate_classifier(model, X, y, folds))
    
    
def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    model = EuclideanDistanceClassifier(x_dim=256, y_dim=1)
    model.fit(X,y)
    
    return stats.mean(evaluate_classifier(model, X, y, folds))
    
'''
def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    raise NotImplementedError    
'''
    

def evaluate_voting_classifier(classifiers, X, y, folds=5, vote_method = 'hard'):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    model = VotingClassifier(estimators = classifiers, voting = vote_method)
    model.fit(X, y)
    
    return stats.mean(evaluate_classifier(model, X, y, folds))
    
    

def evaluate_bagging_classifier(base_clf, X, y, estimators, folds=5 ):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    model = BaggingClassifier(base_clf, n_estimators=estimators)
    model.fit(X, y)
    
    return stats.mean(evaluate_classifier(model, X, y, folds))
    
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


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

    return np.mean(pixels)



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

    return np.var(pixels)
    
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


# Lab1 Naive Bayes Classifier
class CustomNBClassifier(BaseEstimator, ClassifierMixin):
   
    def __init__(self, x_dim = 78, y_dim = 1, use_unit_variance=False):
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
        Calculates self.X_mean_ based on the mean feature values in X for each class.
        self.X_mean_ becomes a numpy.ndarray of shape (n_classes, n_features)
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
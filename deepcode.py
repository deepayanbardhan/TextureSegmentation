import numpy as np

from utils import COLORS, load_image
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


class GMM:
    def __init__(self, ncomp, initial_mus, initial_covs, initial_priors):
        self.ncomp = ncomp
        self.mus = np.asarray(initial_mus)
        self.covs = np.asarray(initial_covs)
        self.priors = np.asarray(initial_priors)
        
    def update(self, Xi, beliefs): # M-step
        new_mus, new_covs, new_priors = [], [], []
        soft_counts = np.sum(beliefs, axis=0)
        for i in range(self.ncomp):
            mu = np.sum(np.expand_dims(beliefs[:, i], -1) * Xi, axis=0)
            mu /= soft_counts[i]
            new_mus.append(mu)

            data_shifted = np.subtract(Xi, np.expand_dims(mu, 0))
            cov = np.matmul(np.transpose(np.multiply(np.expand_dims(beliefs[:, i], -1), data_shifted)), data_shifted)
            cov /= soft_counts[i]
            new_covs.append(cov)

            new_priors.append(soft_counts[i] / np.sum(soft_counts))

        self.mus = np.asarray(new_mus)
        self.covs = np.asarray(new_covs)
        self.priors = np.asarray(new_priors)

    def inference(self, Xi): # E-step
        unnormalized_probs = []
        for i in range(self.ncomp):
            mu, cov, prior = self.mus[i, :], self.covs[i, :, :], self.priors[i]
            unnormalized_prob = prior * multivariate_normal.pdf(Xi, mean=mu, cov=cov)
            unnormalized_probs.append(np.expand_dims(unnormalized_prob, -1))
        preds = np.concatenate(unnormalized_probs, axis=1)
        log_likelihood = np.sum(preds, axis=1)
        log_likelihood = np.sum(np.log(log_likelihood))

        preds = preds / np.sum(preds, axis=1, keepdims=True)
        return np.asarray(preds), log_likelihood

def func(n):
    # Load image
    #image_name = raw_input('Input the image name: ')
    path = 'test_ip//tm3_1_1.png'#'images/{}.jpg'.format(image_name)
    image = load_image(path)
    height, width, channels = image.shape
    X = np.reshape(image, (-1, channels))
    _mean = np.mean(X,axis=0,keepdims=True)
    _std = np.std(X,axis=0,keepdims=True)
    X = (X - _mean) / _std # Normalization

    # Input number of classes
    #ncomp = int(input('Input number of classes: '))
    ncomp=n
    # Apply K-Means to find the initial weights and covariance matrices for GMM
    kmeans = KMeans(n_clusters=ncomp)
    labels = kmeans.fit_predict(X)
    initial_mus = kmeans.cluster_centers_
    initial_priors, initial_covs = [], []
    for i in range(ncomp):
        Xi = np.array([X[j, :] for j in range(len(labels)) if labels[j] == i]).T
        initial_covs.append(np.cov(Xi))
        initial_priors.append(Xi.shape[1] / float(len(labels)))

    # Initialize a GMM
    gmm = GMM(ncomp, initial_mus, initial_covs, initial_priors)

    # EM Algorithm
    loglikelihood=[]
    prev_log_likelihood = None
    for i in range(20):
        beliefs, log_likelihood = gmm.inference(X) # E-step
        gmm.update(X, beliefs)   # M-step
        print('Iteration {}: Log Likelihood = {}'.format(i+1, log_likelihood))
        loglikelihood.append(log_likelihood)
        if prev_log_likelihood != None and abs(log_likelihood - prev_log_likelihood) < 1e-4:
            break
        prev_log_likelihood = log_likelihood

    # Show Result
    beliefs, log_likelihood = gmm.inference(X)
    map_beliefs = np.reshape(beliefs, (height, width, ncomp))
    segmented_map = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            hard_belief = np.argmax(map_beliefs[i, j, :])
            segmented_map[i,j,:] = np.asarray(COLORS[hard_belief]) / 255.0
    plt.imshow(segmented_map)
    #plt.imsave('test20.png',segmented_map)
    plt.show()
    return loglikelihood

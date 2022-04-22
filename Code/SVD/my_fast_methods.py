from glob import glob
import numpy as np
from numba import njit, objmode

__all__ = [
    '_compute_val_metrics',
    '_initialization',
    '_run_epoch',
    '_shuffle'
]

@njit
def _shuffle(X):
    np.random.shuffle(X)
    return X

@njit
def _initialization(n_users, n_items, n_factors):
    """Initializes biases and latent factor matrices.

    Parameters
    ----------
    n_users : int
        Number of unique users.
    n_items : int
        Number of unique items.
    n_factors : int
        Number of factors.

    Returns
    -------
    bu_k1 : numpy.array
        User biases critical user slope.
    bu_k2 : numpy.array
        User biases NLP slope.
    bu_c : numpy.array
        User biases constant
    bi_k1 : numpy.array
        Item biases good product slope.
    bi_k2 : numpy.array
        Item biases NLP slope.
    bi_c : numpy.array
        Item biases constant
    pu : numpy.array
        User latent factors matrix.
    qi : numpy.array
        Item latent factors matrix.
    """
    bu_k1 = np.zeros(n_users)
    bu_c = np.zeros(n_users)
    bi_k1 = np.zeros(n_items)
    bi_c = np.zeros(n_items)

    pu = np.random.normal(0, .1, (n_users, n_factors))
    qi = np.random.normal(0, .1, (n_items, n_factors))

    return bu_k1, bu_c, bi_k1, bi_c, pu, qi

@njit
def _run_epoch(X, bu_k1, bu_c, bi_k1, bi_c, pu, qi, global_mean, n_factors, lr, reg):
    """Runs an epoch, updating model weights (pu, qi, bu, bi).

    Parameters
    ----------
    X : numpy.array
        Training set.
    bu_k1 : numpy.array
        User biases critical user slope.
    bu_k2 : numpy.array
        User biases NLP slope.
    bu_c : numpy.array
        User biases constant
    bi_k1 : numpy.array
        Item biases good product slope.
    bi_k2 : numpy.array
        Item biases NLP slope.
    bi_c : numpy.array
        Item biases constant
    pu : numpy.array
        User latent factors matrix.
    qi : numpy.array
        Item latent factors matrix.
    global_mean : float
        Ratings arithmetic mean.
    n_factors : int
        Number of latent factors.
    lr : float
        Learning rate.
    reg : float
        L2 regularization factor.

    Returns:
    --------
    bu_k1 : numpy.array
        User biases critical user slope.
    bu_k2 : numpy.array
        User biases NLP slope.
    bu_c : numpy.array
        User biases constant
    bi_k1 : numpy.array
        Item biases good product slope.
    bi_k2 : numpy.array
        Item biases NLP slope.
    bi_c : numpy.array
        Item biases constant
    pu : numpy.array
        User latent factors matrix.
    qi : numpy.array
        Item latent factors matrix.
    """
    for i in range(X.shape[0]):
        user, item, rating, user_mean, item_mean = int(X[i, 0]), int(X[i, 1]), X[i, 2], X[i, 3], X[i, 4]

        # Predict current rating
        pred = global_mean + bu_k1[user] * (user_mean - global_mean) + bu_c[user] + bi_k1[item] * (item_mean - global_mean) + bi_c[item]

        for factor in range(n_factors):
            pred += pu[user, factor] * qi[item, factor]

        err = rating - pred

        # Update biases
        bu_k1[user] += lr * (err * (user_mean - global_mean) - reg * bu_k1[user])
        bu_c[user] += lr * (err - reg * bu_c[user])

        bi_k1[item] += lr * (err * (item_mean - global_mean) - reg * bi_k1[item])
        bi_c[item] += lr * (err - reg * bi_c[item])

        # Update latent factors
        for factor in range(n_factors):
            puf = pu[user, factor]
            qif = qi[item, factor]

            pu[user, factor] += lr * (err * qif - reg * puf)
            qi[item, factor] += lr * (err * puf - reg * qif)
        
        with objmode():
            print(f"{i}/{X.shape[0]}", end = '\r')

    return bu_k1, bu_c, bi_k1, bi_c, pu, qi

@njit
def _compute_val_metrics(X_val, bu_k1, bu_c, bi_k1, bi_c, pu, qi, global_mean, n_factors):
    """Computes validation metrics (loss, rmse, and mae).

    Parameters
    ----------
    X_val : numpy.array
        Validation set.
    bu_k1 : numpy.array
        User biases critical user slope.
    bu_k2 : numpy.array
        User biases NLP slope.
    bu_c : numpy.array
        User biases constant
    bi_k1 : numpy.array
        Item biases good product slope.
    bi_k2 : numpy.array
        Item biases NLP slope.
    bi_c : numpy.array
        Item biases constant
    pu : numpy.array
        User latent factors matrix.
    qi : numpy.array
        Item latent factors matrix.
    global_mean : float
        Ratings arithmetic mean.
    n_factors : int
        Number of latent factors.

    Returns
    -------
    loss, rmse, mae : tuple of floats
        Validation loss, rmse and mae.
    """
    residuals = []

    for i in range(X_val.shape[0]):
        user, item, rating, user_mean, item_mean = int(X_val[i, 0]), int(X_val[i, 1]), X_val[i, 2], X_val[i, 3], X_val[i, 4]
        pred = global_mean

        if user > -1:
            pred += bu_k1[user] * (user_mean - global_mean) + bu_c[user]

        if item > -1:
            pred += bi_k1[item] * (item_mean - global_mean) + bi_c[item]

        if (user > -1) and (item > -1):
            for factor in range(n_factors):
                pred += pu[user, factor] * qi[item, factor]

        residuals.append(rating - pred)

    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()

    return loss, rmse, mae
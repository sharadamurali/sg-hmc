# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:43:09 2019

@author: sharada
SG-HMC in 2D
"""

import numpy as np
import scipy.stats as sp
from tqdm import tqdm
from matplotlib import pyplot as plt

# Generating data
def gen_data(n=100):
    # Thus, theta1 is 0 and theta2 is 0.5
    x = 0.5 * np.sqrt(2) * np.random.randn(n) + 0.5 * (np.sqrt(2) * np.random.randn(n) + 1)
    return x

# Gradient of log pdf of a normal distribution wrt x
def grad_log_pdf(x, mu_shift, sigma_scale):
    return (-x + mu_shift)/(sigma_scale**2)


# Gradient of log pdf of prior (normal pdf) --SEEMS FINE
def log_prob_prior_g(theta, sigma_1=np.sqrt(10), sigma_2=1):
    lp_prior = np.array([grad_log_pdf(theta[0], 0, sigma_1), grad_log_pdf(theta[1], 0, sigma_2)])
    return lp_prior

# Gradient of log-likelihood function
def log_prob_lik_g(theta, x):
    theta1 = theta[0]
    theta2 = np.sum(theta)
    sigma_x = np.sqrt(2)
    
    f1 = sp.norm.pdf(x, loc=theta1, scale=sigma_x)
    f2 = sp.norm.pdf(x, loc=theta2, scale=sigma_x)
    
    grad_theta1 = (-1/(f1 + f2)) * (f1 * grad_log_pdf(x, theta1, sigma_x) + f2 * grad_log_pdf(x, theta2, sigma_x))
    grad_theta2 = (-1/(f1 + f2)) * (f2 * grad_log_pdf(x, theta2, sigma_x))
    
    lik_fn_g = np.array([grad_theta1, grad_theta2])
    
    return lik_fn_g


if __name__ == "__main__":

    epochs = 10000

    x = gen_data()
    N = len(x)
    
    # Parameters
    M = 1
    M_inv = [[1, 0], [0, 1]]    # Identity Matrix
    B = 0   # Zero matrix
    C = 1
    C_mat = [[1, 0], [0, 1]] #[[1, -1], [-1, 1]]  # Positive semi-definite matrix
    
    # Initialization
    theta_t = np.zeros(2)
    r_t = np.random.randn(2) * np.sqrt(M)
    
    thetas = np.zeros((epochs, 2)) # Keeps track of history of produced thetas
    eps_history = np.zeros(epochs)
    grad_history = np.zeros((epochs, 2))
    U_history = np.zeros((epochs, 2))
    
    # Empirically chosen
    b = 15500
    a = 2

    for i in tqdm(range(epochs)):
        
        # Optionally resample momentum
        if np.random.random_sample() >= 0.5:
            r_t = np.random.randn(2) * np.sqrt(M)
        
        theta_0 = theta_t
        r_0 = r_t        
        
        for j in range(N):
            eps = a * (b + i*j) ** (-0.55)
            theta_j = theta_0 + eps * np.dot(M_inv, r_0)
            U = - 1 * (log_prob_prior_g(theta_j) + N * log_prob_lik_g(theta_j, x[j]))
            noise = np.random.randn(2) * np.sqrt(2 * (C - B) * eps)
            r_j = r_0 - (eps * U) - (eps * C * np.dot(M_inv, r_0)) + noise            
            # Update r_0 and theta_0
            r_0 = r_j
            theta_0 = theta_j
        
        theta_t = theta_j
        r_t = r_j
        
        thetas[i] = theta_t
        U_history[i] = U
        grad_history[i] = eps * np.dot(M_inv, r_0)
    
    plt.figure()
    plt.title('SG-HMC Estimated Posterior Distribution')
    plt.xlabel('Theta2')
    plt.ylabel('Theta1')
    plt.hist2d(thetas[:, 0], thetas[:, 1], bins=100)
    plt.show()
    
import numpy as np
from scipy.stats import multivariate_normal

def gaussian(x, mu, sigma):
    det = np.linalg.det(sigma)
    if det==0:
        sigma=sigma+0.0001
        det=np.linalg.det(sigma)
    inv_sigma=np.linalg.inv(sigma)
    x_mu=x-mu
    t=-0.5*np.sum(x_mu @ inv_sigma * x_mu, axis=1)
    ret = np.exp(t)
    norm_const_den = (((2 * np.pi) ** (x.shape[1] / 2)) * (det ** 0.5))
    norm=1.0/norm_const_den
    pdf=ret*norm
    return pdf
def E_step(X,k,mu,sigma,prob,pi) :
    for i in range(k):
        prob[:,i] = pi[i] * multivariate_normal.pdf(X, mu[i], sigma[i], allow_singular=True)
    normalized_sum = prob.sum(axis=1, keepdims=True)
    prob = prob / normalized_sum
    return prob
def update_break_flag(old_ll,new_ll) :
    difference = abs(new_ll - old_ll)
    if difference < 1e-3:
        return 1
    else:
        return 0
def M_step(X,k,prob,bias,mu,sigma,pi,updated_probs):
    for i in range(k):
        prob_sum = (prob[:, i]).sum()
        mu[i] = prob[:, i].dot(X)/prob_sum
        sigma[i] = np.cov(X.T, aweights=prob[:, i], bias=True)
        np.fill_diagonal(sigma[i], sigma[i].diagonal() + bias)
        pi[i] = prob_sum / X.shape[0]
        #ekhane cholee naaaa
        updated_probs[:,i] = pi[i] * multivariate_normal.pdf(X, mu[i], sigma[i], allow_singular=True)
    return mu,sigma,pi,updated_probs

def gmm_estimator(X, k, no_of_iter):
    pi=np.ones(k)/k
    n=X.shape[0]
    m=X.shape[1]
    bias=1e-6
    break_flag=0
    mu=np.random.rand(k,m)
    sigma=np.array([np.eye(m)] * k)
    prob= np.zeros((n, k))
    log_likelihood = -10000000
    updated_probs=np.zeros((n,k))
    for q in range(no_of_iter):
        #E-step:probablity computation
        prob=E_step(X,k,mu,sigma,prob,pi)
        #print(prob)



        #M-step:Update step
        mu,sigma,pi,updated_probs=M_step(X,k,prob,bias,mu,sigma,pi,updated_probs)

        #temp_likelihood=np.sum(np.log(np.sum(updated_probs,axis=1)))
        log_of_sum=np.log(np.sum(updated_probs,axis=1))
        temp_likelihood=log_of_sum.sum()
        # temp_likelihood = np.sum(np.log(
        # np.sum([pi[j]*multivariate_normal.pdf(X, mu[j], sigma[j],allow_singular=True) for j in range(k)])))
        #print("hereeeeeeee")
        break_flag=update_break_flag(log_likelihood,temp_likelihood)
        if break_flag==1:
            break

        log_likelihood=temp_likelihood
    #print(log_likelihood)
    return log_likelihood

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from GMM import E_step,M_step,update_break_flag
def show_plots(input, k,dim, no_of_iter):

    pca_cifar = PCA(2)
    graph_data=input
    if(dim>2):
        graph_data=pca_cifar.fit_transform(input)
    pi=np.ones(k)/k
    n=input.shape[0]
    m=input.shape[1]
    bias=1e-6
    fig = plt.figure()
    mu=np.random.rand(k,m)
    sigma=np.array([np.eye(m)]*k)
    prob= np.zeros((n, k))
    log_likelihood=-10000000
    updated_probs=np.zeros((n,k))
    plt.ion()
    scatter_left=graph_data[:,0]
    scatter_right=graph_data[:,1]
    plt.scatter(scatter_left, scatter_right, 10,color="orange")
    for q in range(no_of_iter):
        plt.scatter(graph_data[:, 0], graph_data[:, 1], 10,color="orange")
        #E-step:probablity computation

        prob = E_step(input, k, mu, sigma, prob, pi)
        #print(prob)


        #M-step:Update step
        mu, sigma, pi, updated_probs = M_step(input, k, prob, bias, mu, sigma, pi, updated_probs)
        for cluster in range(k):
            x_limit_left=min(graph_data[:, 0])
            x_limit_right=max(graph_data[:, 0])
            y_limit_left=min(graph_data[:, 1])
            y_limit_right=max(graph_data[:, 1])
            x_sorted=np.sort(graph_data[:,0])
            y_sorted=np.sort(graph_data[:,1])
            if(dim>2):
                vis_mu=pca_cifar.fit_transform(mu)
                cmp_transpose=pca_cifar.components_.T
                #intermediate=sum(x * y for x, y in zip(sigma[cluster], cmp_transpose))
                intermediate=np.dot(sigma[cluster],cmp_transpose)
                vis_sigma=np.dot(pca_cifar.components_,intermediate)
            else:
                vis_mu=mu
                vis_sigma=sigma[cluster]
            X,Y=np.meshgrid(np.linspace(x_limit_left, x_limit_right),np.linspace(y_limit_left,y_limit_right))
            XX=np.array([X.flatten(), Y.flatten()])
            Z=multivariate_normal.pdf(XX.T,vis_mu[cluster],vis_sigma, allow_singular=True).reshape(X.shape[0],X.shape[0])
            #print(Z.shape)
            plt_name="Per iteration animation for k="+str(k)
            plt.title(plt_name)
            plt.contour(X,Y,Z,6)
            plt.draw()
        plt.pause(0.5)

        #temp_likelihood=np.sum(np.log(np.sum(updated_probs,axis=1)))
        log_of_sum=np.log(np.sum(updated_probs,axis=1))
        temp_likelihood=log_of_sum.sum()
        # temp_likelihood = np.sum(np.log(
        # np.sum([pi[j]*multivariate_normal.pdf(X, mu[j], sigma[j],allow_singular=True) for j in range(k)])))

        if update_break_flag(log_likelihood,temp_likelihood)==1:
            break
        log_likelihood = temp_likelihood
        plt.clf()


    #print(log_likelihood)


def visualize(data,k):
    X=data
    dim=X.shape[1]
    show_plots(X,k,dim,100)
    plt.ioff()
    plt.show()


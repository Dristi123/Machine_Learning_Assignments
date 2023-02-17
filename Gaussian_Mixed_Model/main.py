import numpy as np
import matplotlib.pyplot as plt
from GMM import *
from visualization import *

def load_data():
    file_name="data2D.txt"
    file_data=np.loadtxt(file_name)
    return file_data

def plot(X_list,y_list,color):
    plt.plot(X_list,y_list,color=color,linewidth=2)
    plt.ylabel("Converged log-likelihood")
    plt.xlabel("Number of clusters(k)")

if __name__ == '__main__':
    X = load_data()
    range_k=10
    no_of_iterations=100
    log_likelihoods={}
    for k in range(1,range_k+1):
        likelihood=gmm_estimator(X,k,no_of_iterations)
        #log_likelihoods.append(likelihood)
        log_likelihoods[k]=likelihood
        print(k)
    x_list=list(log_likelihoods.keys())
    y_list=list(log_likelihoods.values())
    plot(x_list,y_list,"red")
    plt.show()

    #print("Want to proceed to task 2?")
    ans=input("Want to proceed to task 2?\n 1.Yes 2.No\n")
    if ans=="1":
        k_star=input("Provide the value of k*\n")
        k_star=int(k_star)
        visualize(X,k_star)








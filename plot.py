import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from deepcode import func

if __name__ == '__main__':
    
    l=[]
    for ncomp in range (3,9):
        x=func(ncomp)
        l.append(x)
    
    plt.subplot()
    plt.grid(color='black', linestyle='-', linewidth=0.15)
    
    t = np.arange(0., 20., 1.)         
    for i in range (3,9):
        plt.plot(t,l[i],label="n_comps= %i" %i)
    
    plt.legend(bbox_to_anchor=(.75,0.42), loc="upper left", borderaxespad=0.)
    plt.title('log-likelihood vs no. of iterations')
    plt.xlabel('no. of iterations')
    plt.ylabel('log-likelihood value')
    plt.show()
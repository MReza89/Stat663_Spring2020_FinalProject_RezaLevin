#!/usr/bin/env python
# coding: utf-8

# ## import libraries

# In[1]:


import time
import warnings


# In[2]:


# all the imports needed for this blog
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import imageio
from io import BytesIO
get_ipython().run_line_magic('matplotlib', 'inline')
import numba
from numba import jit
import seaborn as sns


# In[ ]:





# ## Functions

# In[3]:


def Similarity(x,y):
    """computes the similarity, negative eulidian distance, between two vecotrs"""
    return -np.sum((x-y)**2)


# In[4]:


def sentence_similarity(i, k, dictionary):
    '''Returns similarity of sentence i and sentence k, given full dictionary 
    
    From Frey and Dueck (2007):
    
    The similarity of sentence i to sentence k was set to the negative sum of the 
    information-theoretic costs (S5) of encoding every word in sentence i using 
    the words in sentence k and a dictionary of all words in the manuscript. For 
    each word in sentence i, if the word matched a word in sentence k, the coding 
    cost for the word was set to the negative logarithm of the number of words in 
    sentence k (the cost of coding the index of the matched word), and otherwise it 
    was set to the negative logarithm of the number of words in the manuscript 
    dictionary (the cost of coding the index of the word in the manuscript dictionary). 
    A word was considered to match another word if either word was a substring of the other.
    '''
    
    # go through each word in sentence i
    
    sentence_i = i.split()
    sentence_k = k.split()
    
    # return 0 if sentences are the same
    if sentence_i == sentence_k:
        return 0
    
    sim = 0
    for word_i in sentence_i:
        counter = 0
        for word_k in sentence_k:
            counter += 1
            if (word_i in word_k) | (word_k in word_i):
                sim -= np.log(len(sentence_k))
                break
            elif counter < len(sentence_k):
                continue
            else: 
                sim -= np.log(len(dictionary))
                break
                
    return sim


# In[5]:


#@jit(nopython=nopython_, cache=cache_)
def pairwise_dists(x, y):
    """ Computing pairwise distances using memory-efficient
        vectorization.

        Parameters
        ----------
        x : numpy.ndarray, shape=(M, D)
        y : numpy.ndarray, shape=(N, D)

        Returns
        -------
        numpy.ndarray, shape=(M, N)
            The square Euclidean distance between each pair of
            rows between `x` and `y`."""
    return  np.sum(x**2,axis=-1).reshape(-1,1)+np.sum(y**2,axis=-1)-2*np.dot(x,y.T) 


# In[6]:



#@jit(nopython=nopython_, cache=cache_)
def Initialize_Matrices(X_data,Similarity_metric='Euclidean',memory_efficinet=True,BroadCasting=True,Dictionary=None):
    """create the similarity (S), availability (A) and Responsibility (R) matrices given input data
    X_data >> rows represent observations and columns represent features """
    n=X_data.shape[0]
    S=np.zeros((n,n))
    A=np.zeros((n,n))
    R=np.zeros((n,n))
    if Similarity_metric=='Euclidean':
        if memory_efficinet==False and (BroadCasting==False):
            for i in range(n-1):
                for j in range(i+1,n):
                        S[i,j]=Similarity(X_data[i,:],X_data[j,:]) 
            S=S+S.T
            
        elif (memory_efficinet==False) and (BroadCasting==True):
            S=-np.sum((X_data[None,:,:]-X_data[:,None,:])**2,axis=-1)
        else:
            S=-1*pairwise_dists(X_data, X_data)
        return S,R,A
    if Similarity_metric =='Sentences':
        for i in range(n):
            for j in range(n):
                # here, X_data would be an array of sentences
                S[i,j] = sentence_similarity(X_data[i], X_data[j], Dictionary)
        return S,R,A
    else:
        raise ValueError("The only Predefined Similarity metric is 'Euclidean'. For '%s' metric, a new function should be defined"%Similarity_metric)


# In[7]:


def Update_R_loop(S,R,A,temp=None,damping_factor=0.5):
    if temp is None: temp=np.zeros_like(R)
    n=S.shape[0]
    for i in range(n):
        for k in range(n):
            # approach 1
    #         Ids_kp=[kp for kp in list(range(n)) if not kp==k and not kp==i]
    #         temp=np.max(A[i,Ids_kp]+S[i,Ids_kp])
            #approach 2
            temp1=(A[i,:]+S[i,:])
            temp1[i]=-np.inf
            temp1[k]=-np.inf
            temp=np.max(temp1)

            R[i,k]=R[i,k]*damping_factor+(S[i,k]-temp)*(1-damping_factor)
    return R


# In[8]:


#@jit(nopython=nopython_, cache=cache_)
def Update_R_Broadcast(S,R,A,temp=None,damping_factor=0.5):
    if temp is None: temp=np.zeros_like(R)
    temp=A+S
    temp.flat[::(temp.shape[0]+1)]=-np.inf #np.fill_diagonal(temp,-np.inf) 
    rows=np.arange(S.shape[0])
    Id_max_E_row=np.argmax(temp,axis=1)
    first_max=temp[rows,Id_max_E_row]
    temp[rows,Id_max_E_row]=-np.inf
    second_max=temp[rows,np.argmax(temp,axis=1)]
    Max_mat_A_plus_S=np.zeros_like(S)+first_max.reshape(-1,1)#[:,None]
    Max_mat_A_plus_S[rows,Id_max_E_row]=second_max
    R=R*damping_factor+(S-Max_mat_A_plus_S)*(1-damping_factor)
    return R


# In[9]:


def Update_A_loop(R,A,temp=None,damping_factor=0.5):
    if temp is None: temp=np.zeros_like(R)
    n=R.shape[0]
    for i in range(n):
        for k in range(n):
            
            temp1=np.array(R[:, k]) #R[:,k]+0.0 
            temp1[i]=-np.inf
            temp1[k]=-np.inf

            temp_sum=np.sum(temp1[temp1>0])
            
            if i!=k:
                A[i,k]=(damping_factor)*A[i,k]+(1-damping_factor)*min(0, R[k,k]+temp_sum)
            else:            
                A[i,k]=(damping_factor)*A[i,k]+(1-damping_factor)*temp_sum
    return A


# In[10]:


#@jit(nopython=nopython_, cache=cache_)
def Update_A_Broadcast(R,A,temp=None,damping_factor=0.5,Approach_paper=True):
    
    if temp is None: temp=np.zeros_like(R)
    
    if Approach_paper==False:
        temp=np.array(R)
        temp=np.where(temp<0,0,temp)# np.clip(a,0,np.inf) # a[a<0]=0 # np.maximum(R, 0, a)
        #np.fill_diagonal(temp,0)
        temp.flat[::(temp.shape[0]+1)]=0
        temp2=np.sum(temp,axis=0)
        temp=temp2-np.clip(R,0,np.inf) #np.sum(a,axis=0)-np.clip(a,0,np.inf)
        temp=temp+np.diag(R)
        temp[temp>0]=0 # np.clip(a,-np.inf,0)

        #np.fill_diagonal(temp,temp2)
        temp.flat[::(temp.shape[0]+1)]=temp2
    else:
        np.maximum(R, 0, temp)        
        temp.flat[::(temp.shape[0]+1)]=np.diag(R) #np.fill_diagonal(temp,np.diag(R)) #temp.flat[::(temp.shape[0]+1)]=np.diag(R)
        temp=np.sum(temp,axis=0)-temp
        temp2=np.diag(temp).copy()
        temp[temp>0]=0 # np.clip(a,-np.inf,0)
        temp.flat[::(temp.shape[0]+1)]=temp2 #np.fill_diagonal(temp,temp2)#temp.flat[::(temp.shape[0]+1)]=temp2
        
    A=damping_factor*A+(1-damping_factor)*temp
    
    
    return A


# In[11]:


def affinity_propagation(Data_X_samples, Similarity_metric_='Euclidean',preference=None, max_iter=200,max_err=1e-4,
                         damping=0.5, return_n_iter=False,Visulization_track=False,Plot_Clusters=True,Draw_lines=False,
                         Version_Fast=True, Dictionary=None,**plot_kwds):
    """
    Affinity Propagation 
    
    Parameters
    ----------
    Data_X_samples:
        Inout data array (observations in rows and features in columns)    
        
    preference : 
        "array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities." from Skit-learn documnetations
        
    max_iter : 
        int, optional, default: 200
        Maximum number of iterations
        
    damping :
        float, optional, default: 0.5
        Damping factor between 0.5 and 1.
        
    Returns
    -------
    cluster_centers_indices : 
        array, shape (n_clusters,)
        index of clusters centers
        
    labels : 
        array, shape (n_samples,)
        cluster labels for each point
            
        
    """
    
    Data_X=np.copy(Data_X_samples)
    n_samples=Data_X.shape[0]
    S,R,A=Initialize_Matrices(Data_X,Similarity_metric_,Dictionary=Dictionary)
    

    if preference is None:
        preference = np.median(S)
    if damping < 0.5 or damping >= 1:
        raise ValueError('Based on the suggestion of authors (Brendan J. Frey* and Delbert Dueck 2007) ,        damping must be >= 0.5 and < 1')

    preference = np.array(preference)
    S.flat[::(n_samples + 1)] = preference #np.fill_diagonal(S,preference)  #  S.flat[::(n_samples + 1)] = preference
    
    # Intermediate results
    current_sol=np.zeros_like(S)
    temp_arr=np.zeros_like(S)
    
    
    last_sol=np.zeros_like(S)
    current_sol=np.zeros_like(S)
    last_exemplars = np.array([])
    
    # Remove degeneracies
    random_state = np.random.RandomState(0)
    S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) *
          random_state.randn(n_samples, n_samples))
    
    figures=[]
    
    if Version_Fast==True:
        
        
        for it in range(max_iter):

            R=Update_R_Broadcast(S,R,A,temp_arr,damping)
            A=Update_A_Broadcast(R,A,temp_arr,damping)

            current_sol=A+R
            labels = np.argmax(current_sol, axis=1)
            exemplars = np.unique(labels)

            if Visulization_track==True:
                if (np.all(last_exemplars != exemplars)):
                    figures.append(plot_iteration(it,labels,exemplars,Data_X,Draw_lines=Draw_lines))

            if np.allclose(last_sol, current_sol):#np.sqrt(np.sum((last_sol- current_sol)**2)) < max_err:# np.allclose(last_sol, current_sol):
                print(exemplars, it)
                break

            last_sol = current_sol
            last_exemplars = exemplars
        # mapping the labels into a sorted, gapless, list    
        #labels = np.searchsorted(last_exemplars, labels)

        Cluster_Centers=Data_X[last_exemplars]
        
    else:
        
        for it in range(max_iter):

            R=Update_R_loop(S,R,A,temp_arr,damping)
            A=Update_A_loop(R,A,temp_arr,damping)

            current_sol=A+R
            labels = np.argmax(current_sol, axis=1)
            exemplars = np.unique(labels)

            if Visulization_track==True:
                if (np.all(last_exemplars != exemplars)):
                    figures.append(plot_iteration(it,labels,exemplars,Data_X,Draw_lines=Draw_lines))

            if np.allclose(last_sol, current_sol):#np.sqrt(np.sum((last_sol- current_sol)**2)) < max_err:# np.allclose(last_sol, current_sol):
                print(exemplars, it)
                break

            last_sol = current_sol
            last_exemplars = exemplars
        # mapping the labels into a sorted, gapless, list    
        #labels = np.searchsorted(last_exemplars, labels)

        Cluster_Centers=Data_X[last_exemplars]
        
    if Plot_Clusters==True:
        plot_clusters(Data_X,gapless_labels=np.searchsorted(last_exemplars, labels),**plot_kwds)
        #plot_iteration(it,labels,last_exemplars,Data_X,Draw_lines)
    
    if Visulization_track==True:
        return last_exemplars,labels,Cluster_Centers,it,figures
    else:
        return last_exemplars,labels,Cluster_Centers,it
    


# #### Cluster Visualization function

# In[12]:


def plot_clusters(data,gapless_labels,**plot_kwds):
    fig = plt.figure(figsize=(15, 5))
    palette = sns.color_palette(palette='deep', n_colors=np.unique(gapless_labels).max() + 1)
    
    colors = [palette[x] for x in gapless_labels] #[palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.show()


# In[13]:


def plot_iteration(iter_i,labels,exemplars,Data,Draw_lines=False):
    """ Here we assume that we are working with 2d data for the purpose of visualization"""
    fig = plt.figure(figsize=(15, 5))
    
    colors = dict(zip(exemplars, cycle('bgrcmyk')))
    
    for i in range(len(labels)):
        X = Data[i][0]
        Y = Data[i][1]
        
        if i in exemplars:
            exemplar = i
            edge = 'k'
            ms = 10
        else:
            exemplar = labels[i]
            ms = 3
            edge = None
            if Draw_lines==True:
                plt.plot([X, Data[exemplar][0]], [Y, Data[exemplar][1]], c=colors[exemplar])
            else:
                plt.plot([Data[exemplar][0]], [Data[exemplar][1]], c=colors[exemplar])
        plt.plot(X, Y, 'o', markersize=ms,  markeredgecolor=edge, c=colors[exemplar])
        

    plt.title('Number of exemplars: %d @ iteration: %d' % (len(exemplars),iter_i))
    return fig


# ##### Make a gif of Affinity Propagation Process

# In[ ]:


def make_gif(figures, filename, fps=10, **kwargs):
    images = []
    for fig in figures:
        output = BytesIO()
        fig.savefig(output)
        plt.close(fig)  
        output.seek(0)
        images.append(imageio.imread(output))
    imageio.mimsave(filename, images, fps=fps, **kwargs)


# In[ ]:








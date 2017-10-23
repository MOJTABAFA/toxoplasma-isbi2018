# I already extracted the trajectory pointes per video and put them in a 2D numpy Array , it includes the m rows 
# ( for each block) and in each row we have different points of each trajectories.(However, we can change this
# number arbitrary by changing the Trajectory_length parameter)

from __future__ import division
import pandas as pd
import numpy as np 
import seaborn as sns
import csv
import sys
from sklearn.cluster import spectral_clustering
import sklearn.cluster as cluster
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats.stats import pearsonr
from sys import platform as sys_pf
if sys_pf == 'Darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import ar
#import seaborn as sns

def plot_trajectory_2D(n, is_one, x, y,cxx=None,cyy=None, step=1, scatter_plot=False):
    """************************************************************** 
    n:        no_of_frames to plot
    is_one:   True:  Single object's trajectory
              False: Multiple object's trajectories
    step:     step_size
    x,y:      Coordinates to plot
    **************************************************************"""
    if is_one:
        traj_points = 1
        T = np.linspace(0,1,x.shape[0])

    else : 
        traj_points = x.shape[0]
        T = np.linspace(0,1,traj_points)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    s = step
    plt.gca().invert_yaxis()
    for index in range(traj_points): 
        for i in range(0,n-s,s):
            if is_one : 
                ax.plot(x[i:i+s+1],y[i:i+s+1],linewidth=3)
            else : 
                
                cx = x[index]
                cy = y[index]
                if (scatter_plot):
                    ax.scatter(cx[i:i+s+1],cy[i:i+s+1],color=(0.0,0.0,T[index]))
                else:
                    ax.plot(cx[i:i+s+1],cy[i:i+s+1],linewidth=3,color=(0.0,0.0,T[index]))
    if (cxx is not None):
        n=cxx.shape[1]
        for index in range(traj_points): 
            for i in range(0,n-s,s):
                if is_one : 
                    ax.plot(x[i:i+s+1],y[i:i+s+1],linewidth=3)
                else : 

                    cx = cxx[index]
                    cy = cyy[index]
                    if (scatter_plot):
                        ax.scatter(cx[i:i+s+1],cy[i:i+s+1],color=(T[index],0.0,0.0))
                    else:
                        ax.plot(cx[i:i+s+1],cy[i:i+s+1],linewidth=3,color=(0.0,0.0,T[index]))

    
    plt.show()


def AR_Normalization( x, y):

	BCPool = np.stack([x, y])
	print(BCPool)
	print(np.shape(BCPool))
	# BCPool2 = np.transpose(BCPool, (1,2,0))

	# #*******{ Normalization }******
	
	# BCPool_copy = BCPool2.copy()
	# BCPool_copy = np.transpose(BCPool_copy, (1,0,2))
	# BCPool_copy = BCPool_copy.reshape(BCPool_copy.shape[0], BCPool_copy.shape[1] * BCPool_copy.shape[2])

	# BCPool_norm, normal_norms = normalize(BCPool_copy, axis = 0, return_norm = True)  

	# recon = BCPool_norm * normal_norms
	# recon = recon.reshape( BCPool_copy.shape[0], int(BCPool_copy.shape[1] / 2) ,2)
	# recon = np.transpose( recon,(1,0,2))

	# #********{ Testing the correctness of normalization }*************
	# for i in range(recon.shape[0]):
	# 	curr = recon[i,:,:]
	# 	x = curr[:,0]; 
	# 	y=curr[:,1];
	# 	plt.plot(x,y)
	# plt.show()
	# # plot_trajectory_2D( recon.shape[1] , False, recon[:,:,0], recon[:,:,1])

	return BCPool

def State_PCA(num_pca, order, matt):

	pca_components = num_pca
	ar_order = order
	normal_X, normal_C, normal_S, normal_U = ar.state_space( matt.T, pca_components)

	# print(np.shape(normal_S))
	# print(normal_S)
	
	temp_S = normal_S[ normal_S != 0]
	# tot = sum(normal_S)
	# var_exp = [(i / tot) * 100 for i in sorted( normal_S, reverse = True )]
	tot = sum(temp_S)
	var_exp = [(i / tot) * 100 for i in sorted(temp_S, reverse=True)]
	cum_var_exp = np.cumsum(var_exp)
	print ((var_exp))

	with plt.style.context('seaborn-whitegrid'):
	    plt.figure(figsize=(6, 4))

	    plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
	            label='individual explained variance')
	    plt.step(range(len(var_exp)), cum_var_exp, where='mid',
	             label='cumulative explained variance')
	    plt.ylabel('Explained variance ratio')
	    plt.xlabel('Principal components')
	    plt.legend(loc='best')
	    plt.tight_layout()
	plt.show()
	return normal_X

def PCA3D_visualization(normal_X):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(normal_X[0,:], normal_X[1,:], normal_X[2,:],  c='r', marker='o')
	ax.set_xlabel('pc1');ax.set_ylabel('pc2');ax.set_zlabel('pc3');

	plt.show()

def PCA2D_visualization( normal_X, num_pca):
	fig, ax = plt.subplots(1,num_pca, sharey=True, sharex= True, figsize=(10,num_pca))
	ax[0].scatter(normal_X[0,:],normal_X[1,:]);ax[0].set_xlabel('pc1');ax[0].set_ylabel('pc2')
	ax[1].scatter(normal_X[1,:],normal_X[2,:]);ax[1].set_xlabel('pc2');ax[1].set_ylabel('pc3')
	ax[2].scatter(normal_X[0,:],normal_X[2,:]);ax[2].set_xlabel('pc1');ax[2].set_ylabel('pc3')
	plt.show()

	if ( num_pca / 2 ) == 0 : 
	    fig, ax = plt.subplots( int( num_pca / 2 ) , 2 , sharex = True, figsize = ( 15, 5 ))
	else:
		fig, ax = plt.subplots( num_pca - 1 , 2 ,sharex = True, figsize = ( 15 , 5 ))

	for i in range(num_pca):
	    r = int( i / 2 )
	    c = int( i % 2 )
	    ax[ r , c ].scatter( range( len( normal_X[i,:] )), normal_X[ i, :]) 
	    ax[ r , c ].set_ylabel( 'pc' + str(i))
	plt.show()
	
def AR_Train( normal_X, ar_order ):
	normal_X_train = normal_X[:,:]
	is_plot=True
	normal_A, normal_Q = ar.train(normal_X_train, order = ar_order)

	# Feature Vector
	feature_vector = np.array(normal_A).flatten()
	print ('Feature Vector:')
	print (np.array(normal_A))

	# # Plot AR
	# if (is_plot):
	#     fig, ax = plt.subplots(1,len(normal_A), figsize=(10,3))
	#     for i in range(len(normal_A)):
	#         if (len(normal_A) > 1):
	#             ax[i].imshow(normal_A[i], cmap = "jet")
	#             ax[i].set_xlabel('A'+str(i))
	#         else:
	#             ax.imshow(normal_A[i], cmap = "jet")
	#             ax.set_xlabel('A'+str(i))

	# plt.show()
	return normal_A

def Hist_of_Components( Norm_Before, Norm_After):

	pcs = Norm_After.shape[0]
	f, axarr = plt.subplots(1, pcs,sharey=True, figsize=(15,3))

	for i in range(pcs):
	    axarr[i].hist(Norm_Before[i,:],bins=10, alpha=0.5, label='Before_Calcium')
	    axarr[i].hist(Norm_After[i,:],bins=10, alpha=0.5, label='After_calcium')
	    axarr[i].legend(loc='upper right')
	plt.show()

def main():


# Loading all the files ...

	
	XA = np.loadtxt('XA.txt', delimiter=',') 
	XB = np.loadtxt('XB.txt', delimiter=',') 
	YA = np.loadtxt('YA.txt', delimiter=',') 
	YB = np.loadtxt('YB.txt', delimiter=',') 
	AA = np.loadtxt('AA.txt', delimiter=',') 
	AB = np.loadtxt('AB.txt', delimiter=',') 


#%++++===================================================

	# print (XAfter)
			 
	# for i in range(136):
	# 	plt.plot(XA[i],YA[i])
	# plt.show()
#====================================={ Creating XB, XA , YB and YA}======================================
	np.random.seed(19680801)
	Number_of_points = np.shape(XA)[0]
	object_num = 6
	calcium_point = 248
	frame_num = 490
	covMx =[]
	N = frame_num
	
	Z = np.zeros(frame_num)
	for i in range (frame_num):
		Z[i] = i

	Zcal = np.zeros(calcium_point)
	for i in range (calcium_point):
		Zcal[i] = i


#=============================={ Implementing K of Covariance and Pearson }=====================================	

	K_cov = np.zeros(shape= (Number_of_points, Number_of_points))
	K_pearsonR = np.zeros(shape= (Number_of_points, Number_of_points))
	K_CovAngles = np.zeros(shape= (Number_of_points, Number_of_points))

	#for i in range(calcium_point):
	for i in range(Number_of_points):
		for j in range(Number_of_points):
			K_cov[i][j] = np.cov(XA[i], XA[j], ddof=0)[0][1] 
			K_pearsonR[i][j] = pearsonr(XA[i],XA[j])[1]
			K_CovAngles[i][j] = np.cov(AA[i],AA[j], ddof=0)[0][1]

	CzeroCovAng = 0
	CzeroCovx = 0
	for i in range (139):
		for j in range (139):
			if (K_CovAngles[i][j] == 0):
				CzeroCovAng += 1
			if (K_cov[i][j] == 0) :
				CzeroCovx +=1

#=============================={ Implementing AR part }=====================================
	
	AR_Order = 5
	columns = AR_Order * 2 * 2	
	
	Flatten_ARB = np.zeros(shape = (Number_of_points, columns))
	Flatten_ARA = np.zeros(shape = (Number_of_points, columns))

	# XA means X after calcium and XB => X Before calcium
	for index in range (Number_of_points): 
		BefCal = AR_Normalization( XB[index], YB[index])
		AftCal = AR_Normalization( XA[index], YA[index])

	#**********************************************************************************************************#
	# Since we are applying the AR parameters directly to spectral clustering we commented the following parts 
	# In case, you are going to use the the PCA parts you should activate the following lines./
	#**********************************************************************************************************#
		

		# Norm_Before = State_PCA(5, 2, BefCal)
		# Norm_After  = State_PCA(5, 2, AftCal)

		#******************************{ 3D Plot of PCA components }*****************************
		# PCA3D_visualization(Norm_Before)
		# PCA3D_visualization(Norm_After)
		#*****************************{ 2D Plot of PCA components }******************************
		# PCA2D_visualization(Norm_Before, 5)
		# PCA2D_visualization(Norm_After, 5)

		#***************************{ Training AR without PCA }**********************************************
		ARBefore = np.array(AR_Train( BefCal, AR_Order ))
		ARAfter = np.array(AR_Train( AftCal, AR_Order ))
		
		Flatten_ARB[index] = ARBefore.flatten()
		Flatten_ARA[index] = ARAfter.flatten()
	

	print ('ARBefore flatten = ' + str(np.shape(Flatten_ARB)))

	#Hist_of_Components(BefCal, AftCal)
	kernel = rbf_kernel(Flatten_ARB, Flatten_ARB, gamma= 0.1)
	print ( np.shape(kernel))

	# Finding the eigen vectores abd eigenvalues ...

	evals, evecs =  np.linalg.eig(kernel)
	evals = np.sort(evals) 

	print ('Eig values :'+ str(evals))
	print ('Eig vectors :'+ str(evecs))
	
	#Number of eigenvectors to show ...
	k = 10
	#plot KERNEL ...
	plt.imshow(kernel)
	plt.colorbar()

	#plot eigenvectors ...
	plt.figure(0)
	for i in range(k):
		plt.plot(evecs[:, i])
	plt.show()

	
	# Feeding the data to spectral clustering module ....
	sc = cluster.SpectralClustering(n_clusters = 5, affinity = 'precomputed')
	sc.fit(kernel)

	y = sc.labels_
	print (y)

	
	# Plotting the trajectories and showing the clustering using a specific color for each label
	col = ['red', 'black', 'blue', 'green', 'cyan']

	for i in range(139):
		plt.plot(XB[i],YB[i] , color= col[y[i]])
	plt.show()

	
	# Saving the labels in a file
	labelsbefore = np.asarray(y)
	np.savetxt('labelsbefore.txt',labelsbefore, delimiter=',')


	# Plotting each cluster individually 

	for i in range(139):
		if ( col[y[i]] == 'red'):
			plt.plot(XB[i],YB[i] , color= col[y[i]])
	plt.show()
	
	for i in range(139):
		if ( col[y[i]] == 'black'):
			plt.plot(XB[i],YB[i] , color= col[y[i]])
	plt.show()
	
	for i in range(139):	
		if ( col[y[i]] == 'blue'):
			plt.plot(XB[i],YB[i] , color= col[y[i]])
	plt.show()
	for i in range(139):
		if ( col[y[i]] == 'green'):
			plt.plot(XB[i],YB[i] , color= col[y[i]])
	plt.show()
	for i in range(139):
		if ( col[y[i]] == 'cyan'):
			plt.plot(XB[i],YB[i] , color= col[y[i]])
	plt.show()

	
	# This part is just to check each cluster individually and is completely arbitrary ...

	font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
	
	for i in range(0, 139):
		plt.plot(XB[i],YB[i] , linewidth = 2, color= col[y[i]])
		plt.title('object number ' + str(i) + ' cluster number ='+str(y[i]), fontdict=font)
		plt.show()


	#print (len(sc.labels_))

	# labels = sc
	# data = kernel #np.stack([XA, YA])
	# palette = sns.color_palette('deep', np.unique(labels).max() + 1)
	# colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
	# plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
	# frame = plt.gca()
	# frame.axes.get_xaxis().set_visible(False)
	# frame.axes.get_yaxis().set_visible(False)
	# plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)


# Cluster the eigenvectors.
# 	kmeans = cluster.KMeans(n_clusters = k)
# 	kmeans.fit(Ker_Eigvec)
# 	y = kmeans.labels_
# 	print(len(y))

# 	print (y)

# 	plt.scatter(Ker_Eigvec[:, 0],Ker_Eigvec[:, 1], s = 20, c = y)
# 	plt.show()


#***************************{ Histogram of components before and after calcium }*********
#***************************{ Training AR After PCA }**********************************************
# 	ARBefore = AR_Train( Norm_Before, 2 )
# 	ARAfter = AR_Train( Norm_After, 2 )
# 	kernel = rbf_kernel(ARBefore[0], ARAfter[1], gamma=10)
# 	print (kernel)
# 	print ( np.shape(kernel))
# #***************************{ Histogram of components before and after calcium }*********
# 	Hist_of_Components(Norm_Before, Norm_After)


if __name__ == "__main__":
	main()
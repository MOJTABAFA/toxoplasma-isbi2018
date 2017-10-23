from __future__ import division
import numpy as np 
import seaborn as sns
import sys
from sys import platform as sys_pf
if sys_pf == 'Darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

def visualize_cluster(No_of_traj, l_no, nrows, ncols, x, y, c, col):
	Number_of_traj = No_of_traj
	label_no =l_no
	counter = 0
	alpha = c[label_no] - 1
	fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True)

	for row in axes:
		for cols in range(ncols):
			if (counter > alpha):
				break
			X = x[counter]
			Y = y[counter]

			row[cols].plot(X, Y, color=col[int(label_no)])
			counter = counter + 1

	fig.suptitle('Trajectories of Cluster '+str(label_no), fontsize='large')

	plt.show()


def main():

	#loading files ...

	labels = np.loadtxt('labels.txt', delimiter=',')
	labelsbefore = np.loadtxt('labelsbefore.txt', delimiter=',')
	XA = np.loadtxt('XA.txt', delimiter=',') 
	XB = np.loadtxt('XB.txt', delimiter=',') 
	YA = np.loadtxt('YA.txt', delimiter=',') 
	YB = np.loadtxt('YB.txt', delimiter=',') 
	Number_of_traj = np.shape(XA)[0]
	Number_of_frames = np.shape(XA)[1]

	
	col = ['red', 'black', 'blue', 'green', 'cyan']

	c = np.zeros( shape=(5), dtype=int)

	for i in range(139):
		if (int(labels[i]) == 0) : c[0] += 1
		elif (int(labels[i]) == 1) : c[1] += 1
		elif (int(labels[i]) == 2) : c[2] += 1
		elif (int(labels[i]) == 3) : c[3] += 1
		elif (int(labels[i]) == 4) : c[4] += 1
	

	C0x = np.zeros(shape=(c[0],Number_of_frames))
	C1x = np.zeros(shape=(c[1],Number_of_frames))
	C2x = np.zeros(shape=(c[2],Number_of_frames))
	C3x = np.zeros(shape=(c[3],Number_of_frames))
	C4x = np.zeros(shape=(c[4],Number_of_frames))

	C0y = np.zeros(shape=(c[0],Number_of_frames))
	C1y = np.zeros(shape=(c[1],Number_of_frames))
	C2y = np.zeros(shape=(c[2],Number_of_frames))
	C3y = np.zeros(shape=(c[3],Number_of_frames))
	C4y = np.zeros(shape=(c[4],Number_of_frames))

	C0xb = np.zeros(shape=(c[0],Number_of_frames))
	C1xb = np.zeros(shape=(c[1],Number_of_frames))
	C2xb = np.zeros(shape=(c[2],Number_of_frames))
	C3xb = np.zeros(shape=(c[3],Number_of_frames))
	C4xb = np.zeros(shape=(c[4],Number_of_frames))

	C0yb = np.zeros(shape=(c[0],Number_of_frames))
	C1yb = np.zeros(shape=(c[1],Number_of_frames))
	C2yb = np.zeros(shape=(c[2],Number_of_frames))
	C3yb = np.zeros(shape=(c[3],Number_of_frames))
	C4yb = np.zeros(shape=(c[4],Number_of_frames))


	index = np.zeros( shape=(5), dtype= int)
	
	for trajectory in range(139):
		if (col[int(labels[trajectory])]) == 'red' :
			C0x[index[0]] = XA[trajectory]
			C0y[index[0]] = YA[trajectory]
			C0xb[index[0]] = XB[trajectory]
			C0yb[index[0]] = YB[trajectory]
			
			index[0] +=1
		
		elif (col[int(labels[trajectory])]) == 'black' :
			C1x[index[1]] = XA[trajectory]
			C1y[index[1]] = YA[trajectory]
			C1xb[index[1]] = XB[trajectory]
			C1yb[index[1]] = YB[trajectory]

			index[1] +=1
		
		elif (col[int(labels[trajectory])]) == 'blue' :
			C2x[index[2]] = XA[trajectory]
			C2y[index[2]] = YA[trajectory]
			C2xb[index[2]] = XB[trajectory]
			C2yb[index[2]] = YB[trajectory]

			index[2] +=1
		
		elif (col[int(labels[trajectory])]) == 'green' :
			C3x[index[3]] = XA[trajectory]
			C3y[index[3]] = YA[trajectory]
			C3xb[index[3]] = XB[trajectory]
			C3yb[index[3]] = YB[trajectory]

			index[3] +=1
		
		else :
			C4x[index[4]] = XA[trajectory]
			C4y[index[4]] = YA[trajectory]
			C4xb[index[4]] = XB[trajectory]
			C4yb[index[4]] = YB[trajectory]

			index[4] +=1

	print (index)

	visualize_cluster(Number_of_traj, 0, 5, 6, C0xb, C0yb, c, col)
	visualize_cluster(Number_of_traj, 1, 6, 8, C1xb, C1yb, c, col)
	visualize_cluster(Number_of_traj, 2, 3, 6, C2xb, C2yb, c, col)
	visualize_cluster(Number_of_traj, 3, 3, 2, C3xb, C3yb, c, col)
	visualize_cluster(Number_of_traj, 4, 5, 8, C4xb, C4yb, c, col)

	for Trajectories in range (Number_of_traj):
		print (Trajectories, labelsbefore[Trajectories], labels[Trajectories])


if __name__ == "__main__":
	main()




***This Repository is created to share the codes for ISBI 2018 paper with title of _"UNSUPERVISED DISCOVERY OF TOXOPLASMA GONDII MOTILITY PHENOTYPES"_***

We implemented our pipeline using Python 3 and associated scientific computing libraries ***(NumPy, SciPy, scikit-learn,
matplotlib)*** .The core of our tracking algorithm used a combination of tools available in the OpenCV 3.1 computer vision library. 

To use the code, First the coordinates of trajectories should be extracted using Tracking algorithm.( Here we extracted them through our KLT based tracking algorithm and saved them in text file ). we have 6 files for 2 sets of Data: 

* ** XA ** : Denotes the ** X ** coordinations on **After Calcium** data set.
* ** YA ** : Denotes the ** X ** coordinations on **After Calcium** data set.
* ** XB ** : Denotes the ** X ** coordinations on **Before Calcium** data set.
* ** YB ** : Denotes the ** X ** coordinations on **Before Calcium** data set.
* ** AA ** : Denotes the ** angles ** of the objects movement in 2 consecutive frames on **After Calcium** data set.
* ** AB ** : Denotes the ** angles ** of the objects movement in 2 consecutive frames on **Before Calcium** data set.

Then you should run the **RBF_Clustering.py** to load the files, Extract the AR parameters, making the RBF Kernel using eigenvectors and finally cluster the trajectories.

After running **RBF_Clustering.py** It will create a **label.txt** file in the same directory. This file shows the labels for each cluster and will be used in **RBF_Visualization.py** to visualize the clusters of the trajectories. 

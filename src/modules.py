import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

N=100
FILENAME='data_points.csv'

class PolarCoordinates():
    def __init__(self,N,r,theta):
        self.N=N
        self.r=r
        self.theta=theta
        self.x=r*np.cos(self.theta)
        self.y=r*np.sin(self.theta)

def generate_concentric_circles_data(r_min,r_max,N):
    np.random.seed(2003)
    r=np.random.uniform(r_min,r_max,N)
    theta=np.random.uniform(0.0,2*np.pi,N)
    dataPoints=PolarCoordinates(N,r,theta)
    return dataPoints

def concatenate_concentric_circles_data(dataPoints_1,dataPoints_2):
    x=np.concatenate([dataPoints_1.x,dataPoints_2.x])
    y=np.concatenate([dataPoints_1.y,dataPoints_2.y])
    return x,y

def datapoints_to_csv(x,y,filename):
    if os.path.exists(FILENAME):
        with open(filename,'w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['x','y'])
            for i in range(2*N):
                writer.writerow([x[i],y[i]])

def draw_datapoints(x,y):
    plt.scatter(x,y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def csv_to_ndarray(filename):
    cust_df=pd.read_csv(filename)
    data_points_array=np.array([cust_df['x'].tolist(),cust_df['y'].tolist()])
    data_points_array=data_points_array.T
    return data_points_array

def gaussian_similarity_func(data_point_array_1,data_point_array_2):
    sigma=1.0
    return np.exp(-((data_point_array_1[0]-data_point_array_2[0])**2+(data_point_array_1[1]-data_point_array_2[1])**2)/(2*(sigma**2)))

def calc_similarity_matrix(similarity_func,data_points_array,N):
    similarity_matrix=np.empty((N,N))
    for i in range(N):
        for j in range(N):
            similarity_matrix[i][j]=similarity_func(data_points_array[i],data_points_array[j])
    return similarity_matrix

def eigenvector(mat,k):
    eigvals,eigvecs=np.linalg.eigh(mat)
    idx=np.argsort(eigvals)
    eigvecs=eigvecs[:, idx]
    return eigvecs[:, 0:k]

def spectral_clustering(k,similarity_graph):
    v=eigenvector(similarity_graph.laplacian(),k)
    predict_spectral=KMeans(k).fit_predict(v)
    return predict_spectral

def show_clustering_result(title,data_points_array,predict):
    plt.scatter(data_points_array[:,0],data_points_array[:,1],c=predict[:])
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

#CSClass2025のゼミの複雑ネットワークでは次数リストをコンストラクタで持っていたが，この版では関数で持って使うときに計算しているので，逆輸入すると誤作動を起こしかねない

class UndirectedGraph():

    def __init__(self,N):
        #print("頂点数Nを入力してください．")
        #self.N=int(input())
        self.N=N
        self.Adj_mat=np.zeros((self.N,self.N)) #隣接行列
        self.Adj_list=[[]for i in range(self.N)] #隣接リスト
        self.Node_list=[] #ノード番号のリスト
        for i in range(self.N):
            self.Node_list.append(i)
        self.Edge_list=[] #存在するエッジのリスト
        self.Inter_node_distance=[] #ノード間距離のリスト

    #指定したリンクを追加する関数
    def add_link(self,i,j,weight):
        self.Adj_mat[i][j]=weight
        self.Adj_mat[j][i]=weight
        self.Adj_list[i].append(j)
        self.Adj_list[j].append(i)
        self.Edge_list.append([i,j])
        #self.Edge_list.append([j,i])

    #指定したリンクを削除する関数
    def delete_link(self,i,j):
        temp=Adj_mat[i][j]
        self.Adj_mat[i][j]=0
        self.Adj_mat[j][i]=0
        self.Adj_list[i].remove(j)
        self.Adj_list[j].remove(i)
        self.Edge_list.remove([i,j])
        #self.Edge_list.remove([j,i])

    def degree(self):
        A=self.Adj_mat
        D=np.sum(A,axis=1)
        return D

    def laplacian(self):
        A=self.Adj_mat
        D=np.diag(self.degree())
        return D-A

    def draw_graph(self):
        self.G_draw=nx.Graph()
        self.G_draw.add_nodes_from(self.Node_list)
        self.G_draw.add_edges_from(self.Edge_list)

        nx.draw_circular(self.G_draw,with_labels=True,nodelist=self.Node_list)
        plt.show()

    def BFS(self):
        Distance=np.full((self.N,self.N),-1)
        for i in range(self.N):
            Distance[i][i]=0
            Visited_node=[]
            Visited_node.append(i)
            while Visited_node:
                u=Visited_node.pop(0)
                for v in self.Adj_list[u]:
                    if Distance[i][v]!=-1: continue
                    Distance[i][v]=Distance[i][u]+1
                    Visited_node.append(v)
        return Distance
    
    #平均距離Lを計算する関数
    def calc_average_distance(self):
        self.Inter_node_distance=self.BFS()
        sum=0
        count=0
        for i in range(self.N):
            for j in range(i+1,self.N):
                d=self.Inter_node_distance[i][j]
                if d!=-1:
                    sum+=d
                    count+=1
        return (sum)/(count)
    
    #クラスター係数Cを計算する関数
    def calc_cluster_coefficient(self):
        Degree_of_node=self.degree()
        cluster_coefficient=0
        for i in range(self.N):
            number_of_triangle=0
            d=self.Degree_of_node[i]
            if d<2: continue
            Neighbors=self.Adj_list[i]
            for j in range(len(Neighbors)):
                for k in range(j+1,len(Neighbors)):
                    if Neighbors[k] in self.Adj_list[Neighbors[j]]: number_of_triangle+=1
            cluster_coefficient+=((number_of_triangle)/((d*(d-1))/2))
        cluster_coefficient/=self.N
        return cluster_coefficient

class SimilarityGraphKnn(UndirectedGraph):
    def __init__(self,N,k,similarity_matrix,data_points_array):
        super().__init__(N)
        self.k=k
        self.similarity_matrix=similarity_matrix
        self.generate_graph()
        self.draw_graph(data_points_array)
    def generate_graph(self):
        for i in range(self.N):
            idx_desc=np.argsort(self.similarity_matrix[i])[::-1]
            idx_desc=np.delete(idx_desc,0)
            for j in range(self.k):
                self.add_link(i,idx_desc[j],self.similarity_matrix[i][idx_desc[j]])

    def draw_graph(self,data_points_array):
        self.G_draw=nx.Graph()
        self.G_draw.add_nodes_from(self.Node_list)
        self.G_draw.add_edges_from(self.Edge_list)
        pos = {i: (data_points_array[i][0], data_points_array[i][1]) for i in range(self.N)}
        nx.draw(self.G_draw,pos,with_labels=True,node_size=70,nodelist=self.Node_list)
        plt.title("Similarity Graph by knn")
        plt.show()

class SimilarityGraphComplete(UndirectedGraph):
    def __init__(self,N,similarity_matrix,data_points_array):
        super().__init__(N)
        self.similarity_matrix=similarity_matrix
        self.generate_graph()
        self.draw_graph(data_points_array)
    def generate_graph(self):
        for i in range(self.N):
            for j in range(i+1,self.N):
                self.add_link(i,j,self.similarity_matrix[i][j])

    def draw_graph(self,data_points_array):
        self.G_draw=nx.Graph()
        self.G_draw.add_nodes_from(self.Node_list)
        self.G_draw.add_edges_from(self.Edge_list)
        pos = {i: (data_points_array[i][0], data_points_array[i][1]) for i in range(self.N)}
        nx.draw(self.G_draw,pos,with_labels=True,node_size=70,nodelist=self.Node_list)
        plt.title("Similarity Graph by complete")
        plt.show()

class ExpansionCycle(UndirectedGraph):

    def __init__(self,N,k_ave):
        super().__init__(N)
        #print("平均次数<k>を入力してください．平均次数は偶数にしてください．")
        #self.k_ave=int(input())
        self.k_ave=k_ave
        self.number_of_edges=int(self.k_ave*self.N/2)
        #隣接行列の作成
        for i in range(self.N):
            for j in range(int(self.k_ave/2)):
                self.Adj_mat[i][(i+j+1)%self.N]=1
                self.Adj_mat[i][(self.N-1+i-j)%self.N]=1
        #print(Adj_mat)
        #隣接リストの作成
        for i in range(self.N):
            for j in range(self.N):
                if self.Adj_mat[i][j]==1:
                    self.Adj_list[i].append(j)
        #print(Adj_list)
        #エッジリストの作成
        for i in range(self.N):
            for j in range(i+1,self.N):
                if self.Adj_mat[i][j]:
                    self.Edge_list.append([i,j])
        #print(Edge_list)

class WsModel(ExpansionCycle):

    def __init__(self,p,N,k_ave):
        super().__init__(N,k_ave)
        self.p=p
        self.Selected_edges_list=random.sample(self.Edge_list,int(self.p*self.number_of_edges)) #モンテカルロ法を用いてみる
        for u in self.Selected_edges_list:
            i,j=u[0],u[1]
            self.delete_link(i,j)
            while(True):
                new_node=random.randint(0,self.N-1)
                if(new_node==i or new_node==j): continue
                else: break
            cut_node=random.randint(0,1)
            if cut_node==0: self.add_link(i,new_node,1)
            else: self.add_link(new_node,j,1)     
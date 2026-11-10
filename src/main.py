import modules
import GraphModel

#==============================パラメータ設定==============================
k = 2
nearest_k = 10
N = modules.N
generator_flag = True
#=========================================================================
if(generator_flag):
    dataPoints_1 = modules.generate_concentric_circles_data(0.5, 1.0, N)
    dataPoints_2 = modules.generate_concentric_circles_data(1.5, 2.0, N)
    x, y = modules.concatenate_concentric_circles_data(dataPoints_1, dataPoints_2)

    modules.datapoints_to_csv(x, y, modules.FILENAME)

data_points_array = modules.csv_to_ndarray(modules.FILENAME)

"""
#===========================比較用k-means==================================
predict_kmeans = KMeans(2).fit_predict(data_points_array)

show_clustering_result('kmeans Clustering', predict_kmeans)
#=========================================================================
"""

#==============================類似度行列計算===============================
#ガウシアン類似度関数
similarity_matrix = modules.calc_similarity_matrix(modules.gaussian_similarity_func, data_points_array,200)

#=========================================================================

#==============================類似度グラフ生成=============================
#k近傍グラフ
similarity_graph_knn = GraphModel.SimilarityGraphKnn(2 * N, nearest_k, similarity_matrix, data_points_array)

#完全グラフ
similarity_graph_complete = GraphModel.SimilarityGraphComplete(2 * N, similarity_matrix, data_points_array)
#=========================================================================

predict_spectral_knn = modules.spectral_clustering(k, similarity_graph_knn)
predict_spectral_complete = modules.spectral_clustering(k, similarity_graph_complete)

modules.show_clustering_result('Spectral Clustering knn', data_points_array,predict_spectral_knn)
modules.show_clustering_result('Spectral Clustering complete', data_points_array,predict_spectral_complete)
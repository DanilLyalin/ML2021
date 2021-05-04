from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from skimage import io
from sklearn.cluster import MiniBatchKMeans

TASK_NUM = 4

def view_scores(res, _name, _type, metric):
	plt.plot(list(res.keys()), list(res.values()))
	plt.title(_name + ' ' + _type)
	plt.xlabel("N clusters")
	plt.ylabel(metric)
	plt.show()


if __name__ == "__main__":
	if TASK_NUM == 1:
		for j in range(0, 2):
		    # read, standardize
		    X = read_csv('pluton.csv', sep=',').to_numpy()
		    scale = StandardScaler()
		    scaled_data = scale.fit_transform(X)
		    pca = PCA(n_components=2)
		    title = 'Non-standardize'
		    X_principal = pca.fit_transform(X)
		    if j == 1:
		        title = 'Standardize'
		        X_principal = pca.fit_transform(scaled_data)
		    X_principal = pd.DataFrame(X_principal)
		    X_principal.columns = ['P1', 'P2']
		    # visualize
		    plt.scatter(X_principal['P1'], X_principal['P2'],
		                c=KMeans(n_clusters=3).fit_predict(X_principal))
		    plt.title(title)
		    plt.show()

		    # scores
		    silhouette_score = {}
		    davies_bouldin_score = {}
		    calinski_harabasz_score = {}
		    for i in range(1, 10):
		        silhouette_score[i] = metrics.silhouette_score(X_principal,
		                                                       KMeans(n_clusters=3, max_iter=i).fit_predict(X_principal))
		        davies_bouldin_score[i] = metrics.davies_bouldin_score(X_principal,
		                                                               KMeans(n_clusters=3, max_iter=i).fit_predict(
		                                                                   X_principal))
		        calinski_harabasz_score[i] = metrics.calinski_harabasz_score(X_principal,
		                                                                     KMeans(n_clusters=3, max_iter=i).fit_predict(
		                                                                         X_principal))

		    # plots for scores
		    plt.plot(list(silhouette_score.keys()), list(silhouette_score.values()))
		    plt.title(title)
		    plt.xlabel("Max iterations")
		    plt.ylabel("Silhouette score")
		    plt.show()

		    plt.plot(list(davies_bouldin_score.keys()), list(davies_bouldin_score.values()))
		    plt.title(title)
		    plt.xlabel("Max iterations")
		    plt.ylabel("Davies-Bouldin score")
		    plt.show()
		    plt.plot(list(calinski_harabasz_score.keys()), list(calinski_harabasz_score.values()))
		    plt.title(title)
		    plt.xlabel("Max iterations")
		    plt.ylabel("Calinski_Harabasz score")
		    plt.show()
	if TASK_NUM == 2:
		csv_names = ['clustering_1.csv', 'clustering_2.csv', 'clustering_3.csv']

		for name in csv_names:
			# load data
			X = read_csv(name, sep='\t').to_numpy()
			scale = StandardScaler()
			scaled_data = scale.fit_transform(X)

			X_principal = pd.DataFrame(scaled_data)
			X_principal.columns = ['P1', 'P2']

			# visualize
			i = 0
			if name == 'clustering_1.csv':
				i = 2
			if name == 'clustering_2.csv':
				i = 3
			if name == 'clustering_3.csv':
				i = 5

			plt.scatter(X_principal['P1'], X_principal['P2'],
						c=AgglomerativeClustering(n_clusters=i).fit_predict(X_principal))
			plt.title(name + ' Hierarchical')
			plt.show()

			plt.scatter(X_principal['P1'], X_principal['P2'],
						c=KMeans(n_clusters=i).fit_predict(X_principal))
			plt.title(name + ' KMeans')
			plt.show()

			# find scores

			# DBSCAN score
			# not scaled data for DBSCAN
			X_principal = pd.DataFrame(X)
			X_principal.columns = ['P1', 'P2']

			plt.scatter(X_principal['P1'], X_principal['P2'],
						c=DBSCAN().fit_predict(X_principal))
			plt.title(name + ' DBSCAN')
			plt.show()

			clustering = DBSCAN().fit(X_principal)
			labels = clustering.labels_
			n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
			print(n_clusters_)
			if n_clusters_ > 1:
				print(metrics.silhouette_score(X_principal, DBSCAN().fit_predict(X_principal)))
				print(metrics.davies_bouldin_score(X_principal, DBSCAN().fit_predict(X_principal)))
				print(metrics.calinski_harabasz_score(X_principal, DBSCAN().fit_predict(X_principal)))

			# scaled data for KMeans and hierarchical scores
			X_principal = pd.DataFrame(scaled_data)
			X_principal.columns = ['P1', 'P2']
			# KMeans and hierarchical scores
			silhouette_score_k_means = {}
			davies_bouldin_score_k_means = {}
			calinski_harabasz_score_k_means = {}
			silhouette_score_hierarchical = {}
			davies_bouldin_score_hierarchical = {}
			calinski_harabasz_score_hierarchical = {}
			for i in range(2, 10):
				# KMeans
				silhouette_score_k_means[i] = metrics.silhouette_score(X_principal,
																	   KMeans(n_clusters=i).fit_predict(X_principal))
				davies_bouldin_score_k_means[i] = metrics.davies_bouldin_score(X_principal,
																			   KMeans(n_clusters=i).fit_predict(
																				   X_principal))
				calinski_harabasz_score_k_means[i] = metrics.calinski_harabasz_score(X_principal,
																					 KMeans(n_clusters=i).fit_predict(
																						 X_principal))
				# Hierarchical
				silhouette_score_hierarchical[i] = metrics.silhouette_score(X_principal,
																			AgglomerativeClustering(n_clusters=i).fit_predict(
																				X_principal))
				davies_bouldin_score_hierarchical[i] = metrics.davies_bouldin_score(X_principal,
																					AgglomerativeClustering(
																						n_clusters=i).fit_predict(
																						X_principal))
				calinski_harabasz_score_hierarchical[i] = metrics.calinski_harabasz_score(X_principal,
																						  AgglomerativeClustering(
																							  n_clusters=i).fit_predict(
																							  X_principal))

			# plots for scores k_means
			view_scores(silhouette_score_k_means, name, 'KMeans', "Silhouette score")
			view_scores(davies_bouldin_score_k_means, name, 'KMeans', "Davies-Bouldin score")
			view_scores(calinski_harabasz_score_k_means, name, 'KMeans', "Calinski_Harabasz score")

			# plots for scores hierarchical
			view_scores(silhouette_score_hierarchical, name, 'Hierarchical', "Silhouette score")
			view_scores(davies_bouldin_score_hierarchical, name, 'Hierarchical', "Davies-Bouldin score")
			view_scores(calinski_harabasz_score_hierarchical, name, 'Hierarchical', "Calinski_Harabasz score")
	if TASK_NUM == 3:
		picture = io.imread('test.jpg')
		ax = plt.axes(xticks=[], yticks=[])
		ax.imshow(picture)
		print(picture.shape)

		data = picture / 255.0  
		data = data.reshape(228 * 512, 3)

		kmeans = MiniBatchKMeans(16)
		kmeans.fit(data)
		new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

		china_recolored = new_colors.reshape(picture.shape)

		fig, ax = plt.subplots(1, 2, figsize=(16, 6),
							   subplot_kw=dict(xticks=[], yticks=[]))
		fig.subplots_adjust(wspace=0.05)
		ax[0].imshow(picture)
		ax[0].set_title('Original', size=16)
		ax[1].imshow(china_recolored)
		ax[1].set_title('16-color', size=16)
		plt.show()
	if TASK_NUM == 4:
		X = read_csv('votes.csv', sep=',').to_numpy()
		X = np.nan_to_num(X, nan=0)

		plt.figure(figsize=(20, 20))
		plt.grid(True)
		plt.title('Hierarchical Clustering Dendrogram')

		Z = linkage(X)

		dendrogram(Z, truncate_mode='level', p=10)
		plt.show()
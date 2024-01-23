# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WMQ2xtZM2M3BOjClHwKNhui3ZihaSb16
"""

import numpy as np
from tqdm import tqdm
from time import time
from os import getpid
from threading import Thread
from sklearn.manifold import TSNE

class AntCluster:
    def __init__(self, n_ants, n_clusters, data, max_iter=100, alpha=1.0, beta=2.0, rho=0.5):
        self.n_ants = n_ants
        self.n_clusters = n_clusters
        self.data = data
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.workers = {}
        self.pheromone_matrix = np.ones((len(data), n_clusters))
        self.best_solution = None
        self.best_cost = float('inf')

    def initialize_clusters(self):
        # Randomly initialize clusters for each data point
        return np.random.randint(0, self.n_clusters, len(self.data))

    def calculate_distance(self, point, centroid):
        # Euclidean distance calculation
        return np.linalg.norm(point - centroid)

    def update_pheromone(self, solutions, costs):
        # Pheromone update rule
        delta_pheromone = np.zeros_like(self.pheromone_matrix)
        for i, solution in enumerate(solutions):
            for j, cluster in enumerate(solution):
                delta_pheromone[j, cluster] += 1 / costs[i]

        self.pheromone_matrix = (1 - self.rho) * self.pheromone_matrix + delta_pheromone

    def run(self):
        n_threads = 1
        threads = []
        for t in range(n_threads):
            thread = Thread(target=self.costly_function, args=(self.max_iter//n_threads,))
            threads.append(thread)
            thread.start()
        for t in threads:
            t.join()
        return self.best_solution, self.best_cost
        #return self.costly_function()


    def costly_function(self, n, e=None):
        for iteration in tqdm(range(n)):
            solutions = []
            costs = []
            for ant in range(self.n_ants):
                solution = self.explore()
                cost = self.calculate_cost(solution)
                solutions.append(solution)
                costs.append(cost)
                if cost < self.best_cost:
                    self.best_solution = solution
                    self.best_cost = cost

            self.update_pheromone(solutions, costs)

        return self.best_solution, self.best_cost


    def explore(self):
        # Exploration phase
        clusters = self.initialize_clusters()
        for ant in range(self.n_ants):
            for i, point in enumerate(self.data):
                probabilities = self.calculate_probabilities(i, clusters)
                cluster = np.random.choice(np.arange(self.n_clusters), p=probabilities)
                clusters[i] = cluster
        return clusters

    def calculate_probabilities(self, i, clusters):
        # Calculate probabilities for selecting clusters in the exploration phase
        probabilities = np.zeros(self.n_clusters)
        for j in range(self.n_clusters):
            probabilities[j] = (self.pheromone_matrix[i, j] ** self.alpha) * \
                               ((1 / self.calculate_distance(self.data[i], self.calculate_centroid(i, j, clusters))) ** self.beta)
        probabilities /= np.sum(probabilities)
        return probabilities

    def calculate_centroid(self, i, cluster, clusters):
        # Calculate the centroid of points in the same cluster
        cluster_points = self.data[clusters == cluster]
        if len(cluster_points) == 0:
            return self.data[i]  # If no points in the cluster, return the current point
        return np.mean(cluster_points, axis=0)

    def calculate_cost(self, clusters):
        # Calculate the total cost of the solution
        cost = 0
        for i, point in enumerate(self.data):
            cost += self.calculate_distance(point, self.calculate_centroid(i, clusters[i], clusters))
        return cost


if __name__ == '__main__':
    np.random.seed(42)

    # Create a simple dataset with two clusters
    data_cluster1 = np.random.rand(50, 5) * 2
    data_cluster2 = np.random.rand(50, 5) * 2 + np.array([1, 1, 1, 1, 1])
    data = np.vstack([data_cluster1, data_cluster2])
    print(f"{data =}")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    print(f"{tsne_results = }")
    # prompt: plot data
    #print(f"{tsne_results = }")
    import matplotlib.pyplot as plt

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.show()
    
    # # Set parameters and run the Ant Cluster Algorithm
    n_ants = 10
    n_clusters = 2
    ant_cluster = AntCluster(n_ants, n_clusters, data)
    start = time()
    best_solution, best_cost = ant_cluster.run()
    end = time()
    
    print(f"{format(end-start, '0.2f')}s")
    # 
    print("Best solution (cluster assignments):", best_solution)
    print("Best cost:", best_cost)

    # prompt: plot data with best_solution in mind where best_solution returns a list of indices that separate the data by 2 classes. Give each class a different color.

    colors = []
    for i in range(len(data)):
        if best_solution[i] == 1:
            colors.append('red')
        elif best_solution[i] == 2:
            colors.append('green')
        elif best_solution[i] == 0:
            colors.append('blue')
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors)
    plt.show()
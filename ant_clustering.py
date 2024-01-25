import numpy as np
from tqdm import tqdm
from time import time
from os import getpid
from threading import Thread
from sklearn.manifold import TSNE
import streamlit as st
import plotly.express as px

class AntCluster:
    def __init__(self, n_ants, n_clusters, data, max_iter=100, alpha=1.0, beta=2.0, rho=0.5, threads=4):
        self.n_ants = n_ants
        self.n_clusters = n_clusters
        self.data = data
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_threads = threads
        self.workers = {}
        self.pheromone_matrix = np.ones((len(data), n_clusters))
        self.best_solution = [0 for x in range(len(data))]
        self.flag = 0
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

    def run(self, transformed_tsne):
        threads = []
        self.placeholder = st.empty()
        self.colors = [
            "#FF5733",  # Red Orange
            "#33FF57",  # Neon Green
            "#3357FF",  # Blue
            "#F033FF",  # Magenta
            "#33FFF5",  # Cyan
            "#F5FF33",  # Yellow
            "#FF3380",  # Pink
            "#80FF33",  # Light Green
            "#FF9633",  # Orange
            "#33FF80",  # Mint
            "#7F33FF",  # Purple
            "#FF337F",  # Rose
            "#33A2FF",  # Sky Blue
            "#A233FF",  # Violet
            "#33FFA2",  # Aquamarine
            "#FFA233",  # Amber
            "#8CFF33",  # Lime Green
            "#FF338C",  # Hot Pink
            "#337FFF",  # Azure
            "#FF5733"   # Crimson Red
        ]
        for t in range(self.num_threads):
            thread = Thread(target=self.costly_function, args=(self.max_iter//self.num_threads,transformed_tsne))
            threads.append(thread)
            thread.start()
        for t in threads:
            t.join()
        return self.best_solution, self.best_cost
        #return self.costly_function()


    def costly_function(self, n, transformed_tsne, e=None):
        for iteration in tqdm(range(n)):
            c = []
            for i in range(len(self.data)):
                for j in range(len(self.colors)):
                    if self.best_solution[i] == j:
                        c.append(f"class {self.best_solution[i]}")
                        break
            transformed_tsne['color'] = c
            self.color_map = {f"class {i}": self.colors[i] for i in range(self.n_clusters)}
            self.fig = px.scatter(transformed_tsne, x="x", y="y", color="color", color_discrete_map=self.color_map)
            self.fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
            with self.placeholder.container():
                st.plotly_chart(self.fig, use_container_width=True)
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
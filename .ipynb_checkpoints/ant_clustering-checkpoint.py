
import numpy as np
from tqdm import tqdm
from time import time
from os import getpid
from threading import Thread
from sklearn.manifold import TSNE
import streamlit as st
import plotly.express as px
from streamlit.runtime.scriptrunner import add_script_run_ctx as add_report_ctx

class AntCluster:
    def __init__(self, n_ants, n_clusters, data, max_iter=100, alpha=1.0, beta=2.0, rho=0.5, threads=4):
        # Init: Initialisation de l'objet AntCluster avec des propre paramètres
        self.n_ants = n_ants
        self.n_clusters = n_clusters
        self.data = data
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_threads = threads 
        self.stop_counter = 0
        self.workers = {}
        self.pheromone_matrix = np.ones((len(data), n_clusters))
        self.best_solution = [0 for x in range(len(data))]
        self.flag = 0
        self.best_cost = float('inf')
        self.colors = [
            "#FF5733",  
            "#33FF57",  
            "#3357FF",  
            "#F033FF",  
            "#33FFF5",  
            "#F5FF33",  
            "#FF3380",  
            "#80FF33",  
            "#FF9633",  
            "#33FF80",  
            "#7F33FF",  
            "#FF337F",  
            "#33A2FF",  
            "#A233FF",  
            "#33FFA2",  
            "#FFA233",  
            "#8CFF33",  
            "#FF338C",  
            "#337FFF",  
            "#FF5733"   
        ]

    def initialize_clusters(self):
        # Méthode : Initialisation aléatoire des clusters pour chaque point de données
        return np.random.randint(0, self.n_clusters, len(self.data))

    def calculate_distance(self, point, centroid):
        # Méthode : Calcul de la distance euclidienne
        return np.linalg.norm(point - centroid)

    def update_pheromone(self, solutions, costs):
        # Méthode : Règle de mise à jour des phéromones
        delta_pheromone = np.zeros_like(self.pheromone_matrix)

        for i, solution in enumerate(solutions):
            for j, cluster in enumerate(solution):
                delta_pheromone[j, cluster] += 1 / costs[i]

        self.pheromone_matrix = (1 - self.rho) * self.pheromone_matrix + delta_pheromone

    def run(self, transformed_tsne):
        # Méthode : Fonction principale, lance des threads pour exécuter costly_function en parallèle
        threads = []
        self.placeholder = st.empty()
        
        # ============================================ Exécution parallèle (Threading - level 1) ================================================= #
        for t in range(self.num_threads):
            thread = Thread(target=self.costly_function, args=(self.max_iter//self.num_threads,transformed_tsne))
            threads.append(thread)
            add_report_ctx(thread)
            thread.start()

        for t in threads:
            t.join()
        # ======================================================================================================================================== #

        self.placeholder.empty()
        return self.best_solution, self.best_cost


    def draw_plot(self, transformed_tsne):
        # Methode : Crée et affiche un graphique de dispersion 2D en utilisant plotly
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


    def launch_ants(self, ants, solutions, costs):
        # Méthode : Lance des fourmis (ants) pour explorer l'espace des solutions
        for ant in range(ants):
            solution = self.explore()
            cost = self.calculate_cost(solution)
            solutions.append(solution)
            costs.append(cost)
            if cost < self.best_cost:
                self.best_solution = solution
                self.best_cost = cost


    def costly_function(self, n, transformed_tsne, e=None):
        # Méthode : Fonction principale qui exécute l'algorithme d'optimisation
        for iteration in tqdm(range(n)):
            temp_cost = self.best_cost
            self.draw_plot(transformed_tsne)
            solutions = []
            costs = []
            # ============================================ Exécution parallèle (Threading - level 2) ================================================= #
            # Lance des threads pour exécuter la phase d'exploration en parallèle
            self.launch_threads(self.launch_ants, self.n_ants//self.num_threads, solutions, costs)
            # ======================================================================================================================================== #
            if temp_cost == self.best_cost:
                self.stop_counter += 1
                if self.stop_counter == 6:
                    break
            else:
                self.stop_counter = 0

            self.update_pheromone(solutions, costs)

        return self.best_solution, self.best_cost


    def explore(self):
        # Phase d'exploration pour trouver de nouvelles solutions
        clusters = self.initialize_clusters()
        # ============================================ Exécution parallèle (Threading - level 3) ================================================= #
        # Lance des threads pour explorer en parallèle
        self.launch_threads(self.explore_parallel, clusters, self.n_ants//self.num_threads)
        # ======================================================================================================================================== #
        
        return clusters
        
    
    def launch_threads(self, func, *args):
        # Méthode : Fonction auxiliaire pour lancer des threads pour une fonction spécifiée avec des arguments
        threads = []
        for t in range(self.num_threads):
            thread = Thread(target=func, args=(*args,))
            threads.append(thread)
            thread.start()
        for t in threads:
            t.join()


    def explore_parallel(self, clusters, ants):
        # Méthode : Exploration parallèle de l'espace des solutions
        for ant in range(ants):
            for i, point in enumerate(self.data):
                probabilities = self.calculate_probabilities(i, clusters)
                cluster = np.random.choice(np.arange(self.n_clusters), p=probabilities)
                clusters[i] = cluster


    def calculate_probabilities(self, i, clusters):
        # Méthode : Calcule les probabilités de sélection des clusters dans la phase d'exploration
        probabilities = np.zeros(self.n_clusters)
        for j in range(self.n_clusters):
            probabilities[j] = (self.pheromone_matrix[i, j] ** self.alpha) * \
                               ((1 / self.calculate_distance(self.data[i], self.calculate_centroid(i, j, clusters))) ** self.beta)
        probabilities /= np.sum(probabilities)
        return probabilities

    def calculate_centroid(self, i, cluster, clusters):
        # Méthode : Calcule le centroid des points dans le même cluster
        cluster_points = self.data[clusters == cluster]
        if len(cluster_points) == 0:
            return self.data[i]  
        return np.mean(cluster_points, axis=0)

    def calculate_cost(self, clusters):
        # Méthode : Calcule le coût total de la solution
        cost = 0
        for i, point in enumerate(self.data):
            cost += self.calculate_distance(point, self.calculate_centroid(i, clusters[i], clusters))
        return cost
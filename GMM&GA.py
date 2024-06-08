#!/usr/bin/env python
# coding: utf-8

# # Main Algorithm

# In[1]:


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import folium
from collections import deque
from tqdm.notebook import tqdm
from sklearn.mixture import GaussianMixture
import operator

# Ensure Matplotlib plots display directly in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# 1. Import Libraries and Load Data
# Data loading
xl = pd.ExcelFile('./data/EXAMPLE DATA FOR GA_TE ALGORITHM.xlsx', engine='openpyxl')
cities = xl.parse('city_ridership')
cities_dist_matrix = xl.parse('city_dist_matrix')
cities_dist_matrix.set_index('Unnamed: 0', inplace=True)

# 2. Define City Class
class City:
    def __init__(self, name, population, latitude, longitude):
        self.name = name
        self.population = population
        self.latitude = latitude
        self.longitude = longitude
    
    # Method to calculate distance between two cities
    def distance(self, city):
        return cities_dist_matrix.loc[self.name, city.name]
    
    # Method to calculate angle between two cities
    def angle(self, city):
        delta_y = city.latitude - self.latitude
        delta_x = city.longitude - self.longitude
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return (angle + 360) % 360  # Normalize angle to 0-360 degrees
    
    def __repr__(self):
        return f"{self.name} [{self.population}]"

# 3. Create City List
# Function to create a list of City objects
def create_city_list():
    city_list = []
    for i in range(len(cities)):
        city = City(name=cities['City'].iloc[i],
                    population=cities['Population'].iloc[i],
                    latitude=cities['Latitude'].iloc[i],
                    longitude=cities['Longitude'].iloc[i])
        city_list.append(city)
    return city_list

cityList = create_city_list()
centralCity = cityList[0]  # The central city is the first city in the city list

# 4. Calculate Angles and Classify Cities
# Function to calculate angle between central city and other cities
def calculate_angle(center, point):
    delta_y = point.latitude - center.latitude
    delta_x = point.longitude - center.longitude
    angle = np.arctan2(delta_y, delta_x)
    return np.degrees(angle)

# Function to classify cities based on angle using Gaussian Mixture Model
def calculate_angle_frequency_adaptive(center, points, max_components=5):
    angles = [calculate_angle(center, point) for point in points]
    angles = np.array(angles).reshape(-1, 1)

    lowest_aic = np.infty
    best_gmm = None

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(angles)
        aic = gmm.aic(angles)
        if aic < lowest_aic:
            lowest_aic = aic
            best_gmm = gmm

    labels = best_gmm.predict(angles)
    return labels, best_gmm.n_components

# 5. Plot Classified Cities
# Function to plot classified cities on a map
def plot_classified_cities(cityList, labels):
    map_obj = folium.Map(location=[cityList[0].latitude, cityList[0].longitude], zoom_start=12)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    for point, label in zip(cityList[1:], labels):
        folium.Marker(
            location=[point.latitude, point.longitude],
            popup=f"{point.name} [Population: {point.population}]",
            icon=folium.Icon(color=colors[label % len(colors)], icon='info-sign')
        ).add_to(map_obj)

    map_obj.save('classified_cities.html')

def plot_route_folium(routes, colors, filename='route_map.html'):
    map_obj = folium.Map(location=[routes[0][0].latitude, routes[0][0].longitude], zoom_start=12)
    
    for route, color in zip(routes, colors):
        points = [[city.latitude, city.longitude] for city in route if isinstance(city, City)]
        if points:
            folium.PolyLine(points, color=color, weight=2.5, opacity=1).add_to(map_obj)
            for city in route:
                if isinstance(city, City):
                    folium.Marker(
                        location=[city.latitude, city.longitude],
                        popup=f"{city.name} [Population: {city.population}]",
                        icon=folium.Icon(color=color, icon='info-sign')
                    ).add_to(map_obj)
    
    map_obj.save(filename)

# 6. Genetic Algorithm for Route Optimization
# Function to create a random route starting from the central city
def createRoute(cityList, centralCity):
    route = [centralCity] + random.sample([city for city in cityList if city != centralCity], len(cityList) - 1)
    return route

# Function to create an initial population of routes
def initialPopulation(popSize, cityList, centralCity):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList, centralCity))
    return population

# Function to rank routes based on fitness
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

# Function for selection based on fitness ranking
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

# Function to create mating pool
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

# Function to breed two parent routes
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

# Function to breed population
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

# Function to mutate an individual route
def mutate(individual, mutationRate):
    for swapped in range(1, len(individual)):  # Skip the first point
        if random.random() < mutationRate:
            swapWith = int(random.random() * (len(individual) - 1)) + 1  # Ensure not to swap with the first point
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

# Function to mutate population
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# Function to generate next generation
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

# Genetic Algorithm implementation
def geneticAlgorithm(cityList, popSize, eliteSize, mutationRate, generations, centralCity, early_stop_generations=10):
    pop = initialPopulation(popSize, cityList, centralCity)
    progress = []
    best_distance = float('inf')
    early_stop_counter = 0
    
    for i in tqdm(range(generations), desc="Generations"):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        bestRouteIndex = rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        fitness = Fitness(bestRoute)
        current_distance = fitness.routeDistance()
        progress.append(current_distance)
        
        if current_distance < best_distance:
            best_distance = current_distance
            early_stop_counter = 0  # Reset the counter if there is improvement
        else:
            early_stop_counter += 1  # Increment the counter if there is no improvement

        if early_stop_counter >= early_stop_generations:
            print(f"Early stopping at generation {i}")
            break

    return pop[bestRouteIndex], progress

# Fitness class to calculate route distance and fitness
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0
    
    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route) - 1):
                fromCity = self.route[i]
                toCity = self.route[i + 1]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

# 7. Main Execution
# Calculate angle and classify cities
labels, n_components = calculate_angle_frequency_adaptive(centralCity, cityList[1:], max_components=5)
partitions = {i: [] for i in range(n_components)}
for point, label in zip(cityList[1:], labels):
    partitions[label].append(point)

# Output classification results and check
for point, angle, label in zip(cityList[1:], [calculate_angle(centralCity, p) for p in cityList[1:]], labels):
    print(f"City: {point}, Angle: {angle}, Label: {label}")

plot_classified_cities(cityList, labels)

# Perform genetic algorithm calculations for each partition
best_routes = []
for partition in partitions.values():
    if not partition:
        continue
    partition.append(centralCity)
    best_route, _ = geneticAlgorithm(partition, popSize=100, eliteSize=20, mutationRate=0.005, generations=1000, centralCity=centralCity, early_stop_generations=150)
    best_routes.append(best_route)

# Set color list
colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'pink']

# Plot the best routes on the map
plot_route_folium(best_routes, colors=colors, filename='best_route.html')


# # Load routes above onto the real road network(have some bugs right now because of the osmnx service erors)

# In[2]:


import osmnx as ox
import networkx as nx
from folium import Map, Marker, PolyLine

# Load OSM data from local file
osm_file = './path/to/your/XXXX.osm.pbf'  # Replace with the actual file path
G = ox.graph_from_file(osm_file, network_type='drive')

# Convert city objects to the nearest road nodes
def get_nearest_nodes(city, G):
    return ox.distance.nearest_nodes(G, city.longitude, city.latitude)

# Calculate paths on the actual road network
def calculate_real_paths(G, best_routes):
    real_paths = []
    for route in best_routes:
        path = []
        for city in route:
            nearest_node = get_nearest_nodes(city, G)
            path.append(nearest_node)
        real_paths.append(path)
    return real_paths

# Plot routes
def plot_real_route_folium(G, best_routes, colors, filename='real_route_map.html'):
    map_obj = Map(location=[best_routes[0][0].latitude, best_routes[0][0].longitude], zoom_start=12)
    
    real_paths = calculate_real_paths(G, best_routes)
    
    for real_path, color in zip(real_paths, colors):
        points = []
        for node in real_path:
            point = G.nodes[node]
            points.append((point['y'], point['x']))
        
        # Add the actual path
        PolyLine(points, color=color, weight=2.5, opacity=1).add_to(map_obj)
        
        # Add city markers
        for city in best_routes[real_paths.index(real_path)]:
            Marker(
                location=[city.latitude, city.longitude],
                popup=f"{city.name} [Population: {city.population}]",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(map_obj)
    
    map_obj.save(filename)

# Example run (assuming best_routes and colors are already defined)
if G:
    plot_real_route_folium(G, best_routes, colors=colors, filename='real_route.html')


# In[ ]:





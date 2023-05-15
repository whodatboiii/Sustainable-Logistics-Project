from __future__ import print_function
import ortools
import pandas as pd
import numpy as np
import openpyxl
import warnings
import random
from ortools.sat.python import cp_model as cp
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, sin, cos, sqrt, atan2
from geopy.distance import geodesic
from geopy.distance import distance

#Stop the warnings 
warnings.simplefilter(action='ignore', category=UserWarning)


def project_data(num_customers_per_station=100):
    """Import the data"""
    database = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="Netz Bern 01.01.22 - 31.12.22",
        names=['origin', 'dest', 'count'])

    locations = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="locations",
        names=['id', 'name', 'lat', 'lon'])

    return database, locations

demand, locations = project_data()
 

demand,locations = project_data()

#Selecting the 100 most visited stations 
sorted_df = demand.sort_values(by='count', ascending=False)
best_stations = sorted_df.iloc[:410].drop_duplicates('origin').reset_index()
df = best_stations.merge(locations, left_on='origin', right_on='id')

#Visualizing our stations points 
df.plot(x="lon", y="lat", kind="scatter",
        colormap="YlOrRd")

# Convert longitude and latitude columns to radians
df['lat_rad'] = np.radians(df['lat'])
df['lon_rad'] = np.radians(df['lon'])
df

# Define a function to generate points within a circle of radius r
def generate_points(lat, lon, r, n):
    points = []
    for i in range(10):
        # Generate a random angle and distance within the circle
        u = np.random.uniform(0, 1)
        v = np.random.uniform(0, 1)
        w = r * np.sqrt(u)
        t = 2 * np.pi * v
        # Convert distance and angle to Cartesian coordinates
        x = w * np.cos(t)
        y = w * np.sin(t)
        # Convert Cartesian coordinates to latitude and longitude
        new_lat = np.degrees(np.arcsin(np.sin(lat) * np.cos(w) + np.cos(lat) * np.sin(w) * np.cos(t)))
        new_lon = np.degrees(lon + np.arctan2(np.sin(t) * np.sin(w) * np.cos(lat), np.cos(w) - np.sin(lat) * np.sin(new_lat)))
        points.append((new_lat, new_lon))
    return points

# Generate n points within 500 meters of each row in the DataFrame
n = 10 # number of points to generate
r = 0.5 # radius of circle in kilometers
points = []
for _, row in df.iterrows():
    lat, lon = row['lat_rad'], row['lon_rad']
    new_points = generate_points(lat, lon, r / 6371, n)
    for p in new_points:
        points.append((row['id'], row['lat'], row['lon'], p[0], p[1]))


 # Convert points to a new DataFrame
new_df = pd.DataFrame(points, columns=['id', 'orig_lat', 'orig_lon', 'new_lat', 'new_lon'])
new_df['distance'] = new_df.apply(lambda row: geodesic((row['orig_lat'], row['orig_lon']), (row['new_lat'], row['new_lon'])).km, axis=1)

# Filter points by distance
new_df = new_df[new_df['distance'] <= 0.5]

# Remove unnecessary columns
new_df = new_df[['id', 'new_lat', 'new_lon']]
customers = new_df.drop('id',axis=1)

#Visualizing the new customers points 
customers.plot(x="new_lon", y="new_lat", kind="scatter",
        colormap="YlOrRd")

#Creating the distance matrix (between customers and best stations)
num_customers = len(customers)
num_stations = len(df)

distance_matrix = np.zeros((num_customers, num_stations))

for i, row in customers.iterrows():
    for j, station_row in df.iterrows():
        coords_1 = (row['new_lat'], row['new_lon'])
        coords_2 = (station_row['lat'], station_row['lon'])
        dist = distance(coords_1, coords_2).m  # distance in meters
        distance_matrix[i, j] = dist

print(distance_matrix)

#Demand of our customers 
demand = list(range(num_customers))
for i in range(num_customers):
  demand[i] = 1
print(demand)

#MODELING

def main():
  # Create the solver.
  solver = pywrapcp.Solver('P-median problem')

  # Data declaration 
  p = 100

  num_customers = 1000
  customers = list(range(num_customers))

  num_stations = 100
  stations = list(range(num_stations))
    
  demand = list(range(num_customers))
  for i in range(num_customers):
    demand[i] = 1

  distance = distance_matrix.astype(int)

  #Variable declaration 
  open = [solver.IntVar(stations, 'open[%i]% % i') for s in stations]
  path = {}
  for c in customers:
    for s in stations:
      path[c, s] = solver.IntVar(0, 1, 'path[%i,%i]' % (c, s))
  path_flat = [path[c, s] for c in customers for s in stations]

  z = solver.IntVar(0, 1000, 'z')


  #CONSTRAINTS

  #Path[i,j] = 1 if customer i is allocated to be served by warehouse j
  for c in customers:
    s = solver.Sum([path[c, s] for s in stations])
    solver.Add(s <= 1)
      
  #Fixes the number of plants to p (15)
  solver.Add(solver.Sum(open) <= p) 
    
  #No more than one warehouse must be assigned to each customer. Warehouse w is assigned to c only if w has been declared open in the solution
  for c in customers:
    for s in stations:
      solver.Add(path[c, s] <= open[s])
    
  #Maximum distance contraint 
  max_distance = 1000  #A set nous meme 
  for c in customers:
    for s in stations:
        solver.Add(path[c, s] * distance[c][s] <= max_distance)

  #Pourcentage of customers demands are met
  min_service_level = 0.7  # Set an appropriate value between 0 and 1
  solver.Add(solver.Sum([path[c, s] * demand[c] for c in customers for s in stations]) >= int(min_service_level * sum(demand)))


    
  #Calculate the total number of bicycles for each station
  total_bikes_per_station = [solver.Sum([path[c, s] * demand[c] for c in customers]) for s in stations]

  #Calculate the total number of bicycles in the system
  total_bikes = solver.Sum(total_bikes_per_station)

  #Define the necessary cost parameters
  bicycle_cost = 4300
  charge_cost_per_bike = 0.40425
  num_charges = 50  # Set an appropriate value 
  max_budget = 100000  # Set an appropriate value

  #Add the cost constraint to the model
  total_cost = (bicycle_cost * total_bikes) + (charge_cost_per_bike * total_bikes * num_charges)
  solver.Add(total_cost <= max_budget)
    
    
  #Set the objective function 

  # objective function : Minimization of the demand weighted distance between facilities and customers (demand of customer i X distance between i and j X points i allocated to facilities) 
  #z is what we want to minimize in our objective function, and it should be equal to the sum below

  z_sum = solver.Sum([
      demand[c] * distance[c][s] * path[c, s]
      for c in customers
      for s in stations
    ])
  solver.Add(z == z_sum)


  objective = solver.Minimize(z, 1)

    #Solution and search
  db = solver.Phase(open + path_flat, solver.INT_VAR_DEFAULT,
                      solver.ASSIGN_MIN_VALUE)
  solver.Solve(db)

  solver.NewSearch(db, [objective])

  num_solutions = 0
  while solver.NextSolution():
    num_solutions += 1
    print('z:', z.Value())
    print('open:', [open[s].Value() for s in stations])
    for c in customers:
      for s in stations:
        print(path[c, s].Value(), end=' ')
      print()
    print()

  print('num_solutions:', num_solutions)
  print('failures:', solver.Failures())
  print('branches:', solver.Branches())

if __name__ == '__main__':
  main()

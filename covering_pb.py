from __future__ import print_function
import ortools
import pandas as pd
import numpy as np
import openpyxl
import warnings
import random
import math,sys
from ortools.sat.python import cp_model as cp
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, sin, cos, sqrt, atan2
from geopy.distance import geodesic
from geopy.distance import distance

#Stop the warnings 
warnings.simplefilter(action='ignore', category=UserWarning) #stop the warnings

def project_data():
    """Import the data"""
    
    demand = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="Netz Bern 01.01.22 - 31.12.22",
        names=['origin', 'dest', 'count'])

    locations = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="locations",
        names=['id', 'name', 'lat', 'lon'])
    
    return demand,locations

demand,locations = project_data()

sorted_df = demand.sort_values(by='count', ascending=False)
sorted_df

best_stations = sorted_df.iloc[:100].drop_duplicates('origin').reset_index()

df = best_stations.merge(locations, left_on='origin', right_on='id')

df.plot(x="lon", y="lat", kind="scatter",
        colormap="YlOrRd")

# Convert longitude and latitude columns to radians
df['lat_rad'] = np.radians(df['lat'])
df['lon_rad'] = np.radians(df['lon'])
df

# Define a function to generate points within a circle of radius r
def generate_points(lat, lon, r, n):
    points = []
    for i in range(1):
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
n = 1 # number of points to generate
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

distance_matrix = [[int((sqrt(entry))) for entry in row] for row in distance_matrix]
distance_matrix

demand = [random.choice([1, 2]) for _ in range(num_customers)]
demand

def main():
  # Create the solver.
  model = cp.CpModel()

  # Data declaration 
  p = 14

  customers = 37
  customer = list(range(customers))

  best_stations = 37
  stations = list(range(best_stations))
    
  distance = distance_matrix
  
  #maximum capacity
  max_capacity= 40 

  #maximum distance travelled by user 
  max_distance = 23 #(sqrt of 500)

  #Variable declaration
  open = [model.NewIntVar(0,1, 'open[%i]% % i') for s in stations]

  capacity = {}
  for s in stations:
    capacity[s] = model.NewIntVar(0,576000, 'capacity[%i]' % s)
  
  path = {}
  for c in customer:
    for s in stations:
      path[c, s] = model.NewIntVar(0, 1, 'x[%i,%i]' % (c, s))
 
  z = model.NewIntVar(0, 99999999999999, 'z')
    
  # Constraints
 
  model.Add(z == sum([
      demand[c] * distance[c][s] * path[c, s]
      for c in customer
      for s in stations
  ]))

  #Path[i,s] = 1 if customer i is allocated to be served by station s
  for c in customer:
    model.Add(sum([path[c, s] for s in stations]) == 1)
     
  #Fixes the number of open stations to p 
    model.Add(sum(open) == p)

  #No more than one station must be assigned to each customer. Warehouse w is assigned to c only if w has been declared open in the solution
  for c in customer:
    for s in stations:
      model.Add(path[c, s] <= open[s])
  
  #Maximum distance constraint 
  for c in customer:
    for s in stations:
        model.Add(path[c, s] * distance[c][s] <= max_distance)

  #Pourcentage of customers demands are met
  for s in stations:
    model.Add(sum(path[c, s] * demand[c] for c in customer) <= capacity[s])


  #Objective function optimization

  model.Minimize(z)

 # Model solution
  solver = cp.CpSolver()
  status = solver.Solve(model)

  opened = []
  if status == cp.OPTIMAL:
    print('z:', solver.Value(z))
    print('open:', [solver.Value(open[s]) for s in stations])
    opened = [solver.Value(open[s]) for s in stations]
    print('capacity:', [solver.Value(capacity[s]) for s in stations])
    for c in customer:
      for s in stations:
        print(solver.Value(path[c, s]), end=' ')
      print()
    print()
  print('WallTime:', solver.WallTime())

  print_df = pd.DataFrame()

  for station in range(len(opened)):
    if opened[station] == 1:
      print_df = print_df.append(sorted_df.iloc[[station]])

  print(print_df)
  print_df.to_csv('opti.csv', index=False)
  print('CSV created')
if __name__ == '__main__':
  main()


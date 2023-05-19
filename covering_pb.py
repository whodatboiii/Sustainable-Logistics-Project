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

#Loading the data 
def project_data():
    """Import the data"""
    
    demand = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="Netz Bern 01.01.22 - 31.12.22",
        names=['origin', 'dest', 'count'])
    
    print(demand.nunique())

    locations = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="locations",
        names=['id', 'name', 'lat', 'lon'])
    
    return demand,locations

demand,locations = project_data()

#Sorting the stations df in descending order
sorted_df = demand.sort_values(by='count', ascending=False)
sorted_df

#Selecting the 100 busiest stations ansd removing duplicated stations 
best_stations = sorted_df.iloc[:100].drop_duplicates('origin').reset_index()
print("bbbbb", best_stations)

#Merging the busiest stations df and df about their location 
df = best_stations.merge(locations, left_on='origin', right_on='id')

#Visualizing the best_station data points  
df.plot(x="lon", y="lat", kind="scatter",
        colormap="YlOrRd")

# Convert longitude and latitude columns to radians
df['lat_rad'] = np.radians(df['lat'])
df['lon_rad'] = np.radians(df['lon'])
#df

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

# Generate 1 customer point within 500 meters of each row in the DataFrame
n = 1 # number of points to generate
r = 0.5 # radius of circle in kilometers
points = []
for _, row in df.iterrows():
    lat, lon = row['lat_rad'], row['lon_rad']
    new_points = generate_points(lat, lon, r / 6371, n)
    for p in new_points:
        points.append((row['id'], row['lat'], row['lon'], p[0], p[1]))
        
# Convert points to a new df 
new_df = pd.DataFrame(points, columns=['id', 'orig_lat', 'orig_lon', 'new_lat', 'new_lon'])
new_df['distance'] = new_df.apply(lambda row: geodesic((row['orig_lat'], row['orig_lon']), (row['new_lat'], row['new_lon'])).km, axis=1)

# Keeping customer points within a radius of 0.5 from stations 
new_df = new_df[new_df['distance'] <= 0.5]

# Remove unnecessary columns
new_df = new_df[['id', 'new_lat', 'new_lon']]
customers = new_df.drop('id',axis=1)

#Visualizing the customers data points 
customers.plot(x="new_lon", y="new_lat", kind="scatter",
        colormap="YlOrRd")

#Creating the distance matrix (between customers and best stations)
num_customers = len(customers)
num_stations = len(df)

#Setting the distance matrix
distance_matrix = np.zeros((num_customers, num_stations))

for i, row in customers.iterrows():
    for j, station_row in df.iterrows():
        coords_1 = (row['new_lat'], row['new_lon'])
        coords_2 = (station_row['lat'], station_row['lon'])
        dist = distance(coords_1, coords_2).m  # distance in meters
        distance_matrix[i, j] = dist
print(distance_matrix)

#Decreasing the order of magnitude of our distances 
distance_matrix = [[int((sqrt(entry))) for entry in row] for row in distance_matrix]
distance_matrix

#Setting our customers demand 
demand = [random.choice([1, 2]) for _ in range(num_customers)]
demand

#Solving the covering problem 

def main():
  # Create the solver.
  model = cp.CpModel()

  # Data declaration 
  p = 14

  customers = 37
  customer = list(range(customers))

  best_stationsss = 37
  stations = list(range(best_stationsss))
    
  distance = distance_matrix
 

  #Maximum distance travelled by bike user 
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

  #Flows correspond to capacity of every station 
  for s in stations:
    model.Add(sum(path[c, s] * demand[c] for c in customer) <= capacity[s])


  #Objective function optimization

  model.Minimize(z)

 # Model solution
  solver = cp.CpSolver()
  status = solver.Solve(model)

  opened = []
  bikes = []
  if status == cp.OPTIMAL:
    print('z:', solver.Value(z))
    print('open:', [solver.Value(open[s]) for s in stations])
    opened = [solver.Value(open[s]) for s in stations]
    print('capacity:', [solver.Value(capacity[s]) for s in stations])
    bikes = [solver.Value(capacity[s]) for s in stations]
    for c in customer:
      for s in stations:
        print(solver.Value(path[c, s]), end=' ')
      print()
    print()
  print('WallTime:', solver.WallTime())

  print_df = pd.DataFrame()

  for station in range(len(opened)):
    if opened[station] == 1:
      print_df = print_df.append(best_stations.iloc[[station]])

  # Assuming you have a dataframe named 'print_df' with an existing 'count' column

  for index, bike in enumerate(bikes):
      if bike != 0:
          print_df.loc[index, 'count'] = bike

  print(print_df)
  print_df.to_csv('opti.csv', index=False)
  print('CSV created')

if __name__ == '__main__':
  main()


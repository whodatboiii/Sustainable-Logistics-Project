pip install ortools
import ortools
import pandas as pd
import numpy as np
import openpyxl
import warnings
import random
from __future__ import print_function
from ortools.sat.python import cp_model as cp
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, sin, cos, sqrt, atan2
from geopy.distance import geodesic
from geopy.distance import distance

#Stop the warnings 
warnings.simplefilter(action='ignore', category=UserWarning)

# Facility location problem 

def project_data():
    """Import the data"""
    
    demand  = pd.read_excel('/content/drive/MyDrive/Sustainable_logistics/Publike_bern.xlsx' , sheet_name = "demand",
        names = ['origin', 'dest', 'count']
    ) 
    
    
    locations = pd.read_excel('/content/drive/MyDrive/Sustainable_logistics/Publike_bern.xlsx'
        , sheet_name = "locations",
        names = ['id', 'name', 'lat', 'lon']
    ) 

 
    
    return demand,locations

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

#Capacity of our stations 

capacity = list(range(num_stations))
for i in range(num_stations):
capacity[i] = [random.randint(30, 40) for _ in range(num_stations)]
print(capacity)

#Modeling 

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


  #Constraints 

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

#Facility location ( with our objective of function - maximize the demand fulfillement) 

def main():
  # Create the solver.
  solver = pywrapcp.Solver('P-median problem')

  # Data declaration 
  p = 15

  num_customers = 36
  customers = list(range(num_customers))

  num_stations = 36
  stations = list(range(num_stations))
    
  demand = list(range(num_customers))
  for i in range(num_customers):
    demand[i] = 1

  distance = distance_matrix.astype(int)

  capacity = 

  #Variable declaration 
  open = [solver.IntVar(stations, 'open[%i]% % i') for s in stations]
  path = {}
  for c in customers:
    for s in stations:
      path[c, s] = solver.IntVar(0, 1, 'path[%i,%i]' % (c, s))
  path_flat = [path[c, s] for c in customers for s in stations]

  z = solver.IntVar(0, 1000000000, 'z')


  #Constraints 

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

  
  #Set the objective function 

  # objective function : Minimization of the demand weighted distance between facilities and customers (demand of customer i X distance between i and j X points i allocated to facilities) 
  #z is what we want to minimize in our objective function, and it should be equal to the sum below

  z_sum = solver.Sum([
    path[c, s]
      for c in customers
      for s in stations
    ])
  solver.Add(z == z_sum)


  objective = solver.Maximize(z, 1)

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



# Vehicle routing problem 


def generate_random_customers(center_lat, center_lon, min_radius, max_radius, num_customers):
    """Generate random customer positions within a circle with radius around a center point."""
    radii = np.random.uniform(min_radius, max_radius, num_customers)
    angles = np.random.uniform(0, 2 * np.pi, num_customers)
    
    customer_lats = center_lat + (radii * np.sin(angles)) / 111.32
    customer_lons = center_lon + (radii * np.cos(angles)) / (111.32 * np.cos(np.radians(center_lat)))

    customers = pd.DataFrame({'lat': customer_lats, 'lon': customer_lons})
    return customers

def generate_all_customers(locations, min_radius, max_radius, num_customers_per_station):
    all_customers = []

    for _, location in locations.iterrows():
        station_customers = generate_random_customers(location['lat'], location['lon'], min_radius, max_radius, num_customers_per_station)
        all_customers.append(station_customers)

    all_customers_df = pd.concat(all_customers, ignore_index=True)
    all_customers_df['id'] = np.arange(1, len(all_customers_df) + 1)
    
    return all_customers_df

def project_data(num_customers_per_station=100):
    """Import the data"""
    database = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="Netz Bern 01.01.22 - 31.12.22",
        names=['origin', 'dest', 'count'])

    locations = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="locations",
        names=['id', 'name', 'lat', 'lon'])
    
    # Remove rows where origin and dest are the same
    database = database.loc[database['origin'] != database['dest']]

    # Clean the locations DataFrame
    locations = locations[locations['id'].isin(database['origin'].unique()) | locations['id'].isin(database['dest'].unique())]

    # Select a random sample of 5 stations
    locations = locations.sample(n=15, random_state=42069)

    # Convert meters to kilometers
    min_radius = 100 / 1000
    max_radius = 150 / 1000

    # Generate customer positions
    customers = generate_all_customers(locations, min_radius, max_radius, num_customers_per_station)

    return database, locations, customers

def add_warehouse(locations, warehouse_loc):
    locations = pd.concat([locations, warehouse_loc], ignore_index=True)
    return locations

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on the Earth."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = [radians(float(coord)) for coord in [lat1, lon1, lat2, lon2]]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def weighted_distance_callback(manager, database, locations, from_index, to_index):
    """Returns the weighted distance between two locations considering charging cost and popularity."""
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    lat1, lon1 = locations.at[from_node, 'lat'], locations.at[from_node, 'lon']
    lat2, lon2 = locations.at[to_node, 'lat'], locations.at[to_node, 'lon']
    distance = haversine(lat1, lon1, lat2, lon2)
    
    if to_node != from_node:
        if to_node in database.index:
            popularity_weight = 1 + database.at[to_node, 'count'] / database['count'].max()
            charge_cost = 5 #arbitrary
            charging_time = 1 * database.at[to_node, 'count']
            weighted_distance = distance * popularity_weight + charge_cost + charging_time
        else:
            weighted_distance = distance
    else:
        weighted_distance = distance

    return int(weighted_distance)


def print_solution(manager, routing, solution, locations, database):
    """Prints the solution."""
    print(f"Objective: {solution.ObjectiveValue()} units")
    capacity_dimension = routing.GetDimensionOrDie("Capacity")
    
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            next_node_index = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            route_distance += weighted_distance_callback(manager, database, locations, index, solution.Value(routing.NextVar(index)))
            current_location = locations.iloc[node_index]
            next_location = locations.iloc[next_node_index]
            
            load = solution.Value(capacity_dimension.CumulVar(index))
            route_load += load
            plan_output += f"(Cur. Load: {load}) {current_location['name']} ({current_location['id']}) -> "
            index = solution.Value(routing.NextVar(index))
        
        last_location = locations.iloc[manager.IndexToNode(index)]
        last_load = solution.Value(capacity_dimension.CumulVar(index))
        plan_output += f"{last_location['name']} ({last_location['id']}) Load: {last_load}\n"
        plan_output += f"Route distance: {route_distance} units\n"
        plan_output += f"Total load of the route: {last_load}\n"
        
        print(plan_output)


def demand_callback(database, manager, node):
    """Returns the demand of the node."""
    if node == manager.GetNumberOfNodes() - 1:  # Warehouse
        return 0
    return database.iloc[node]['count']


def modelling():
    database, locations, customers = project_data()

    warehouse_loc = pd.DataFrame({'id': ['9999'], 'name': ['Warehouse'], 'lat': ['46.9489'], 'lon': ['7.4378']})
    locations = add_warehouse(locations, warehouse_loc)

    # Create the routing model
    num_nodes = len(locations)
    depot_index = num_nodes - 1
    num_vehicles = 3
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_index)
    model = pywrapcp.RoutingModel(manager)

    # Define the weighted distance callback
    transit_callback_index = model.RegisterTransitCallback(lambda from_index, to_index: weighted_distance_callback(manager, database, locations, from_index, to_index))

    # Set the cost of travel
    model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Define the demand callback
    demand_callback_index = model.RegisterUnaryTransitCallback(lambda index: demand_callback(database, manager, manager.IndexToNode(index)))
    #for i in range(manager.GetNumberOfNodes()):
    #    print(f"Node {i}: demand {demand_callback(database, manager, i)}")

    # Set the demand function
    model.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [200] * num_vehicles,  # vehicle capacity of 200 bike batteries
        True,  # start cumul to zero
        "Capacity"
    )

    # Solve the problem
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    search_parameters.time_limit.seconds = 120  # Set time limit to 120 seconds
    search_parameters.log_search = False

    solution = model.SolveWithParameters(search_parameters)

    if solution:
        # Print the solution
        print_solution(manager=manager, routing=model, solution=solution, locations=locations, database=database)
    else:
        print('No solution found.')
        

if __name__ == '__main__':
    modelling()



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

#Stop the warnings 
warnings.simplefilter(action='ignore', category=UserWarning)

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
    database = pd.read_csv("opti.csv")

    locations = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="locations",
        names=['id', 'name', 'lat', 'lon'])
    
    # Remove rows where origin and dest are the same
    database = database.loc[database['origin'] != database['dest']]

    # Clean the locations DataFrame
    locations = locations[locations['id'].isin(database['origin'].unique())] # | locations['id'].isin(database['dest'].unique())]
    print(locations)

    # Select a random sample of 15 stations
    #locations = locations.sample(n=10, random_state=5)

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

def weighted_distance_callback(manager, database, locations, from_index, to_index, num_customers_per_station):
    """Returns the weighted distance between two locations considering charging cost and popularity."""
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    lat1, lon1 = locations.at[from_node, 'lat'], locations.at[from_node, 'lon']
    lat2, lon2 = locations.at[to_node, 'lat'], locations.at[to_node, 'lon']
    distance = haversine(lat1, lon1, lat2, lon2)
    
    #Constraints
    if to_node != from_node:
        if to_node in database.index:
            popularity_weight = 1 + database.at[to_node, 'count'] / database['count'].max()
            charge_cost = 5 #arbitrary
            charging_time = 1 * database.at[to_node, 'count']
            weighted_distance = distance * popularity_weight + charge_cost + charging_time
            # Generate customer positions
            customers = generate_all_customers(locations, min_radius=0.0001, max_radius=0.1, num_customers_per_station=100)
            weighted_distance += len(customers) 
        else:
            weighted_distance = distance
    else:
        weighted_distance = distance

    return int(weighted_distance)


def print_solution_with_customers(manager, routing, solution, locations, database, num_customers_per_station):
    """Prints the solution."""
    print(f"Objective: {solution.ObjectiveValue()} units")
    capacity_dimension = routing.GetDimensionOrDie("Capacity")
    
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            next_node_index = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            route_distance += weighted_distance_callback(manager, database, locations, index, solution.Value(routing.NextVar(index)), num_customers_per_station)
            current_location = locations.iloc[node_index]
            next_location = locations.iloc[next_node_index]
            
            load = solution.Value(capacity_dimension.CumulVar(index))
            plan_output += f"{current_location['name']} ({current_location['id']}) Load: {load} -> "
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

    warehouse_loc = pd.DataFrame({'id': ['9999'], 'name': ['Warehouse'], 'lat': [46.9489], 'lon': [7.4378]})
    locations = add_warehouse(locations, warehouse_loc)

    # Create the routing model
    num_nodes = len(locations)
    depot_index = num_nodes - 1
    num_vehicles = 5
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_index)
    model = pywrapcp.RoutingModel(manager)

    # Define the weighted distance callback
    transit_callback_index = model.RegisterTransitCallback(lambda from_index, to_index: weighted_distance_callback(manager, database, locations, from_index, to_index))

    # Set the cost of travel
    model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Define the demand callback
    demand_callback_index = model.RegisterUnaryTransitCallback(lambda index: demand_callback(database, manager, manager.IndexToNode(index)))

    # Set the demand function
    model.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [16] * num_vehicles,  #arbitrary capacity
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
        print_solution_with_customers(manager=manager, routing=model, solution=solution, locations=locations, database=database, num_customers_per_station=100)
    else:
        print('No solution found.')


if __name__ == '__main__':
    modelling()

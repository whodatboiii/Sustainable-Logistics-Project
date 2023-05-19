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

    return database, locations

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
    
    #Constraints
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


from collections import defaultdict
import folium

def print_solution(manager, routing, solution, locations, database):
    """Prints the solution."""
    print(f"Objective: {solution.ObjectiveValue()} units")
    capacity_dimension = routing.GetDimensionOrDie("Capacity")

    # dictionary to hold the route of each vehicle
    vehicle_routes = defaultdict(list)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
              'darkpurple', 'pink', 'lightblue', 'lightgreen',
              'gray', 'black', 'lightgray']

    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            current_location = locations.iloc[node_index]
            vehicle_routes[vehicle_id].append((current_location['lat'], current_location['lon']))

            next_node_index = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            route_distance += weighted_distance_callback(manager, database, locations, index, solution.Value(routing.NextVar(index)))
            
            load = solution.Value(capacity_dimension.CumulVar(index))
            plan_output += f"{current_location['name']} ({current_location['id']}) Load: {load} -> "
            index = solution.Value(routing.NextVar(index))
        
        last_location = locations.iloc[manager.IndexToNode(index)]
        last_load = solution.Value(capacity_dimension.CumulVar(index))
        plan_output += f"{last_location['name']} ({last_location['id']}) Load: {last_load}\n"
        plan_output += f"Route distance: {route_distance} units\n"
        plan_output += f"Total load of the route: {last_load}\n"
        
        print(plan_output)

    # plot the routes
    base_map = folium.Map(location=[locations['lat'].mean(), locations['lon'].mean()], zoom_start=13)
    for vehicle_id, route in vehicle_routes.items():
        folium.PolyLine(route, color=colors[vehicle_id % len(colors)], weight=2.5, opacity=1).add_to(base_map)
        marker = route[-1]
        folium.Marker(marker, popup=f'Vehicle {vehicle_id + 1}', icon=folium.Icon(color=colors[vehicle_id % len(colors)])).add_to(base_map)
    base_map.save('routes.html')



def demand_callback(database, manager, node):
    """Returns the demand of the node."""
    if node == manager.GetNumberOfNodes() - 1:  # Warehouse
        return 0
    return database.iloc[node]['count']


def modelling():
    database, locations = project_data()

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
        print_solution(manager=manager, routing=model, solution=solution, locations=locations, database=database)
    else:
        print('No solution found.')


if __name__ == '__main__':
    modelling()

import ortools
import pandas as pd
import openpyxl
import warnings
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, sin, cos, sqrt, atan2

warnings.simplefilter(action='ignore', category=UserWarning)

def project_data():
    """Import the data"""
    database = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="Netz Bern 01.01.22 - 31.12.22",
        names=['origin', 'dest', 'count'])

    locations = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name="locations",
        names=['id', 'name', 'lat', 'lon'])

    locations = locations.sample(n=5, random_state=1)
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
    """Returns the weighted distance between two locations considering charging cost, popularity, and charging time."""
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    lat1, lon1 = locations.at[from_node, 'lat'], locations.at[from_node, 'lon']
    lat2, lon2 = locations.at[to_node, 'lat'], locations.at[to_node, 'lon']
    distance = haversine(lat1, lon1, lat2, lon2)
    
    if to_node != from_node:
        popularity_weight = 1 + database.at[to_node, 'count'] / database['count'].max()
        charge_cost = 5
        charging_time = 0.2 * database.at[to_node, 'count']
        weighted_distance = distance * popularity_weight + charge_cost + charging_time
    else:
        weighted_distance = distance

    return int(weighted_distance)


def print_solution(manager, routing, solution, database, locations):
    """Prints the solution."""
    print(f"Objective: {solution.ObjectiveValue()} units")
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            next_node_index = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            route_distance += weighted_distance_callback(manager, database, locations, index, solution.Value(routing.NextVar(index)))
            current_location = locations.iloc[node_index]
            next_location = locations.iloc[next_node_index]
            plan_output += f"{current_location['name']} ({current_location['id']}) -> "
            index = solution.Value(routing.NextVar(index))
        last_location = locations.iloc[manager.IndexToNode(index)]
        plan_output += f"{last_location['name']} ({last_location['id']})\n"
        plan_output += f"Route distance: {route_distance} units\n"
        print(plan_output)


def demand_callback(database, manager, node):
    """Returns the demand of a node."""
    return database.at[node, 'count']

def modelling():
    database, locations = project_data()

    warehouse_loc = pd.DataFrame({'id': ['9999'], 'name': ['Warehouse'], 'lat': ['46.9489'], 'lon': ['7.4378']})
    locations = add_warehouse(locations, warehouse_loc)

    # Create the routing model
    num_nodes = len(locations)
    depot_index = num_nodes - 1
    num_vehicles = 15
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
        [200] * num_vehicles,  # vehicle capacity of 200 bike batteries
        True,  # start cumul to zero
        "Capacity"
    )

    # Solve the problem
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 120  # Set time limit to 60 seconds

    solution = model.SolveWithParameters(search_parameters)

    if solution:
        # Print the solution
        print_solution(manager=manager, routing=model, solution=solution, locations=locations, database=database)
    else:
        print('No solution found.')
        

if __name__ == '__main__':
    modelling()


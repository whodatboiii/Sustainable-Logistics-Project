import ortools
import pandas as pd
import openpyxl
import warnings
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

warnings.simplefilter(action='ignore', category=UserWarning) #stops the warnings

def project_data():
    """Import the data"""
    #Manually deleted columns we didn't need in Excel
    database = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name = "Netz Bern 01.01.22 - 31.12.22",
        names = ['origin', 'dest', 'count']
    ) 

    locations = pd.read_excel(
        "Flowmap CH 01.01.2022 - 31.10.2022.xlsx", sheet_name = "locations",
        names = ['id', 'name', 'lat', 'lon']
    )

   #print(database.head(),"\n", locations.head())
    #So pretty

    #Will have to change data below (charge_cost & the Warehouse coordinates)
    charge_cost = 5
    warehouse_loc = pd.DataFrame({'id': ['9999'], 'name': ['Warehouse'], 'lat': ['46.9489'], 'lon': ['7.4378']})
    locations = pd.concat([locations, warehouse_loc], ignore_index=True)

    return database, locations

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points on the Earth."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def print_solution(manager, routing, solution):
    """Prints the solution."""
    print(f"Objective: {solution.ObjectiveValue()} units")
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            next_node_index = manager.IndexToNode(solution.Value(routing.NextVar(index)))
            route_distance += weighted_distance_callback(index, manager.IndexToNode(solution.Value(routing.NextVar(index))))
            plan_output += f"{node_index} -> "
            index = solution.Value(routing.NextVar(index))
        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Route distance: {route_distance} units\n"
        print(plan_output)

def weighted_distance_callback(from_index, to_index):
    """Returns the weighted distance between two locations considering charging cost and popularity."""
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    lat1, lon1 = locations.at[from_node, 'lat'], locations.at[from_node, 'lon']
    lat2, lon2 = locations.at[to_node, 'lat'], locations.at[to_node, 'lon']
    distance = haversine(lat1, lon1, lat2, lon2)
    
    if to_node != depot_index:
        popularity_weight = 1 + database.at[to_node, 'count'] / database['count'].max()
        charge_cost = 5
        weighted_distance = distance * popularity_weight + charge_cost
    else:
        weighted_distance = distance

    return int(weighted_distance)

def modelling():
    database, locations = project_data()

    # Create the routing model
    num_nodes = len(locations)
    depot_index = num_nodes - 1
    num_vehicles = 1
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_index)
    model = pywrapcp.RoutingModel(manager)

    # Define the weighted distance callback
    transit_callback_index = model.RegisterTransitCallback(weighted_distance_callback)

    # Set the cost of travel
    model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Solve the problem
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = model.SolveWithParameters(search_parameters)

    if solution:
        # Print the solution
        print_solution(manager=manager, routing=model, solution=solution)
    else:
        print('No solution found.')

if __name__ == '__main__':
    modelling()

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


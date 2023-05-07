# Sustainable-Logistics-Project
Analysis of the Flowmap from Publibike in Bern. 

Concepts to be used:
1. TSP to better charge the batteries of all the bikes (minimize the cost, so the length, to go to each station)
2. Warehouse placement problem; where should the truck come from?

***Step 1 : Facility location problem***

**Objective**: Choose what stations to close while maximizing the demand fulfillement - at the end of this process we will compare how many customers did we loose closing these stations (as a cost) and how much did we gain closing these stations - find a balance while playing with the number of stations to close 

**Parameters : **
Decision variables : 
xi : binary variable which is equal to 1 if customer i is served 
yij : binary variable which is equal to 1 if customer i is served by station j 
yj : binary variable which is equal to 1 if station j is open 

**Objective function: **
Max sum(i from 1 to I) yij

**Constraints :** 
Total number of stations than can be open ( we have to play with this number)
Each demand point has to served at least by one station) 
Each station can serve a limited number of demand points based on the number of bikes available and the distance between station and demand point)

What we have to do:

1.Generate a customers data frame - customers position ( between 100-500m)   
1.1 Eliminate all the rows where origin=dest 
1.2 Select the 100 (ex) busiest stations 
1.3 Generate a certain number of customer points at a perimeter between 100-500m)

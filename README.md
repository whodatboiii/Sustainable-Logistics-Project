# Sustainable-Logistics-Project
Analysis of the Flowmap from Publibike in Bern. 

Concepts to be used:
1. TSP to better charge the batteries of all the bikes (minimize the cost, so the length, to go to each station)
2. Warehouse placement problem; where should the truck come from?

***Step 1 : Facility location problem***

** Objective : ** 

Choose what stations to close while maximizing the demand fulfillement - at the end of this process we will compare how many customers did we loose closing these stations (as a cost) and how much did we gain closing these stations - find a balance while playing with the number of stations to close 

** Parameters : **

Decision variables : 
xi : binary variable which is equal to 1 if customer i is served 
yij : binary variable which is equal to 1 if customer i is served by station j 
yj : binary variable which is equal to 1 if station j is open 

** Objective function : **

Max sum(i from 1 to I) yij

** Constraints : ** 

Total number of stations than can be open ( we have to play with this number)
Each demand point has to served at least by one station) 
Each station can serve a limited number of demand points based on the number of bikes available and the distance between station and demand point)

** What we have to do : **

1.Generate a customers data frame - customers position ( between 100-500m)   
1.1 Eliminate all the rows where origin=dest 
1.2 Select the 100 (ex) busiest stations 
1.3 Generate a certain number of customer points at a perimeter between 100-500m)

#### RETRIEVED COMMENT

Parameters to add :
**Coûts vélo **
Entre 900-3500$ ( purchasing cost,docking station(lock the bike ? ), maintenance equipment)
source: https://cff-prod.s3.amazonaws.com/storage/files/ZcBkyUB3K1YL4MzzkjS76l9G2PypVK23TnO70viG.pdf

or
Coûts (Vélo + Station) : $3000-$5000
https://www.itskrs.its.dot.gov/its/benecost.nsf/ID/f72abdbb00d6ebb58525856d0060feea
or
Coût( admin and op) : average of $4,200-5,400 per bicycle, including all system components, staff and administrative.

**Coût chargement **
Elec en Suisse : 0,2695.-/kWatth
source : https://www.admin.ch/gov/fr/accueil/documentation/communiques.msg-id-90237.html
Puissance vélo : 250W ( normes européennes)
source : https://cosmoconnected.com/blogs/news/puissance-velo-electrique#:~:text=Puissance%20de%20v%C3%A9lo%20%C3%A9lectrique%20%3A%20que,%C3%AAtre%20limit%C3%A9e%20%C3%A0%20250%20Watts.
Temps de charge : 6h
Coût total : 0.1010.- / chargement / vélo
Puissance de la batterie : 375 Wh
source: https://www.publibike.ch/fr/velos

Capacité vélo
Distance : 50km
source https://www.cyclable.com/quelle-est-lautonomie-de-mon-velo-electrique/

Capacité camion

Time required pour changer les vélo

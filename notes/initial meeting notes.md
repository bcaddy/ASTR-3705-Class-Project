# Initial meeting notes

## CGOLS
- Boxes 10x10x20 kpc
- static mesh, of 2048x2048x4096. Most other simulations use adaptive grids
- somewhat mimics M82, a nearby starburst galaxy. SFR = 5 solar masses a year
- Run at 3 resolutions. Each cell at 5, 10, and 20 pc
- Studying the clouds blown out by star formation
- SFR lowers partway through the sim
-

## Cloud Catalog
- Cloud = contiguous region meeting criteria (T < 2e4 K)
- N Cloud Catalog: numpy array of cloud properties P. Size NxP for eac simulation snapshot
- Properties: volume, mass, center of mass(CoM) position, CoM velocity
- Appears to be a resolution effect that is turning the cloud mass function down
  as a function of resolution
-

## Project Details
- Does the cloud mass function change with time? How much? Same distribution/population?
- Are the variations just Poisson?
- Is the relationship between the different parameters stable with time
- Monte Carlo to test if a model agrees with the mass function or mass weighted mass function
- Where is the point where the grid starts to effect the results?

## How to Split the Project
- MCMC maximum likelihood infrastructure
  - write down likelihood, then use MCMC code to find stuff
- Once we have the MCMC done then we can compare time steps by comparing MCMC best fits, i.e. where x% of the MCMC results sit
- MC sampling
- Fitting volume and mass distribution

## Plan
1. Ingest Catalog
2. Make plots like what Alwin has, plotting various properties against each
   other, mass weighting, number plots, etc
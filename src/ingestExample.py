#!/usr/bin/env python3
"""
================================================================================
 Example on how to ingest catalogs

 Dependencies:
     numpy v1.19.2
================================================================================
"""

import ingestion

# First ingest the data in simulation units and save a copy to <repo-root>/data/
simCatalog512  = ingestion.ingestCatalog(resolution=512,  save=True)  # Execution time ~90ms
simCatalog1024 = ingestion.ingestCatalog(resolution=1024, save=True)  # Execution time ~800ms
simCatalog2048 = ingestion.ingestCatalog(resolution=2048, save=True)  # Execution time ~23.5s

# Next we convert the catalogs to physical units
physCatalog512  = ingestion.simUnits2PhysicalUnits(simCatalog512,  save=True)  # Execution time ~14ms
physCatalog1024 = ingestion.simUnits2PhysicalUnits(simCatalog1024, save=True)  # Execution time ~80ms
physCatalog2048 = ingestion.simUnits2PhysicalUnits(simCatalog2048, save=True)  # Execution time ~700ms

# Now we have catalogs with the ID, volume, mass, position, velocity, resolution, and time for each cloud
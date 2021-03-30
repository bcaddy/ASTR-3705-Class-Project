#!/usr/bin/env python3
"""
================================================================================
 Various utility functions for this project

 Dependencies:
     numpy v1.19.2
================================================================================
"""

import numpy as np

# ==============================================================================
def cellVol2Pc(resolution):
    """Returns conversion factor from simulation volume units to cubic parsecs

    Args:
        resolution (int): The resolution of the simulation. Options are 512, 1024, 2048

    Returns:
        float: Conversion factor
    """
    # Check for correct resolution value
    if not ((resolution == 512) or
            (resolution == 1024) or
            (resolution == 2048)):
        raise Exception(f"Incorrect value for resolution in utils.cellVol2Pc")

    return (10000.0/float(resolution))**3.
# ==============================================================================

# ==============================================================================
def cellMass2Mstar(resolution):
    """Returns conversion factor from simulation mass units to solar masses

    Args:
        resolution (int): The resolution of the simulation. Options are 512, 1024, 2048

    Returns:
        float: Conversion factor
    """
    # Check for correct resolution value
    if not ((resolution == 512) or
            (resolution == 1024) or
            (resolution == 2048)):
        raise Exception(f"Incorrect value for resolution in utils.cellMass2Mstar")

    return (10.0/float(resolution))**3.
# ==============================================================================
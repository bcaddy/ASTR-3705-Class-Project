#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy and Alwin Mao  Created on March 30th, 2020

 Ingest and return cloud catalog data from Alwin's catalogs

 Dependencies:
     numpy v1.19.2
     os

 Changelog:
     Version 1.0 - First Version
================================================================================
"""

import os
import numpy as np
import pathlib
import utils

# ==============================================================================
def _getfilenames(res):
    """Find and return the absolute paths of all the cloud catalogs for a given
    resolution. Contains the values for the base directory and directory paths.
    It only "finds" the names of the files themselves

    Args:
        res ([int]): The resolution of the simulation, options are 512, 1024,
        and 2048

    Returns:
        [list of strings]: All the filenames for the catalogs with resolution
        equal to res
    """
    basedict = '/bgfs/eschneider/alwin/'
    dirdict = {}
    dirdict[512] = '512new/'
    dirdict[1024] = '1024new/'
    dirdict[2048] = '2048new/'

    odict = basedict + dirdict[res] + 'output/'
    return [odict+filename for filename in os.listdir(odict) if 'mpicloud.npy' in filename]
# ==============================================================================

# ==============================================================================
def _zSelect(array,res):
    zm = array[:,5]
    mass = array[:,2]
    sel = mass > 0.0
    z = np.zeros(len(zm))
    z[sel] = zm[sel] / mass[sel]
    mask = (np.abs(z - res) > res/10.) & sel
    return array[mask]
# ==============================================================================

# ==============================================================================
def _merge_tuples_unionfind(tuples):
    # use classic algorithms union find with path compression
    # https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    parent_dict = {}

    def subfind(x):
        # update roots while visiting parents

        # if parent of x is not x
        if parent_dict[x] != x:
            # update parent of x by finding parent of parent
            # until the progenitor is reached parent_dict[progenitor] = progenitor
            parent_dict[x] = subfind(parent_dict[x])
        return parent_dict[x]

    def find(x):
        if x not in parent_dict:
            # x forms new set and becomes a root
            parent_dict[x] = x
            return x
        if parent_dict[x] != x:
            # follow chain of parents of parents to find root
            parent_dict[x] = subfind(parent_dict[x])
        return parent_dict[x]

    # each tuple represents a connection between two items
    # so merge them by setting root to be the lower root.
    for p0,p1 in list(tuples):
        r0 = find(p0)
        r1 = find(p1)
        if r0 < r1:
            parent_dict[r1] = r0
        elif r1 < r0:
            parent_dict[r0] = r1

    # for unique parents, subfind the root, replace occurrences with root
    vs = set(parent_dict.values())
    for parent in vs:
        sp = subfind(parent)
        if sp != parent:
            for key in parent_dict:
                if parent_dict[key] == parent:
                    parent_dict[key] = sp

    return parent_dict
# ==============================================================================

# ==============================================================================
def _combine_array2(parent_dict,ids,array):
    # 3x faster and correct

    # new id for each line of array data
    new_ids = [parent_dict[x] if x in parent_dict else x for x in ids]
    # sorted list
    uni = np.unique(new_ids)
    # new indices for bincount
    suni = np.searchsorted(uni,new_ids)
    # assert np.all([np.all(suni[new_ids == uni[i]] == i) for i in range(len(uni))])
    # basically, suni = i whenever new_ids = uni[i] so suni is reduced indices
    new_array = np.zeros([len(uni),len(array[0])])
    for i in range(len(array[0])):
        weight = array[:,i].astype(float)
        new_array[:,i] = np.bincount(suni,weights=weight)
    return new_array,uni
# ==============================================================================

# ==============================================================================
def _loadAndMerge(filename):
    # raw_array: N x M_properties
    # boundary: list of tuples
    # both saved to a numpy file filename
    raw_array,boundary = np.load(filename, allow_pickle=True)
    ids = raw_array[:,0]
    parent_dict = _merge_tuples_unionfind(boundary)
    array,new_ids = _combine_array2(parent_dict,ids,raw_array)
    array[:,0] = new_ids
    return array
# ==============================================================================

# ==============================================================================
def ingestCatalog(resolution, save=False, filename='', savePath=''):
    """Ingest and return the catalog of clouds for a given resolution.

    Args:
        resolution (int): The resolution simulation to use. Options are 512,
                          1024, and 2048
        save (bool, optional): Set to True to save the catalog to a file.
                               Defaults to False.
        filename (str, optional): The name of the file to save. Defaults to
                                  'catalog{resolution}'.
        savePath (str, optional): The directory to save the file in, must end in
                                  '/'. Defaults to '../data/'.

    Raises:
        Exception: Checks to make sure that the number of columns in the catalog
                   is correct and fails if it's not 9

    Returns:
        numpy structured array: A catalog of all the clouds for a given resolution.
        The format of this array is as follows. Each row has all the data for a
        given cloud and the columns can be referenced either by index or name.
        The columns are
        | Column Name | Column Index | Description                                | Type       | Units                         |
        |-------------|--------------|--------------------------------------------|------------|-------------------------------|
        | 'ID'        | 0            | The cloud ID                               | np.int64   | unitless                      |
        | 'volume'    | 1            | Cloud volume                               | np.int64   | number of cells               |
        | 'mass'      | 2            | sum(rho)                                   | np.float64 | simulation density            |
        | 'positionX' | 3            | sum(rho*x)                                 | np.float64 | simulation density * position |
        | 'positionY' | 4            | sum(rho*y)                                 | np.float64 | simulation density * position |
        | 'positionZ' | 5            | sum(rho*z)                                 | np.float64 | simulation density * position |
        | 'momentumX' | 6            | sum(rho*v_x)                               | np.float64 | simulation momentum           |
        | 'momentumY' | 7            | sum(rho*v_y)                               | np.float64 | simulation momentum           |
        | 'momentumZ' | 8            | sum(rho*v_z)                               | np.float64 | simulation momentum           |
        | 'resolution'| 9            | The resolution of the simulation           | np.int64   | unitles                       |
        | 'time'      | 10           | The time/snapshot that the cloud came from | np.int64   | Myr (also snapshot number)    |
    """
    # Check for correct resolution value
    if not ((resolution == 512) or
            (resolution == 1024) or
            (resolution == 2048)):
        raise Exception(f"Incorrect value for resolution in ingestion.ingestCatalog")

    # Get the names of all the cloud files
    filenames = _getfilenames(resolution)

    # Determine the times that that snapshots were taken at. This is the
    # snapshot number and the time in the simulation in megayears
    times = [file.split('/')[-1][:2] for file in filenames]

    # Create the structured array
    structureFormat = ([('ID',         np.int64),
                        ('volume',     np.int64),
                        ('mass',       np.float64),
                        ('positionX',  np.float64),
                        ('positionY',  np.float64),
                        ('positionZ',  np.float64),
                        ('momentumX',  np.float64),
                        ('momentumY',  np.float64),
                        ('momentumZ',  np.float64),
                        ('resolution', np.int64),
                        ('time',       np.int64)])
    catalog = np.empty(0, dtype=structureFormat)

    # Read in the catalogs
    for i in range(len(filenames)):
        # load the catalog
        newCatalog = _zSelect(_loadAndMerge(filenames[i]),resolution)
        newLen     = newCatalog.shape[0]

        # check if it has 9 columns
        if (newCatalog.shape[1] != 9):
            raise Exception(f"Catalog ingestion failed. Incorrect number of "
                            f"columns in {filenames[i]}. Expected 9 got "
                            f"{newCatalog.shape[1]}""")

        # Expand the main catalog for the new catalog
        catalog = np.hstack((catalog, np.empty(newLen,dtype=structureFormat)))

        # Append the new catalog
        catalog['ID'][-newLen:]         = newCatalog[:,0]
        catalog['volume'][-newLen:]     = newCatalog[:,1]
        catalog['mass'][-newLen:]       = newCatalog[:,2]
        catalog['positionX'][-newLen:]  = newCatalog[:,3]
        catalog['positionY'][-newLen:]  = newCatalog[:,4]
        catalog['positionZ'][-newLen:]  = newCatalog[:,5]
        catalog['momentumX'][-newLen:]  = newCatalog[:,6]
        catalog['momentumY'][-newLen:]  = newCatalog[:,7]
        catalog['momentumZ'][-newLen:]  = newCatalog[:,8]
        catalog['resolution'][-newLen:] = np.full(newLen, resolution, np.int64)
        catalog['time'][-newLen:]       = np.full(newLen, times[i], np.int64)

    # Choose to save or not
    if save:
        if not savePath:
            savePath = str((pathlib.Path(__file__).parent).absolute())[:-4] + '/data/'

        if filename:
            path = savePath + filename + '.npy'
        else:
            path = savePath + f'catalog{resolution}.npy'

        np.save(path, catalog, allow_pickle=False)

    # Return the ingested Catalog
    return catalog
# ==============================================================================

# ==============================================================================
def simUnits2PhysicalUnits(simCatalog, save=False, filename='', savePath=''):
    """Convert the ingested catalog to physical units

    Args:
        simCatalog (struct. array): The catalog in simulation units
        save (bool, optional): Set to True to save the catalog to a file.
                               Defaults to False.
        filename (str, optional): The name of the file to save. Defaults to
                                  'physCatalog{resolution}'.
        savePath (str, optional): The directory to save the file in, must end in
                                  '/'. Defaults to '../data/'.

    Returns:
        numpy structured array: A catalog of all the clouds for a given resolution.
        The format of this array is as follows. Each row has all the data for a
        given cloud and the columns can be referenced either by index or name.
        The columns are
        | Column Name | Column Index | Description                                | Type       | Units                         |
        |-------------|--------------|--------------------------------------------|------------|-------------------------------|
        | 'ID'        | 0            | The cloud ID                               | np.int64   | unitless                      |
        | 'volume'    | 1            | Cloud volume                               | np.float64 | cubic parsecs                 |
        | 'mass'      | 2            | Cloud mass                                 | np.float64 | Solar masses                  |
        | 'positionX' | 3            | Center of mass x position                  | np.float64 | simulation density * position |
        | 'positionY' | 4            | Center of mass y position                  | np.float64 | simulation density * position |
        | 'positionZ' | 5            | Center of mass z position                  | np.float64 | simulation density * position |
        | 'velocityX' | 6            | Center of mass x velocity                  | np.float64 | simulation momentum           |
        | 'velocityY' | 7            | Center of mass y velocity                  | np.float64 | simulation momentum           |
        | 'velocityZ' | 8            | Center of mass z velocity                  | np.float64 | simulation momentum           |
        | 'resolution'| 9            | The resolution of the simulation           | np.int64   | unitles                       |
        | 'time'      | 10           | The time/snapshot that the cloud came from | np.int64   | Myr (also snapshot number)    |
    """

    # Declare the catalog with physical units
    structureFormat = ([('ID',         np.int64),
                        ('volume',     np.float64),
                        ('mass',       np.float64),
                        ('positionX',  np.float64),
                        ('positionY',  np.float64),
                        ('positionZ',  np.float64),
                        ('velocityX',  np.float64),
                        ('velocityY',  np.float64),
                        ('velocityZ',  np.float64),
                        ('resolution', np.int64),
                        ('time',       np.int64)])
    physCatalog = np.empty(simCatalog.shape[0], dtype=structureFormat)

    # Get the resolution
    resolution = simCatalog['resolution'][0]

    # Compute the conversion factors
    volumeConversion = utils.cellVol2Pc(resolution)
    massConversion   = utils.cellMass2Mstar(resolution)

    physCatalog['ID']         = simCatalog['ID']
    physCatalog['volume']     = simCatalog['volume'].astype(np.float64) * volumeConversion
    physCatalog['mass']       = simCatalog['mass'] * massConversion
    physCatalog['positionX']  = (simCatalog['positionX'] / simCatalog['mass'] - resolution / 2.) * 1.E4 / float(resolution)
    physCatalog['positionY']  = (simCatalog['positionY'] / simCatalog['mass'] - resolution / 2.) * 1.E4 / float(resolution)
    physCatalog['positionZ']  = (simCatalog['positionZ'] / simCatalog['mass'] - resolution)      * 1.E4 / float(resolution)
    physCatalog['velocityX']  = (simCatalog['momentumX'] / simCatalog['mass']) * 977813.
    physCatalog['velocityY']  = (simCatalog['momentumY'] / simCatalog['mass']) * 977813.
    physCatalog['velocityZ']  = (simCatalog['momentumZ'] / simCatalog['mass']) * 977813.
    physCatalog['resolution'] = simCatalog['resolution']
    physCatalog['time']       = simCatalog['time']

    # Choose to save or not
    if save:
        if not savePath:
            savePath = str((pathlib.Path(__file__).parent).absolute())[:-4] + '/data/'

        if filename:
            path = savePath + filename + '.npy'
        else:
            path = savePath + f'physCatalog{resolution}.npy'

        np.save(path, physCatalog, allow_pickle=False)

    # Return
    return physCatalog
# ==============================================================================
#!/usr/bin/env python3
"""
================================================================================
 Written by Robert Caddy.  Created on 2021-04-16

 Perform t-SNE dimensionality reduction to check for clusters in cloud datasets

 Dependencies:
     numpy
     pandas
     sklearn.manifold
     seaborn
     matplotlib
     timeit

 Changelog:
     Version 1.0 - First Version
================================================================================
"""

import numpy as np
import pandas as pd

import sklearn.manifold as skmanifold
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from timeit import default_timer
from joblib import Parallel, delayed

# ==============================================================================
# Settings
maxSamples = 20000  # Choose the number of clouds to select from each catalog
n_jobs = 8

featureColumns = ['volume','mass','rPosition','vMag', 'polarAngle']

# coloring = 'volume'
# coloring = 'mass'
coloring = 'logMass'
# coloring = 'rPosition'
# coloring = 'vMag'
# coloring = 'time'

figureFormat = '.pdf'
# pathRoot = '/ihome/eschneider/rvc9/ASTR-3705-Class-Project/figures/tSNE/'
pathRoot = '/Users/Bob/Desktop/ASTR-3705-Class-Project/figures/tSNE/'
# ==============================================================================

# ==============================================================================
def converter(catalog):
    # Copy data to the dataFrame
    outputDF = pd.DataFrame()

    outputDF['ID']         = catalog['ID']
    outputDF['volume']     = catalog['volume']
    outputDF['mass']       = catalog['mass']
    outputDF['logMass']    = np.log(catalog['mass'])
    outputDF['rPosition']  = np.sqrt(catalog['positionX']**2
                                 + catalog['positionY']**2
                                 + catalog['positionZ']**2)
    outputDF['zPosition']  = catalog['positionZ'].reshape(-1, 1)
    outputDF['vMag']       = np.sqrt(catalog['velocityX']**2
                                + catalog['velocityY']**2
                                + catalog['velocityZ']**2)
    outputDF['polarAngle'] = np.arccos(np.abs(outputDF['zPosition'])/outputDF['rPosition'])
    outputDF['resolution'] = catalog['resolution']
    outputDF['time']       = catalog['time']

    return outputDF
# ==============================================================================

# ==============================================================================
def learnAndPlot(catalog, perp):
    print(f"started res={catalog['resolution'].iloc[1]}, perp={perp}")
    # Make data only dataframe
    catalogData  = catalog[featureColumns].values

    # Rescale the data
    scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)
    catalogData = scaler.fit_transform(catalogData)

    # Setup the model
    tsneModel   = skmanifold.TSNE(n_components=2,
                                  perplexity=perp,
                                  init='pca')

    # Run the model
    tsneResults = tsneModel.fit_transform(catalogData)

    # Add the new columns to catalogSubset
    catalog['tsne-0'] = tsneResults[:,0]
    catalog['tsne-1'] = tsneResults[:,1]

    # Plotting
    plt.figure(figsize=(16,10))
    sns.scatterplot(x='tsne-0',
                    y='tsne-1',
                    hue=coloring,
                    data=catalog,
                    alpha=1.)

    resolution = catalog['resolution'].iloc[1]
    plt.title(f't-SNE for resolution={resolution} and perplexity={perp}')

    filename = f'r{resolution}-p{perp}'
    plt.savefig(pathRoot + filename + figureFormat)
    plt.close()


# ==============================================================================

# ==============================================================================
def main():
    start = default_timer()

    # Load the datasets and print headers
    catalog512 = np.load('../data/physCatalog512.npy')
    catalog1024 = np.load('../data/physCatalog1024.npy')
    catalog2048 = np.load('../data/physCatalog2048.npy')

    # Convert to dataframes with mass, volume, radial distance, and magnitude of velocity
    processed512  = converter(catalog512)
    processed1024 = converter(catalog1024)
    processed2048 = converter(catalog2048)

    # Choose the number of samples and make subsets from dataframes
    minSize = np.min((processed512.shape[0], processed1024.shape[0], processed2048.shape[0]))
    if minSize < maxSamples:
        numSamples = minSize
    else:
        numSamples = maxSamples
    print(f'Number of samples is {numSamples}')

    subset512  = processed512.sample(n=numSamples, replace=False)
    subset1024 = processed1024.sample(n=numSamples, replace=False)
    subset2048 = processed2048.sample(n=numSamples, replace=False)

    # Choose perplexity values
    perps = np.arange(10, 110, 10)

    # Make a list of the sampled dataframes
    sampleData = [subset512, subset1024, subset2048]

    # Do the dimensionality reduction and generating figures
    Parallel(n_jobs=n_jobs)(delayed(learnAndPlot)
                                   (sampleData[j], perps[i])
                                   for i in range(len(perps))
                                   for j in range(len(sampleData)))

    print(f'\nTime to execute: {round(default_timer()-start,2)} seconds')
# ==============================================================================

main()

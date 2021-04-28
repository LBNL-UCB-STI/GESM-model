import pandas as pd
import os


miles2meters = 1609.34 # TODO: Change back to right number

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT_DIR, "..", "input-data", "MicrotypeAssignment.csv")
assignment = pd.read_csv(path)

path = os.path.join(ROOT_DIR, "..", "input-data", "Microtypes.csv")
microtypes = pd.read_csv(path)

path = os.path.join(ROOT_DIR, "..", "input-data", "DistanceBins.csv")
distanceBins = pd.read_csv(path)

binWeights = {"short": 0.5, "medium": 0.75, "long": 0.9}
collected = dict()
for originMicrotype in microtypes.MicrotypeID:
    for destinationMicrotype in microtypes.MicrotypeID:
        for distBin in distanceBins.DistanceBinID:
            for currentMicrotype in microtypes.MicrotypeID:
                out = {mID: binWeights[distBin]/len(microtypes.MicrotypeID) for mID in microtypes.MicrotypeID}
                collected[(originMicrotype, destinationMicrotype, distBin, currentMicrotype)] = out
collected = pd.DataFrame(collected)
collected.index.set_names(['To'], inplace=True)
collected.columns.set_names(['OriginMicrotypeID','DestinationMicrotypeID','DistanceBinID','From'], inplace=True)
collected = collected.transpose()


path = os.path.join(ROOT_DIR, "..", "input-data", "TransitionMatrices.csv")
collected.to_csv(path)
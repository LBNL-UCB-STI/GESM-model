import os
# from noisyopt import minimizeCompass
# from line_profiler_pycharm import profile
from collections import OrderedDict
from copy import deepcopy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from scipy.optimize import shgo

from utils.OD import TripCollection, OriginDestination, TripGeneration, TransitionMatrices, DemandIndex
from utils.choiceCharacteristics import CollectedChoiceCharacteristics
from utils.demand import Demand, CollectedTotalUserCosts, ODindex
from utils.microtype import MicrotypeCollection, CollectedTotalOperatorCosts
from utils.misc import TimePeriods, DistanceBins
from utils.network import CollectedNetworkStateData
from utils.population import Population


# from skopt import gp_minimize


class Optimizer:
    """
    Wrapper for the Model opject that allows model inputs to be optimized over.
    
    Attributes
    ----------
    path : str
        File path to input data
    fromToSubNetworkIDs : dict | None
        1:1 mapping of subnetworks between which ROW can be reassigned, e.g. mixed traffic -> bus only
    modesAndMicrotypes : dict | None
        List of tuples of mode/microtype pairs for which we will optimize headways
        e.g. [('A', 'bus'), ('B','rail')]
    method : str
        Optimization method
        
    Methods
    ---------
    evaluate(reallocations):
        Evaluate the objective funciton given a set of modifications to the transportation system
    minimize():
        Minimize the objective function using the set method
    """

    def __init__(self, path: str, fromToSubNetworkIDs=None, modesAndMicrotypes=None, method="shgo"):
        self.__path = path
        self.__fromToSubNetworkIDs = fromToSubNetworkIDs
        self.__modesAndMicrotypes = modesAndMicrotypes
        self.__method = method
        self.model = Model(path)
        print("Done")

    def nSubNetworks(self):
        if self.__fromToSubNetworkIDs is not None:
            return len(self.__fromToSubNetworkIDs)
        else:
            return 0

    def nModes(self):
        if self.__modesAndMicrotypes is not None:
            return len(self.__modesAndMicrotypes)
        else:
            return 0

    def toSubNetworkIDs(self):
        return [toID for fromID, toID in self.__fromToSubNetworkIDs]

    def fromSubNetworkIDs(self):
        return [fromID for fromID, toID in self.__fromToSubNetworkIDs]

    def getDedicationCost(self, reallocations: np.ndarray) -> float:
        if self.nSubNetworks() > 0:
            microtypes = self.model.scenarioData["subNetworkData"].loc[self.toSubNetworkIDs(), "MicrotypeID"]
            modes = self.model.scenarioData["modeToSubNetworkData"].loc[
                self.model.scenarioData["modeToSubNetworkData"]["SubnetworkID"].isin(
                    self.toSubNetworkIDs()), "ModeTypeID"]
            perMeterCosts = self.model.scenarioData["laneDedicationCost"].loc[
                pd.MultiIndex.from_arrays([microtypes, modes]), "CostPerMeter"].values
            cost = np.sum(reallocations[:self.nSubNetworks()] * perMeterCosts)
            if np.isnan(cost):
                return np.inf
            else:
                return cost
        else:
            return 0.0

    def evaluate(self, reallocations: np.ndarray) -> float:
        # self.model.resetNetworks()
        if self.__fromToSubNetworkIDs is not None:
            networkModification = NetworkModification(reallocations[:self.nSubNetworks()], self.__fromToSubNetworkIDs)
        else:
            networkModification = None
        if self.__modesAndMicrotypes is not None:
            transitModification = TransitScheduleModification(reallocations[-self.nSubNetworks():],
                                                              self.__modesAndMicrotypes)
        else:
            transitModification = None
        self.model.modifyNetworks(networkModification, transitModification)
        userCosts, operatorCosts, vectorUserCosts = self.model.collectAllCosts()
        dedicationCosts = self.getDedicationCost(reallocations)
        print(reallocations)
        print(userCosts.total, operatorCosts.total, dedicationCosts)
        return np.sum(vectorUserCosts) + operatorCosts.total + dedicationCosts

    def getBounds(self):
        if self.__fromToSubNetworkIDs is not None:
            upperBoundsROW = list(
                self.model.scenarioData["subNetworkData"].loc[self.fromSubNetworkIDs(), "Length"].values)
            lowerBoundsROW = [0.0] * len(self.fromSubNetworkIDs())
        else:
            upperBoundsROW = []
            lowerBoundsROW = []
        upperBoundsHeadway = [3600.] * self.nModes()
        lowerBoundsHeadway = [120.] * self.nModes()
        defaultHeadway = [300.] * self.nModes()
        bounds = list(zip(lowerBoundsROW + lowerBoundsHeadway, upperBoundsROW + upperBoundsHeadway))
        if self.__method == "shgo":
            return bounds
        elif self.__method == "sklearn":
            return list(zip(lowerBoundsROW + lowerBoundsHeadway, upperBoundsROW + upperBoundsHeadway, defaultHeadway))
        elif self.__method == "noisy":
            return bounds
        else:
            return Bounds(lowerBoundsROW + lowerBoundsHeadway, upperBoundsROW + upperBoundsHeadway)

    def x0(self) -> np.ndarray:
        network = [10.0] * self.nSubNetworks()
        headways = [300.0] * self.nModes()
        return np.array(network + headways)

    def minimize(self):
        if self.__method == "shgo":
            return shgo(self.evaluate, self.getBounds(), sampling_method="simplicial")
        # elif self.__method == "sklearn":
        #    b = self.getBounds()
        #    return gp_minimize(self.evaluate, self.getBounds(), n_calls=100)
        # elif self.__method == "noisy":
        #     return minimizeCompass(self.evaluate, self.x0(), bounds=self.getBounds(), paired=False, deltainit=500000.0,
        #                            errorcontrol=False)
        else:
            return minimize(self.evaluate, self.x0(), bounds=self.getBounds(), method=self.__method)
        # return dual_annealing(self.evaluate, self.getBounds(), no_local_search=False, initial_temp=150.)
        # return minimize(self.evaluate, self.x0(), method='trust-constr', bounds=self.getBounds(),
        #                 options={'verbose': 3, 'xtol': 10.0, 'gtol': 1e-4, 'maxiter': 15, 'initial_tr_radius': 10.})


class TransitScheduleModification:
    def __init__(self, headways: np.ndarray, modesAndMicrotypes: list):
        self.headways = headways
        self.modesAndMicrotypes = modesAndMicrotypes

    def __iter__(self):
        for i in range(len(self.headways)):
            yield (self.modesAndMicrotypes[i]), self.headways[i]


class NetworkModification:
    def __init__(self, reallocations: np.ndarray, fromToSubNetworkIDs: list):
        self.reallocations = reallocations
        self.fromToSubNetworkIDs = fromToSubNetworkIDs

    def __iter__(self):
        for i in range(len(self.reallocations)):
            yield (self.fromToSubNetworkIDs[i]), self.reallocations[i]


class ScenarioData:
    """
    Class to fetch and store data in a dictionary for specified scenario.

    ...

    Attributes
    ----------
    path : str
        File path to input data
    data : dict
        Dictionary containing input data from respective inputs

    Methods
    -------
    loadMoreData():
        Loads more modes of transportation.
    loadData():
        Read in data corresponding to various inputs.
    copy():
        Return a new ScenarioData copy containing data.
    """

    def __init__(self, path: str, data=None):
        """
        Constructs and loads all relevant data of the scenario into the instance.

        Parameters
        ----------
            path : str
                File path to input data
            data : dict
                Dictionary containing input data from respective inputs
        """
        self.__path = path
        self.__diToIdx = dict()  # TODO: Just define this once at the beginning of everything
        self.__odiToIdx = dict()
        self.__modeToIdx = dict()
        self.__dataToIdx = dict()
        self.__microtypeIdToIdx = dict()
        self.__paramToIdx = dict()
        if data is None:
            self.data = dict()
            self.loadData()
        else:
            self.data = data
            self.loadData()

    @property
    def paramToIdx(self):
        return self.__paramToIdx

    @property
    def diToIdx(self):
        return self.__diToIdx

    @property
    def odiToIdx(self):
        return self.__odiToIdx

    @property
    def modeToIdx(self):
        return self.__modeToIdx

    @property
    def dataToIdx(self):
        return self.__dataToIdx

    @property
    def microtypeIdToIdx(self):
        return self.__microtypeIdToIdx

    def __setitem__(self, key: str, value):
        self.data[key] = value

    def __getitem__(self, item: str):
        return self.data[item]

    def loadModeData(self):
        """
        Follows filepath to modes and microtpe ID listed in the data files section under __path/modes.

        Returns
        -------
        A dict() with mode type as key and dataframe as value.
        """
        collected = OrderedDict()
        (_, _, fileNames) = next(os.walk(os.path.join(self.__path, "modes")))
        for file in fileNames:
            collected[file.split(".")[0]] = pd.read_csv(os.path.join(self.__path, "modes", file),
                                                        dtype={"MicrotypeID": str}).set_index("MicrotypeID")
        return collected

    def loadData(self):
        """
        Fills the data dict() with values, the dict() contains data pertaining to various data labels and given csv
        data.
        """
        self["subNetworkData"] = pd.read_csv(os.path.join(self.__path, "SubNetworks.csv"),
                                             usecols=["SubnetworkID", "Length", "vMax", "densityMax", "avgLinkLength"],
                                             index_col="SubnetworkID", dtype={"MicrotypeID": str}).fillna(0.0)
        self["subNetworkDataFull"] = pd.read_csv(os.path.join(self.__path, "SubNetworks.csv"),
                                                 index_col="SubnetworkID", dtype={"MicrotypeID": str})
        self["modeToSubNetworkData"] = pd.read_csv(os.path.join(self.__path, "ModeToSubNetwork.csv"))
        self["microtypeAssignment"] = pd.read_csv(os.path.join(self.__path, "MicrotypeAssignment.csv"),
                                                  dtype={"FromMicrotypeID": str, "ToMicrotypeID": str,
                                                         "ThroughMicrotypeID": str}).fillna("None")
        self["populations"] = pd.read_csv(os.path.join(self.__path, "Population.csv"), dtype={"MicrotypeID": str})
        self["populationGroups"] = pd.read_csv(os.path.join(self.__path, "PopulationGroups.csv"))
        self["timePeriods"] = pd.read_csv(os.path.join(self.__path, "TimePeriods.csv"))
        self["distanceBins"] = pd.read_csv(os.path.join(self.__path, "DistanceBins.csv"))
        self["originDestinations"] = pd.read_csv(os.path.join(self.__path, "OriginDestination.csv"),
                                                 dtype={"HomeMicrotypeID": str, "OriginMicrotypeID": str,
                                                        "DestinationMicrotypeID": str})
        self["distanceDistribution"] = pd.read_csv(os.path.join(self.__path, "DistanceDistribution.csv"),
                                                   dtype={"OriginMicrotypeID": str, "DestinationMicrotypeID": str})
        self["tripGeneration"] = pd.read_csv(os.path.join(self.__path, "TripGeneration.csv"))
        self["transitionMatrices"] = pd.read_csv(os.path.join(self.__path, "TransitionMatrices.csv"),
                                                 dtype={"OriginMicrotypeID": str, "DestinationMicrotypeID": str,
                                                        "From": str}).set_index(
            ["OriginMicrotypeID", "DestinationMicrotypeID", "DistanceBinID", "From"])
        self["laneDedicationCost"] = pd.read_csv(os.path.join(self.__path, "LaneDedicationCost.csv"),
                                                 dtype={"MicrotypeID": str}).set_index(["MicrotypeID", "ModeTypeID"])
        self["modeData"] = self.loadModeData()
        self["microtypeIDs"] = pd.read_csv(os.path.join(self.__path, "Microtypes.csv"),
                                           dtype={"MicrotypeID": str}).set_index("MicrotypeID", drop=False)

        self.defineIndices()

    def defineIndices(self):
        self.__modeToIdx = {mode: idx for idx, mode in enumerate(self["modeData"].keys())}

        homeMicrotypeIDs = self.data['populations'].MicrotypeID.unique()
        groupAndPurpose = self.data['populationGroups'].groupby(
            ['PopulationGroupTypeID', 'TripPurposeID']).groups.keys()
        nestedDIs = list(product(homeMicrotypeIDs, groupAndPurpose))
        DIs = [(hID, popGroup, purpose) for hID, (popGroup, purpose) in nestedDIs]
        self.__diToIdx = {DemandIndex(*di): idx for idx, di in enumerate(DIs)}

        allODIs = list(product(self["microtypeIDs"].MicrotypeID, self["microtypeIDs"].MicrotypeID,
                               self["distanceBins"].DistanceBinID))
        self.__odiToIdx = {ODindex(*odi): idx for idx, odi in enumerate(allODIs)}

        self.__dataToIdx = {'tripStarts': 0, 'tripEnds': 1, 'throughTrips': 2, 'throughDistance': 3}

        self.__microtypeIdToIdx = {mID: idx for idx, mID in enumerate(self["microtypeIDs"].MicrotypeID)}

        self.__paramToIdx = {'intercept': 0, 'travel_time': 1, 'cost': 2, 'wait_time': 3, 'access_time': 4,
                             'protected_distance': 5, 'distance': 6}

    def copy(self):
        """
        Creates a deep copy of the data contained in this ScenarioData instance

        Returns
        -------
        A complete copy of the self.data dict()
        """
        return ScenarioData(self.__path, deepcopy(self.data))

    # def reallocate(self, fromSubNetwork, toSubNetwork, dist):

    def getModes(self):
        return set(self.__modeToIdx.keys())


class Model:
    """
    A class representing the GEMS Model.

    ...

    Attributes
    ----------
    path : str
        File path to input data
    scenarioData : ScenarioData
        Class object to fetch and store mode and parameter data
    initialScenarioData : ScenarioData
        Initial state of the scenario
    currentTimePeriod : str
        Description of the current time period (e.g. 'AM-Peak')
    microtypes : dict(str, MicrotypeCollection)
        Stores currentTimePeriod to MicrotypeCollection object
    demand : dict(str, Demand)
        Stores currentTimePeriod to Demand object
    choice : dict(str, CollectedChoiceCharacteristics)
        Stores currentTimePeriod to CollectedChoiceCharacteristics object
    population : Population
        Stores demandClasses, populationGroups, and totalCosts
    trips : TripCollection
        Stores a collection of trips through microtypes
    distanceBins : DistanceBins
        Set of distance bins for different distances of trips
    timePeriods  : TimePeriods
        Contains amount of time in each time period
    tripGeneration : TripGeneration
        Contains initialized trips with all the sets and classes
    originDestination : OriginDestination
        Stores origin/destination form of trips

    Methods
    -------
    microtypes:
        Returns the microtypes loaded into the model
    demand:
        Returns demand of the trips
    choice:
        Returns a choiceCharacteristic object
    readFiles():
        Initializes the model with file data
    initializeTimePeriod:
        Initializes the model with time periods
    findEquilibrium():
        Finds the ideal mode splits for the model
    getModeSplit(timePeriod=None, userClass=None, microtypeID=None, distanceBin=None):
        Returns the optimal mode splits
    getUserCosts(mode=None):
        Returns total user costs
    getModeUserCosts():
        Returns costs per mode found by the model
    getOperatorCosts():
        Returns total costs to the operator
    modifyNetworks(networkModification=None, scheduleModification=None):
        Used in the optimizer class to edit network after initialization
    resetNetworks():
        Reset network to original initialization
    setTimePeriod(timePeriod: str):

    getModeSpeeds(timePeriod=None):
        Returns speeds for each mode in each microtype
    """

    def __init__(self, path: str):
        self.__path = path
        self.scenarioData = ScenarioData(path)
        self.__initialScenarioData = ScenarioData(path)
        self.__currentTimePeriod = None
        self.__microtypes = dict()  # MicrotypeCollection(self.modeData.data)
        self.__demand = dict()  # Demand()
        self.__choice = dict()  # CollectedChoiceCharacteristics()
        self.__population = Population(self.scenarioData)
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()
        self.__timePeriods = TimePeriods()
        self.__tripGeneration = TripGeneration()
        self.__originDestination = OriginDestination()
        self.__transitionMatrices = TransitionMatrices(self.scenarioData)
        self.__networkStateData = dict()
        self.readFiles()
        self.initializeAllTimePeriods()

    @property
    def diToIdx(self):
        return self.scenarioData.diToIdx

    @property
    def odiToIdx(self):
        return self.scenarioData.odiToIdx

    @property
    def modeToIdx(self):
        return self.scenarioData.modeToIdx

    @property
    def dataToIdx(self):
        return self.scenarioData.dataToIdx

    @property
    def microtypeIdToIdx(self):
        return self.scenarioData.microtypeIdToIdx

    @property
    def currentTimePeriod(self):
        return self.__currentTimePeriod

    @property
    def microtypes(self):
        if self.__currentTimePeriod not in self.__microtypes:
            self.__microtypes[self.__currentTimePeriod] = MicrotypeCollection(self.scenarioData)
        return self.__microtypes[self.__currentTimePeriod]

    def getMicrotypeCollection(self, timePeriod) -> MicrotypeCollection:
        return self.__microtypes[timePeriod]

    @property
    def demand(self):
        if self.__currentTimePeriod not in self.__demand:
            self.__demand[self.__currentTimePeriod] = Demand(self.scenarioData)
        return self.__demand[self.__currentTimePeriod]

    @property
    def choice(self):
        if self.__currentTimePeriod not in self.__choice:
            self.__choice[self.__currentTimePeriod] = CollectedChoiceCharacteristics(self.scenarioData, self.demand)
        return self.__choice[self.__currentTimePeriod]

    @property
    def networkStateData(self):
        if self.__currentTimePeriod not in self.__networkStateData:
            self.__networkStateData[self.__currentTimePeriod] = CollectedNetworkStateData()
        return self.__networkStateData[self.__currentTimePeriod]

    def getNetworkStateData(self, timePeriod) -> CollectedNetworkStateData:
        return self.__networkStateData[timePeriod]

    def getCurrentTimePeriodDuration(self):
        return self.__timePeriods[self.currentTimePeriod]

    def readFiles(self):
        self.__trips.importTrips(self.scenarioData["microtypeAssignment"])
        self.__population.importPopulation(self.scenarioData["populations"], self.scenarioData["populationGroups"])
        self.__timePeriods.importTimePeriods(self.scenarioData["timePeriods"])
        self.__distanceBins.importDistanceBins(self.scenarioData["distanceBins"])
        self.__originDestination.importOriginDestination(self.scenarioData["originDestinations"],
                                                         self.scenarioData["distanceDistribution"])
        self.__tripGeneration.importTripGeneration(self.scenarioData["tripGeneration"])
        self.__transitionMatrices.importTransitionMatrices(self.scenarioData["transitionMatrices"],
                                                           self.scenarioData["microtypeIDs"],
                                                           self.scenarioData["distanceBins"])

    def initializeTimePeriod(self, timePeriod: str):
        self.__currentTimePeriod = timePeriod
        if timePeriod not in self.__microtypes:
            print("-------------------------------")
            print("|  Loading time period ", timePeriod, " ", self.__timePeriods.getTimePeriodName(timePeriod))
        self.microtypes.importMicrotypes()
        self.__originDestination.initializeTimePeriod(timePeriod, self.__timePeriods.getTimePeriodName(timePeriod))
        self.__tripGeneration.initializeTimePeriod(timePeriod, self.__timePeriods.getTimePeriodName(timePeriod))
        self.demand.initializeDemand(self.__population, self.__originDestination, self.__tripGeneration, self.__trips,
                                     self.microtypes, self.__distanceBins, self.__transitionMatrices,
                                     self.__timePeriods, self.__currentTimePeriod, 1.0)
        self.choice.initializeChoiceCharacteristics(self.__trips, self.microtypes, self.__distanceBins)

    def initializeAllTimePeriods(self):
        # self.__transitionMatrices.adoptMicrotypes(self.scenarioData["microtypeIDs"])
        for timePeriod, durationInHours in self.__timePeriods:
            self.initializeTimePeriod(timePeriod)
            print('Done Initializing')

    # @profile
    def findEquilibrium(self):
        diff = 1000.
        i = 0
        while (diff > 0.00001) & (i < 20):
            oldModeSplit = self.getModeSplit(self.__currentTimePeriod)
            # print(oldModeSplit)
            self.demand.updateMFD(self.microtypes)
            self.choice.updateChoiceCharacteristics(self.microtypes, self.__trips)
            diff = self.demand.updateModeSplit(self.choice, self.__originDestination, oldModeSplit)

            i += 1

    def getModeSplit(self, timePeriod=None, userClass=None, microtypeID=None, distanceBin=None):
        if timePeriod is None:
            timePeriods = self.scenarioData["timePeriods"].index
            weights = self.scenarioData["timePeriods"].DurationInHours.values
        else:
            timePeriods = [timePeriod]
            weights = [1]
        modes = self.scenarioData.getModes()
        # ms = ModeSplit(modeToIdx=self.modeToIdx, data=np.zeros(len(modes)))
        ms = np.zeros(len(self.modeToIdx))
        for tp, weight in zip(timePeriods, weights):
            if tp in self.__demand:
                ms += self.__demand[tp].getMatrixModeCounts() * weight
        return ms / np.sum(ms)

    def getUserCosts(self, mode=None):
        return self.demand.getUserCosts(self.choice, self.__originDestination, mode)

    def getMatrixUserCosts(self):
        return self.demand.getMatrixUserCosts(self.choice)

    def getModeUserCosts(self):
        out = dict()
        for mode in self.scenarioData['modeData'].keys():
            out[mode] = self.getUserCosts([mode]).toDataFrame()
        return pd.concat(out)

    def getOperatorCosts(self):
        return self.microtypes.getOperatorCosts()

    def modifyNetworks(self, networkModification=None,
                       scheduleModification=None):
        originalScenarioData = self.__initialScenarioData.copy()
        if networkModification is not None:
            for ((fromNetwork, toNetwork), laneDistance) in networkModification:
                oldFromLaneDistance = originalScenarioData["subNetworkData"].loc[fromNetwork, "Length"]
                self.scenarioData["subNetworkData"].loc[fromNetwork, "Length"] = oldFromLaneDistance - laneDistance
                oldToLaneDistance = originalScenarioData["subNetworkData"].loc[toNetwork, "Length"]
                self.scenarioData["subNetworkData"].loc[toNetwork, "Length"] = oldToLaneDistance + laneDistance

        if scheduleModification is not None:
            for ((microtypeID, modeName), newHeadway) in scheduleModification:
                self.scenarioData["modeData"][modeName].loc[microtypeID, "Headway"] = newHeadway

    def resetNetworks(self):
        self.scenarioData = self.__initialScenarioData.copy()

    def setTimePeriod(self, timePeriod: str, init=False):
        """Note: Are we always going to go through them in order? Should maybe just store time periods
        as a dataframe and go by index. But, we're not keeping track of all accumulations so in that sense
        we always need to go in order."""
        networkStateData = self.networkStateData
        self.__currentTimePeriod = timePeriod
        self.__originDestination.setTimePeriod(timePeriod)
        self.__tripGeneration.setTimePeriod(timePeriod)
        if networkStateData:
            if not init:
                self.microtypes.importPreviousStateData(networkStateData)
            else:
                self.microtypes.resetStateData()

    def collectAllCosts(self):
        userCosts = CollectedTotalUserCosts()
        operatorCosts = CollectedTotalOperatorCosts()
        vectorUserCosts = 0.0
        init = True
        for timePeriod, durationInHours in self.__timePeriods:
            self.setTimePeriod(timePeriod, init)
            self.microtypes.updateNetworkData()
            init = False
            self.findEquilibrium()
            matCosts = self.getMatrixUserCosts() * durationInHours
            vectorUserCosts += matCosts
            # userCosts += self.getUserCosts() * durationInHours
            operatorCosts += self.getOperatorCosts() * durationInHours
            self.__networkStateData[timePeriod] = self.microtypes.getStateData()
            print(self.getModeSplit(self.__currentTimePeriod))
            print(self.getModeSpeeds())
        return userCosts, operatorCosts, vectorUserCosts

    def getModeSpeeds(self, timePeriod=None):
        if timePeriod is None:
            timePeriod = self.__currentTimePeriod
        return pd.DataFrame(self.__microtypes[timePeriod].getModeSpeeds())

    def plotAllDynamicStats(self, type):
        ts = []
        vs = []
        ns = []
        runningTotal = 0.0
        for id, dur in self.__timePeriods:
            # out = self.getMicrotypeCollection(id).transitionMatrixMFD(dur, self.getNetworkStateData(id),
            #                                                           self.getMicrotypeCollection(
            #                                                               id).getModeStartRatePerSecond("auto"))
            sd = self.getNetworkStateData(id)
            t, v, n, label = sd.getAutoSpeeds()
            ts.append(t)
            vs.append(v)
            ns.append(n)
        if type.lower() == "n":
            x = np.concatenate(ts)
            y = np.concatenate(ns)
            # plt.plot(x, y)
            return x, y
        elif type.lower() == "v":
            x = np.concatenate(ts)
            y = np.concatenate(vs)
            # plt.plot(x, y)
            return x, y
        # elif type.lower() == "density":
        #     x = np.concatenate(ts)
        #     y = np.concatenate(reldensity)
        #     # plt.plot(x, y)
        #     return x, y
        else:
            print('WTF')
        print('AA')


if __name__ == "__main__":
    model = Model("input-data-geotype-A")
    userCosts, operatorCosts, vectorUserCosts = model.collectAllCosts()
    ms = model.getModeSplit()
    # a.plotAllDynamicStats("N")
    x, y = model.plotAllDynamicStats("v")
    plt.plot(x, y)
    print(ms)
    # o = Optimizer("input-data", list(zip([2, 4, 6, 8], [13, 14, 15, 16])))
    # o = Optimizer("input-data", fromToSubNetworkIDs=list(zip([2, 8], [13, 16])),
    #               modesAndMicrotypes=list(zip(["A", "D", "A", "D"], ["bus", "bus", "rail", "rail"])),
    #               method="shgo")
    # o = Optimizer("input-data-production",
    #               fromToSubNetworkIDs=list(zip([1, 7, 43, 49, 85, 91, 121, 127], [3, 9, 45, 51, 87, 93, 123, 129])),
    #               method="noisy")
    # o.evaluate(np.array([0., 30., 200., 200., 300., 300.]))
    # o.evaluate(np.array([0., 30., 200., 200., 300., 300.]))
    # o.evaluate(np.array([300., 200., 200., 200.]))
    # output = o.minimize()
    # print("DONE")
    # print(output.x)
    # print(output.fun)
    # print(output.message)
    #
    # print("DONE")
    # cost2 = o.evaluate(np.array([10., 1.]))
    # print(cost2)
    # cost3 = o.evaluate(np.array([500., 10.]))
    # print(cost3)
    # a = Model("input-data")
    # a.initializeTimePeriod("AM-Peak")
    # a.findEquilibrium()
    # ms = a.getModeSplit()
    # speeds = pd.DataFrame(a.microtypes.getModeSpeeds())
    # print(speeds)
    # userCosts = a.getUserCosts()
    # opCosts = a.getOperatorCosts()
    # mod = NetworkModification(np.array([200., 100.]), [2, 4], [13, 14])
    # a.modifyNetworks(mod)
    # userCosts2 = userCosts * 2
    # print("aah")

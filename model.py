import os

import numpy as np
import pandas as pd
from scipy.optimize import shgo
from scipy.optimize import minimize, Bounds
from skopt import gp_minimize
from noisyopt import minimizeCompass

from utils.OD import TripCollection, OriginDestination, TripGeneration
from utils.choiceCharacteristics import CollectedChoiceCharacteristics
from utils.demand import Demand, CollectedTotalUserCosts
from utils.microtype import MicrotypeCollection, CollectedTotalOperatorCosts
from utils.misc import TimePeriods, DistanceBins
from utils.population import Population


class Optimizer:
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
        self.model.resetNetworks()
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
        userCosts, operatorCosts = self.model.collectAllCosts()
        dedicationCosts = self.getDedicationCost(reallocations)
        print(reallocations)
        print(userCosts.total, operatorCosts.total, dedicationCosts)
        return userCosts.total + operatorCosts.total + dedicationCosts

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
        elif self.__method == "sklearn":
            b = self.getBounds()
            return gp_minimize(self.evaluate, self.getBounds(), n_calls=100)
        elif self.__method == "noisy":
            return minimizeCompass(self.evaluate, self.x0(), bounds=self.getBounds(), paired=False, deltainit=200.0,
                                   errorcontrol=False)
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
    def __init__(self, path: str, data=None):
        self.__path = path
        if data is None:
            self.data = dict()
            self.loadData()
        else:
            self.data = data

    def __setitem__(self, key: str, value):
        self.data[key] = value

    def __getitem__(self, item: str):
        return self.data[item]

    def loadModeData(self):
        collected = dict()
        (_, _, fileNames) = next(os.walk(os.path.join(self.__path, "modes")))
        for file in fileNames:
            collected[file.split(".")[0]] = pd.read_csv(os.path.join(self.__path, "modes", file),
                                                        index_col="MicrotypeID")
        return collected

    def loadData(self):
        self["subNetworkData"] = pd.read_csv(os.path.join(self.__path, "SubNetworks.csv"),
                                             index_col="SubnetworkID")
        self["modeToSubNetworkData"] = pd.read_csv(os.path.join(self.__path, "ModeToSubNetwork.csv"))
        self["microtypeAssignment"] = pd.read_csv(os.path.join(self.__path, "MicrotypeAssignment.csv"))
        self["populations"] = pd.read_csv(os.path.join(self.__path, "Population.csv"))
        self["populationGroups"] = pd.read_csv(os.path.join(self.__path, "PopulationGroups.csv"))
        self["timePeriods"] = pd.read_csv(os.path.join(self.__path, "TimePeriods.csv"))
        self["distanceBins"] = pd.read_csv(os.path.join(self.__path, "DistanceBins.csv"))
        self["originDestinations"] = pd.read_csv(os.path.join(self.__path, "OriginDestination.csv"))
        self["distanceDistribution"] = pd.read_csv(os.path.join(self.__path, "DistanceDistribution.csv"))
        self["tripGeneration"] = pd.read_csv(os.path.join(self.__path, "TripGeneration.csv"))
        self["laneDedicationCost"] = pd.read_csv(os.path.join(self.__path, "LaneDedicationCost.csv"),
                                                 index_col=["MicrotypeID", "ModeTypeID"])
        self["modeData"] = self.loadModeData()

    def copy(self):
        return ScenarioData(self.__path, self.data.copy())

    # def reallocate(self, fromSubNetwork, toSubNetwork, dist):


class Model:
    def __init__(self, path: str):
        self.__path = path
        self.scenarioData = ScenarioData(path)
        self.__initialScenarioData = ScenarioData(path)
        self.__currentTimePeriod = None
        self.__microtypes = dict()  # MicrotypeCollection(self.modeData.data)
        self.__demand = dict()  # Demand()
        self.__choice = dict()  # CollectedChoiceCharacteristics()
        self.__population = Population()
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()
        self.__timePeriods = TimePeriods()
        self.__tripGeneration = TripGeneration()
        self.__originDestination = OriginDestination()
        self.readFiles()

    @property
    def microtypes(self):
        if self.__currentTimePeriod not in self.__microtypes:
            self.__microtypes[self.__currentTimePeriod] = MicrotypeCollection(self.scenarioData["modeData"])
        return self.__microtypes[self.__currentTimePeriod]

    @property
    def demand(self):
        if self.__currentTimePeriod not in self.__demand:
            self.__demand[self.__currentTimePeriod] = Demand()
        return self.__demand[self.__currentTimePeriod]

    @property
    def choice(self):
        if self.__currentTimePeriod not in self.__choice:
            self.__choice[self.__currentTimePeriod] = CollectedChoiceCharacteristics()
        return self.__choice[self.__currentTimePeriod]

    def readFiles(self):
        self.__trips.importTrips(self.scenarioData["microtypeAssignment"])
        self.__population.importPopulation(self.scenarioData["populations"], self.scenarioData["populationGroups"])
        self.__timePeriods.importTimePeriods(self.scenarioData["timePeriods"])
        self.__distanceBins.importDistanceBins(self.scenarioData["distanceBins"])
        self.__originDestination.importOriginDestination(self.scenarioData["originDestinations"],
                                                         self.scenarioData["distanceDistribution"])
        self.__tripGeneration.importTripGeneration(self.scenarioData["tripGeneration"])

    def initializeTimePeriod(self, timePeriod: str):
        self.__currentTimePeriod = timePeriod
        if timePeriod not in self.__microtypes:
            print("-------------------------------")
            print("|  Loading time period ", timePeriod)
        self.microtypes.importMicrotypes(self.scenarioData["subNetworkData"], self.scenarioData["modeToSubNetworkData"])
        self.__originDestination.initializeTimePeriod(timePeriod)
        self.__tripGeneration.initializeTimePeriod(timePeriod)
        self.demand.initializeDemand(self.__population, self.__originDestination, self.__tripGeneration, self.__trips,
                                     self.microtypes, self.__distanceBins, 0.3)
        self.choice.initializeChoiceCharacteristics(self.__trips, self.microtypes, self.__distanceBins)

    def findEquilibrium(self):
        diff = 1000.
        i = 0
        while (diff > 0.00001) & (i < 20):
            ms = self.getModeSplit()
            self.demand.updateMFD(self.microtypes, 5)
            self.choice.updateChoiceCharacteristics(self.microtypes, self.__trips)
            diff = self.demand.updateModeSplit(self.choice, self.__originDestination, ms)
            # c = self.getOperatorCosts().total
            # if np.isnan(c):
            #     print("----")
            #
            # c2 = self.getUserCosts().total
            # if np.isnan(c2):
            #     print("----")
            # i += 1
            print(ms)
            # print(self.getModeSpeeds().loc['auto', ['A_1', 'A_2', 'A_4', 'B_1', 'B_2', 'B_4']])
        ms = self.getModeSplit()

    def getModeSplit(self, timePeriod=None):
        if timePeriod is None:
            timePeriod = self.__currentTimePeriod
        mode_split = self.__demand[timePeriod].getTotalModeSplit()
        return mode_split

    def getUserCosts(self):
        return self.demand.getUserCosts(self.choice, self.__originDestination)

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

    def collectAllCosts(self):
        userCosts = CollectedTotalUserCosts()
        operatorCosts = CollectedTotalOperatorCosts()
        for timePeriod, durationInHours in self.__timePeriods:
            self.initializeTimePeriod(timePeriod)
            self.findEquilibrium()
            userCosts += self.getUserCosts() * durationInHours
            operatorCosts += self.getOperatorCosts() * durationInHours
            print(self.getModeSplit())
        return userCosts, operatorCosts

    def getModeSpeeds(self, timePeriod=None):
        if timePeriod is None:
            timePeriod = self.__currentTimePeriod
        return pd.DataFrame(self.__microtypes[timePeriod].getModeSpeeds())


if __name__ == "__main__":
    a = Model("input-data-production")
    a.initializeTimePeriod("morning_rush")
    a.findEquilibrium()
    ms = a.getModeSplit()
    # a = Model("input-data")
    # a.initializeTimePeriod("AM-Peak")
    # a.findEquilibrium()
    # ms = a.getModeSplit()
    # print(a.getModeSpeeds())
    # print(ms)
    # o = Optimizer("input-data", list(zip([2, 4, 6, 8], [13, 14, 15, 16])))
    # o = Optimizer("input-data", fromToSubNetworkIDs=list(zip([2, 8], [13, 16])),
    #               modesAndMicrotypes=list(zip(["A", "D", "A", "D"], ["bus", "bus", "rail", "rail"])),
    #               method="shgo")
    # o = Optimizer("input-data",
    #               fromToSubNetworkIDs=list(zip([2, 8], [13, 16])),
    #               method="noisy")
    # # o.evaluate(np.array([0., 30., 200., 200., 300., 300.]))
    # # o.evaluate(np.array([0., 30., 200., 200., 300., 300.]))
    # # o.evaluate(np.array([300., 200., 200., 200.]))
    # output = o.minimize()
    print("DONE")
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

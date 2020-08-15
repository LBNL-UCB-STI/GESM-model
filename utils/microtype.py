#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .choiceCharacteristics import ChoiceCharacteristics
from .network import Network, NetworkCollection, NetworkFlowParams, Costs, TotalOperatorCosts


class CollectedTotalOperatorCosts:
    def __init__(self):
        self.__costs = dict()
        self.total = 0.

    def __setitem__(self, key: str, value: TotalOperatorCosts):
        self.__costs[key] = value
        self.updateTotals(value)

    def __getitem__(self, item: str) -> TotalOperatorCosts:
        return self.__costs[item]

    def updateTotals(self, value: TotalOperatorCosts):
        for mode, cost in value:
            self.total += cost

    def __mul__(self, other):
        out = CollectedTotalOperatorCosts()
        for mode in self.__costs.keys():
            out[mode] = self[mode] * other
        return out

    def __add__(self, other):
        out = CollectedTotalOperatorCosts()
        for mode in other.__costs.keys():
            if mode in self.__costs:
                out[mode] = self[mode] + other[mode]
            else:
                out[mode] = other[mode]
        return out

    def __iadd__(self, other):
        for mode in other.__costs.keys():
            if mode in self.__costs:
                self[mode] = self[mode] + other[mode]
            else:
                self[mode] = other[mode]
        return self


class Microtype:
    def __init__(self, microtypeID: str, networks: NetworkCollection, costs=None):
        self.microtypeID = microtypeID
        if costs is None:
            costs = dict()
        self.mode_names = set(networks.getModeNames())
        self.networks = networks
        self.updateModeCosts(costs)

    def updateModeCosts(self, costs):
        for (mode, modeCosts) in costs.items():
            assert (isinstance(mode, str) and isinstance(modeCosts, Costs))
            self.networks.modes[mode].costs = modeCosts

    def updateNetworkSpeeds(self, nIters=None):
        self.networks.updateModes(nIters)

    def getModeSpeeds(self) -> dict:
        return {mode: self.getModeSpeed(mode) for mode in self.mode_names}

    def getModeSpeed(self, mode) -> float:
        return self.networks.modes[mode].getSpeed()

    def getModeFlow(self, mode) -> float:
        return self.networks.demands.getRateOfPMT(mode)

    def getModeDemandForPMT(self, mode):
        return self.networks.demands.getRateOfPMT(mode)

    def addModeStarts(self, mode, demand):
        self.networks.demands.addModeStarts(mode, demand)

    def addModeEnds(self, mode, demand):
        self.networks.demands.addModeEnds(mode, demand)

    def addModeDemandForPMT(self, mode, demand, trip_distance_in_miles):
        self.networks.demands.addModeThroughTrips(mode, demand, trip_distance_in_miles)

    def setModeDemand(self, mode, demand, trip_distance_in_miles):
        self.networks.demands.setSingleDemand(mode, demand, trip_distance_in_miles)
        self.networks.updateModes()

    def resetDemand(self):
        self.networks.resetModes()
        self.networks.demands.resetDemand()

    def getStartAndEndRate(self, mode: str) -> (float, float):
        return self.networks.demands.getStartRate(mode), self.networks.demands.getStartRate(mode)

    def getModeMeanDistance(self, mode: str):
        return self.networks.demands.getAverageDistance(mode)

    def getThroughTimeCostWait(self, mode: str, distanceInMiles: float) -> ChoiceCharacteristics:
        speedMilesPerHour = np.max([self.getModeSpeed(mode), 0.01]) * 2.23694
        if np.isnan(speedMilesPerHour):
            speedMilesPerHour = self.getModeSpeed("auto")
        timeInHours = distanceInMiles / speedMilesPerHour
        cost = distanceInMiles * self.networks.modes[mode].perMile
        wait = 0.
        accessTime = 0.
        protectedDistance = self.networks.modes[mode].getPortionDedicated() * distanceInMiles
        return ChoiceCharacteristics(timeInHours, cost, wait, accessTime, protectedDistance)

    def getStartTimeCostWait(self, mode: str) -> ChoiceCharacteristics:
        time = 0.
        cost = self.networks.modes[mode].perStart
        if mode in ['bus', 'rail']:
            wait = self.networks.modes[
                       'bus'].headwayInSec / 3600. / 4.  # TODO: Something better than average of start and end
        else:
            wait = 0.
        walkAccessTime = self.networks.modes[mode].getAccessDistance() * self.networks.modes[
            'walk'].speedInMetersPerSecond / 3600.0
        return ChoiceCharacteristics(time, cost, wait, walkAccessTime)

    def getEndTimeCostWait(self, mode: str) -> ChoiceCharacteristics:
        time = 0.
        cost = self.networks.modes[mode].perEnd
        if mode == 'bus':
            wait = self.networks.modes['bus'].headwayInSec / 3600. / 4.
        else:
            wait = 0.
        walkEgressTime = self.networks.modes[mode].getAccessDistance() * self.networks.modes[
            'walk'].speedInMetersPerSecond / 3600.0
        return ChoiceCharacteristics(time, cost, wait, walkEgressTime)

    def getFlows(self):
        return [mode.getPassengerFlow() for mode in self.networks.modes.values()]

    def getSpeeds(self):
        return [mode.getSpeed() for mode in self.networks.modes.values()]

    def getDemandsForPMT(self):
        return [mode.getPassengerFlow() for mode in
                self.networks.modes.values()]

    # def getPassengerOccupancy(self):
    #     return [self.getModeOccupancy(mode) for mode in self.modes]

    # def getTravelTimes(self):
    #     speeds = np.array(self.getSpeeds())
    #     speeds[~(speeds > 0)] = np.nan
    #     distances = np.array([self.getModeMeanDistance(mode) for mode in self.modes])
    #     return distances / speeds

    # def getTotalTimes(self):
    #     speeds = np.array(self.getSpeeds())
    #     demands = np.array(self.getDemandsForPMT())
    #     times = speeds * demands
    #     times[speeds == 0.] = np.inf
    #     return times

    def __str__(self):
        return 'Demand: ' + str(self.getFlows()) + ' , Speed: ' + str(self.getSpeeds())


class MicrotypeCollection:
    def __init__(self, modeData: dict):
        self.__microtypes = dict()
        self.modeData = modeData

    def __setitem__(self, key: str, value: Microtype):
        self.__microtypes[key] = value

    def __getitem__(self, item: str) -> Microtype:
        return self.__microtypes[item]

    def __contains__(self, item):
        return item in self.__microtypes

    def importMicrotypes(self, subNetworkData: pd.DataFrame, modeToSubNetworkData: pd.DataFrame):
        uniqueMicrotypes = subNetworkData["MicrotypeID"].unique()
        for microtypeID in uniqueMicrotypes:
            if microtypeID in self:
                self[microtypeID].resetDemand()
            else:
                subNetworkToModes = dict()
                modeToModeData = dict()
                allModes = set()
                for idx in subNetworkData.loc[subNetworkData["MicrotypeID"] == microtypeID].index:
                    joined = modeToSubNetworkData.loc[
                        modeToSubNetworkData['SubnetworkID'] == idx]
                    subNetwork = Network(subNetworkData, idx)
                    for n in joined.itertuples():
                        subNetworkToModes.setdefault(subNetwork, []).append(n.ModeTypeID.lower())
                        allModes.add(n.ModeTypeID.lower())
                for mode in allModes:
                    modeToModeData[mode] = self.modeData[mode]
                networkCollection = NetworkCollection(subNetworkToModes, modeToModeData, microtypeID)
                self[microtypeID] = Microtype(microtypeID, networkCollection)
                print("|  Loaded ", len(subNetworkData.loc[subNetworkData["MicrotypeID"] == microtypeID].index),
                      " subNetworks in microtype ", microtypeID)

    def __iter__(self) -> (str, Microtype):
        return iter(self.__microtypes.items())

    def getModeSpeeds(self) -> dict:
        return {idx: m.getModeSpeeds() for idx, m in self}

    def getOperatorCosts(self) -> CollectedTotalOperatorCosts:
        operatorCosts = CollectedTotalOperatorCosts()
        for mID, m in self:
            assert isinstance(m, Microtype)
            operatorCosts[mID] = m.networks.getModeOperatingCosts()
        return operatorCosts

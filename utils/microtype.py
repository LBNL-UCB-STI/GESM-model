#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
import pandas as pd

from .OD import TransitionMatrix, Allocation
from .choiceCharacteristics import ChoiceCharacteristics
from .network import Network, NetworkCollection, Costs, TotalOperatorCosts, CollectedNetworkStateData


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

    def toDataFrame(self):
        return pd.concat([val.toDataFrame([key]) for key, val in self.__costs.items()])


class Microtype:
    def __init__(self, microtypeID: str, networks: NetworkCollection, costs=None):
        self.microtypeID = microtypeID
        if costs is None:
            costs = dict()
        self.mode_names = set(networks.getModeNames())
        self.networks = networks
        self.updateModeCosts(costs)

    def __contains__(self, item):
        return item in self.mode_names

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

    # def setModeDemand(self, mode, demand, trip_distance_in_miles):
    #     self.networks.demands.setSingleDemand(mode, demand, trip_distance_in_miles)
    #     self.networks.updateModes()

    def resetDemand(self):
        self.networks.resetModes()
        self.networks.demands.resetDemand()

    def getModeStartAndEndRate(self, mode: str) -> (float, float):
        return self.networks.demands.getStartRate(mode), self.networks.demands.getStartRate(mode)

    def getModeStartRate(self, mode: str) -> float:
        return self.networks.demands.getStartRate(mode)

    def getModeMeanDistance(self, mode: str):
        return self.networks.demands.getAverageDistance(mode)

    # def getThroughTimeCostWait(self, mode: str, distanceInMiles: float) -> ChoiceCharacteristics:
    #     speedMilesPerHour = np.max([self.getModeSpeed(mode), 0.01]) * 2.23694
    #     if np.isnan(speedMilesPerHour):
    #         speedMilesPerHour = self.getModeSpeed("auto")
    #     timeInHours = distanceInMiles / speedMilesPerHour
    #     cost = distanceInMiles * self.networks.modes[mode].perMile
    #     wait = 0.
    #     accessTime = 0.
    #     protectedDistance = self.networks.modes[mode].getPortionDedicated() * distanceInMiles
    #     return ChoiceCharacteristics(timeInHours, cost, wait, accessTime, protectedDistance, distanceInMiles)

    def addStartTimeCostWait(self, mode: str, cc: ChoiceCharacteristics):
        if mode in self:
            cc.cost += self.networks.modes[mode].perStart
            if mode in ['bus', 'rail']:
                cc.wait_time += self.networks.modes[
                                    'bus'].headwayInSec / 3600. / 4.  # TODO: Something better than average of start and end
            cc.access_time += self.networks.modes[mode].getAccessDistance() * self.networks.modes[
                'walk'].speedInMetersPerSecond / 3600.0

    def addThroughTimeCostWait(self, mode: str, distanceInMiles: float, cc: ChoiceCharacteristics):
        if mode in self:
            speedMilesPerHour = max([self.getModeSpeed(mode), 0.01]) * 2.23694
            if np.isnan(speedMilesPerHour):
                speedMilesPerHour = self.getModeSpeed("auto")
            timeInHours = distanceInMiles / speedMilesPerHour
            cc.travel_time += timeInHours
            cc.cost += distanceInMiles * self.networks.modes[mode].perMile
            cc.distance += distanceInMiles
            cc.protected_distance += self.networks.modes[mode].getPortionDedicated() * distanceInMiles

    def addEndTimeCostWait(self, mode: str, cc: ChoiceCharacteristics):
        if mode in self:
            cc.cost += self.networks.modes[mode].perEnd
            if mode == 'bus':
                cc.wait_time += self.networks.modes['bus'].headwayInSec / 3600. / 4.
            cc.access_time += self.networks.modes[mode].getAccessDistance() * self.networks.modes[
                'walk'].speedInMetersPerSecond / 3600.0

    # def getStartTimeCostWait(self, mode: str) -> ChoiceCharacteristics:
    #     time = 0.
    #     cost = self.networks.modes[mode].perStart
    #     if mode in ['bus', 'rail']:
    #         wait = self.networks.modes[
    #                    'bus'].headwayInSec / 3600. / 4.  # TODO: Something better than average of start and end
    #     else:
    #         wait = 0.
    #     walkAccessTime = self.networks.modes[mode].getAccessDistance() * self.networks.modes[
    #         'walk'].speedInMetersPerSecond / 3600.0
    #     return ChoiceCharacteristics(time, cost, wait, walkAccessTime)
    #
    # def getEndTimeCostWait(self, mode: str) -> ChoiceCharacteristics:
    #     time = 0.
    #     cost = self.networks.modes[mode].perEnd
    #     if mode == 'bus':
    #         wait = self.networks.modes['bus'].headwayInSec / 3600. / 4.
    #     else:
    #         wait = 0.
    #     walkEgressTime = self.networks.modes[mode].getAccessDistance() * self.networks.modes[
    #         'walk'].speedInMetersPerSecond / 3600.0
    #     return ChoiceCharacteristics(time, cost, wait, walkEgressTime)

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
    def __init__(self, scenarioData):
        self.__timeStepInSeconds = 30.0
        self.__microtypes = dict()
        self.__scenarioData = scenarioData
        self.modeData = scenarioData["modeData"]
        self.transitionMatrix = None
        self.collectedNetworkStateData = CollectedNetworkStateData()
        self.__modeToMicrotype = dict()
        self.__numpyDemand = np.ndarray([0])
        self.__numpySpeed = np.ndarray([0])
        self.__diameters = np.ndarray([0])

    @property
    def diToIdx(self):
        return self.__scenarioData.diToIdx

    @property
    def odiToIdx(self):
        return self.__scenarioData.odiToIdx

    @property
    def modeToIdx(self):
        return self.__scenarioData.modeToIdx

    @property
    def dataToIdx(self):
        return self.__scenarioData.dataToIdx

    @property
    def microtypeIdToIdx(self):
        return self.__scenarioData.microtypeIdToIdx

    @property
    def diameters(self):
        return self.__diameters

    @property
    def numpySpeed(self):
        return self.__numpySpeed

    def updateNumpyDemand(self, data):
        np.copyto(self.__numpyDemand, data)

    def updateNetworkData(self):
        for m in self.__microtypes.values():
            # assert isinstance(m, Microtype)
            m.networks.updateModeData()
            for _, n in m.networks:
                # assert isinstance(n, Network)
                n.updateScenarioInputs()

    def __setitem__(self, key: str, value: Microtype):
        self.__microtypes[key] = value

    def __getitem__(self, item: str) -> Microtype:
        return self.__microtypes[item]

    def __contains__(self, item):
        return item in self.__microtypes

    def __len__(self):
        return len(self.__microtypes)

    # def getAllStartCosts(self, microtypeToIdx: dict, characteristicToIdx: dict) -> np.ndarray:
    #     out = np.ndarray((len(microtypeToIdx), len(characteristicToIdx)))
    #
    #     return out

    def microtypeNames(self):
        return list(self.__microtypes.keys())

    def getModeStartRatePerSecond(self, mode):
        return np.array([microtype.getModeStartRate(mode) / 3600. for mID, microtype in self])

    def importMicrotypes(self):
        # uniqueMicrotypes = subNetworkData["MicrotypeID"].unique()

        subNetworkData = self.__scenarioData["subNetworkData"]
        subNetworkCharacteristics = self.__scenarioData["subNetworkDataFull"]
        modeToSubNetworkData = self.__scenarioData["modeToSubNetworkData"]
        microtypeData = self.__scenarioData["microtypeIDs"]

        self.__diameters = np.zeros(len(microtypeData), dtype=float)
        for microtypeId, idx in self.microtypeIdToIdx.items():
            self.__diameters[idx] = microtypeData.loc[microtypeId, 'DiameterInMiles'] * 1609.34

        self.transitionMatrix = TransitionMatrix(self.microtypeIdToIdx,
                                                 diameters=self.__diameters)

        print(microtypeData.MicrotypeID.to_list())
        print(self.microtypeIdToIdx)
        if len(self.__microtypes) == 0:
            self.__numpyDemand = np.zeros(
                (len(self.microtypeIdToIdx), len(self.modeToIdx), len(self.dataToIdx)), dtype=float)
            self.__numpySpeed = np.zeros((len(self.microtypeIdToIdx), len(self.modeToIdx)), dtype=float)
            self.__modeToMicrotype = dict()

        for microtypeID, diameter in microtypeData.itertuples(index=False):
            if microtypeID in self:
                self[microtypeID].resetDemand()
            else:
                subNetworkToModes = OrderedDict()
                modeToModeData = OrderedDict()
                allModes = set()
                for idx in subNetworkCharacteristics.loc[subNetworkCharacteristics["MicrotypeID"] == microtypeID].index:
                    joined = modeToSubNetworkData.loc[
                        modeToSubNetworkData['SubnetworkID'] == idx]
                    subNetwork = Network(subNetworkData, subNetworkCharacteristics, idx, diameter, microtypeID,
                                         self.__numpySpeed[self.microtypeIdToIdx[microtypeID], :],
                                         self.modeToIdx)
                    for n in joined.itertuples():
                        subNetworkToModes.setdefault(subNetwork, []).append(n.ModeTypeID.lower())
                        allModes.add(n.ModeTypeID.lower())
                        self.__modeToMicrotype.setdefault(n.ModeTypeID.lower(), set()).add(microtypeID)
                for mode in allModes:
                    modeToModeData[mode] = self.modeData[mode]
                networkCollection = NetworkCollection(subNetworkToModes, modeToModeData, microtypeID,
                                                      self.__numpyDemand[self.microtypeIdToIdx[microtypeID], :, :],
                                                      self.__numpySpeed[self.microtypeIdToIdx[microtypeID], :],
                                                      self.dataToIdx, self.modeToIdx)
                self[microtypeID] = Microtype(microtypeID, networkCollection)
                self.collectedNetworkStateData.addMicrotype(self[microtypeID])

                print("|  Loaded ",
                      len(subNetworkCharacteristics.loc[subNetworkCharacteristics["MicrotypeID"] == microtypeID].index),
                      " subNetworks in microtype ", microtypeID)

    # @profile
    def transitionMatrixMFD(self, durationInHours, collectedNetworkStateData=None, tripStartRate=None):
        if collectedNetworkStateData is None:
            collectedNetworkStateData = self.collectedNetworkStateData
            writeData = True
        else:
            writeData = False

        if tripStartRate is None:
            tripStartRate = self.getModeStartRatePerSecond("auto")

        def v(n, v_0, n_0, n_other, minspeed=0.005) -> np.ndarray:
            n_eff = n + n_other
            v = v_0 * (1. - n_eff / n_0)
            v[v < minspeed] = minspeed
            v[v > v_0] = v_0[v > v_0]
            return v

        def outflow(n, L, v_0, n_0, n_other) -> np.ndarray:
            return v(n, v_0, n_0, n_other, 0.005) * n / L

        def inflow(n, X, L, v_0, n_0, n_other) -> np.ndarray:
            os = X @ (v(n, v_0, n_0, n_other, 0.005) * n / L)
            return os

        def spillback(n: np.ndarray, N_0: np.ndarray, demand: np.ndarray, inflow: np.ndarray, outflow: np.ndarray,
                      dt: float, n_other=0.0, criticalDensity=0.9) -> np.ndarray:
            requestedN = (demand + inflow - outflow) * dt + n
            criticalN = criticalDensity * (N_0 - n_other)
            overLimit = requestedN > criticalN
            counter = 0
            while np.any(overLimit):
                if np.all(overLimit):
                    if counter == 0:
                        vals = np.linspace(criticalDensity, 1.0, 5)
                    if counter <= 5:
                        criticalDensity = vals[counter]
                        criticalN = criticalDensity * (N_0 - n_other)
                        counter += 1
                    else:
                        print('Youre Effed')
                        return criticalN
                before = np.sum(requestedN)
                totalSpillback = np.sum(requestedN[overLimit] - criticalN[overLimit])
                toBeLimited = inflow[~overLimit]
                requestedN[~overLimit] += inflow[~overLimit] * totalSpillback / np.sum(
                    toBeLimited)  # TODO: Matrix math here?
                requestedN[overLimit] = criticalN[overLimit]
                overLimit = (requestedN / N_0) > criticalN
                after = np.sum(requestedN)
            return requestedN

        def dn(n, demand, L, X, v_0, n_0, n_other, dt) -> np.ndarray:
            inflowval = inflow(n, X, L, v_0, n_0, n_other)
            outflowval = outflow(n, L, v_0, n_0, n_other)
            return (demand + inflow(n, X, L, v_0, n_0, n_other) - outflow(n, L, v_0, n_0, n_other)) * dt

        # print(tripStartRate)
        characteristicL = np.zeros((len(self)), dtype=float)
        V_0 = np.zeros((len(self)), dtype=float)
        N_0 = np.zeros((len(self)), dtype=float)
        n_other = np.zeros((len(self)), dtype=float)
        n_init = np.zeros((len(self)), dtype=float)
        for microtypeID, microtype in self:
            idx = self.transitionMatrix.idx(microtypeID)
            for modes, autoNetwork in microtype.networks:
                if "auto" in autoNetwork:
                    # for autoNetwork in microtype.networks["auto"]:
                    networkStateData = collectedNetworkStateData[(microtypeID, modes)]
                    # nsd2 = autoNetwork.getNetworkStateData()
                    # assert (isinstance(autoNetwork, Network))
                    L_eff = autoNetwork.L - networkStateData.blockedDistance
                    characteristicL[idx] += autoNetwork.diameter * 1609.34
                    V_0[idx] = autoNetwork.freeFlowSpeed
                    N_0[idx] = L_eff * autoNetwork.jamDensity
                    n_other[idx] = networkStateData.nonAutoAccumulation
                    n_init[idx] = networkStateData.initialAccumulation
        #            tripStartRate[idx] = microtype.getModeStartRate("auto") / 3600.

        X = np.transpose(self.transitionMatrix.matrix.values)

        dt = self.__timeStepInSeconds
        ts = np.arange(0, durationInHours * 3600., dt)
        ns = np.zeros((len(self), np.size(ts)), dtype=float)
        vs = np.zeros((len(self), np.size(ts)), dtype=float)
        n_t = n_init.copy()

        for i, ti in enumerate(ts):
            # deltaN = dn(n_t, tripStartRate, characteristicL, X, V_0, N_0, n_other, dt)
            # otherval = deltaN + n_t
            # n_t += deltaN
            infl = inflow(n_t, X, characteristicL, V_0, N_0, n_other)
            outfl = outflow(n_t, characteristicL, V_0, N_0, n_other)
            n_t = spillback(n_t, N_0, tripStartRate, infl, outfl, dt, n_other, 0.9)
            # print(otherval, n_t)
            # n_t[n_t > (N_0 - n_other)] = N_0[n_t > (N_0 - n_other)]
            n_t[n_t < 0] = 0.0
            pct = n_t / N_0
            ns[:, i] = np.squeeze(n_t)
            vs[:, i] = np.squeeze(v(n_t, V_0, N_0, n_other))

        # self.transitionMatrix.setAverageSpeeds(np.mean(vs, axis=1))
        averageSpeeds = np.mean(vs, axis=1)
        # print(averageSpeeds)

        self.__numpySpeed[:, self.modeToIdx['auto']] = averageSpeeds
        # np.copyto(self.__numpySpeed[:, self.modeToIdx['auto']], averageSpeeds)

        if writeData:
            for microtypeID, microtype in self:
                idx = self.transitionMatrix.idx(microtypeID)
                for modes, autoNetwork in microtype.networks:
                    if "auto" in autoNetwork:
                        networkStateData = collectedNetworkStateData[(microtypeID, modes)]
                        networkStateData.finalAccumulation = ns[idx, -1]
                        networkStateData.finalSpeed = vs[idx, -1]
                        # networkStateData.averageSpeed = averageSpeeds[idx]
                        networkStateData.n = np.squeeze(ns[idx, :])
                        networkStateData.v = np.squeeze(vs[idx, :])
                        networkStateData.t = np.squeeze(ts) + networkStateData.initialTime
        return {"t": np.transpose(ts), "v": np.transpose(vs), "n": np.transpose(ns),
                "max_accumulation": N_0}

    def __iter__(self) -> (str, Microtype):
        return iter(self.__microtypes.items())

    def getModeSpeeds(self) -> dict:
        return {mode: {microtypeId: self.__numpySpeed[microtypeIdx, modeIdx] for microtypeId, microtypeIdx in
                       self.microtypeIdToIdx.items()} for mode, modeIdx in self.modeToIdx.items()}
        # return {idx: m.getModeSpeeds() for idx, m in self}

    def getOperatorCosts(self) -> CollectedTotalOperatorCosts:
        operatorCosts = CollectedTotalOperatorCosts()
        for mID, microtype in self:
            assert isinstance(microtype, Microtype)
            operatorCosts[mID] = microtype.networks.getModeOperatingCosts()
        return operatorCosts

    def getStateData(self) -> CollectedNetworkStateData:
        data = CollectedNetworkStateData()
        for mID, microtype in self:
            data.addMicrotype(microtype)
        return data

    def importPreviousStateData(self, networkStateData: CollectedNetworkStateData):
        for mID, microtype in self:
            networkStateData.adoptPreviousMicrotypeState(microtype)

    def resetStateData(self):
        for _, nsd in self.collectedNetworkStateData:
            nsd.reset()

    def updateTransitionMatrix(self, transitionMatrix: TransitionMatrix):
        if self.transitionMatrix.names == transitionMatrix.names:
            self.transitionMatrix = transitionMatrix
        else:
            print("MICROTYPE NAMES IN TRANSITION MATRIX DON'T MATCH")

    def emptyTransitionMatrix(self):
        return TransitionMatrix(self.transitionMatrix.names)

    def filterAllocation(self, mode, inputAllocation: Allocation):
        validMicrotypes = self.__modeToMicrotype[mode]
        if inputAllocation.keys() == validMicrotypes:
            return inputAllocation.mapping
        else:
            return inputAllocation.filterAllocation(validMicrotypes)

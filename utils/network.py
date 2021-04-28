from math import sqrt, cosh, sinh, cos, sin
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.supply import TravelDemand, TravelDemands

np.seterr(all='ignore')

mph2mps = 1609.34 / 3600


class TotalOperatorCosts:
    def __init__(self):
        self.__costs = dict()
        self.__revenues = dict()
        self.__net = dict()

    def __setitem__(self, key: str, value: (float, float)):
        self.__costs[key] = value[0]
        self.__revenues[key] = value[1]
        self.__net[key] = value[0] - value[1]

    def __getitem__(self, item) -> float:
        return self.__net[item]

    def __iter__(self):
        return iter(self.__net.items())

    def __mul__(self, other):
        output = TotalOperatorCosts()
        for key in self.__costs.keys():
            output[key] = (self.__costs[key] * other, self.__revenues[key] * other)
        return output

    def __add__(self, other):
        output = TotalOperatorCosts()
        for key in self.__costs.keys():
            if key in self.__costs:
                output[key] = (self.__costs[key] + other.__costs[key], self.__revenues[key] + other.__revenues[key])
            else:
                output[key] = (other.__costs[key], other.__revenues[key])
        return output

    def __str__(self):
        return [key + ' ' + str(item) for key, item in self.__costs.items()]

    def toDataFrame(self, index=None):
        return pd.DataFrame(self.__costs, index=index)


class NetworkFlowParams:
    def __init__(self, smoothing, free_flow_speed, wave_velocity, jam_density, max_flow, avg_link_length):
        self.lam = smoothing
        self.freeFlowSpeed = free_flow_speed
        self.w = wave_velocity
        self.kappa = jam_density
        self.Q = max_flow
        self.avgLinkLength = avg_link_length


class Costs:
    def __init__(self, per_meter=0.0, per_start=0.0, per_end=0.0, vott_multiplier=1.0):
        self.perMeter = per_meter
        self.perStart = per_start
        self.perEnd = per_end
        self.vottMultiplier = vott_multiplier


class Mode:
    def __init__(self, networks=None, params=None, idx=None, name=None, travelDemandData=None, speedData=None):
        self.name = name
        self.params = params
        self._idx = idx
        # self._inds = self.initInds(idx)
        self.networks = networks
        # self._N_tot = 0.0
        self._N_eff = dict()
        self._L_blocked = dict()
        self._averagePassengerDistanceInSystem = 0.0
        self._VMT_tot = 0.0
        self._VMT = dict()
        self._speed = dict()
        self.__bad = False
        if networks is not None:
            for n in networks:
                n.addMode(self)
                self._N_eff[n] = 0.0
                self._L_blocked[n] = 0.0
                self._VMT[n] = 0.0
                self._speed[n] = n.base_speed
        self.travelDemand = TravelDemand(travelDemandData)
        self.__speedData = speedData

    # def initInds(self, idx):
    #     inds = dict()
    #     for column in self.params.columns():
    #         inds[column] = [(self.params.index.get_loc(idx), self.params.columns.get_loc(column))]
    #     return inds

    @property
    def relativeLength(self):
        # return self.params.to_numpy()[self._inds["VehicleSize"]]
        return self.params.at[self._idx, "VehicleSize"]

    @property
    def perStart(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self.params.at[self._idx, "PerStartCost"]

    @property
    def perEnd(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self.params.at[self._idx, "PerEndCost"]

    @property
    def perMile(self):
        # return self.params.to_numpy()[self._inds["PerMileCost"]]
        return self.params.at[self._idx, "PerMileCost"]

    def updateScenarioInputs(self):
        pass

    def updateDemand(self, travelDemand=None):
        if travelDemand is None:
            travelDemand = self.travelDemand
        else:
            self.travelDemand = travelDemand
        self._VMT_tot = travelDemand.rateOfPmtPerHour * self.relativeLength

    def getDemandForVmtPerHour(self):
        return self.travelDemand.rateOfPmtPerHour * self.relativeLength

    def getAccessDistance(self) -> float:
        return 0.0

    def updateModeBlockedDistance(self):
        for n in self.networks:
            self._L_blocked[n] = n.L_blocked[self.name]

    # def addVehicles(self, n):
    #     self._N_tot += n
    #     self.allocateVehicles()

    def getSpeedDifference(self, allocation: list):
        speeds = np.array([n.NEF(a * self._VMT_tot * mph2mps, self.name) for n, a in zip(self.networks, allocation)])
        return np.linalg.norm(speeds - np.mean(speeds))

    def assignVmtToNetworks(self):
        Ltot = sum([n.L for n in self.networks])
        for n in self.networks:
            VMT = self._VMT_tot * n.L / Ltot
            self._VMT[n] = VMT
            n.setVMT(self.name, self._VMT[n])
            # self._speed[n] = n.NEF()  # n.NEF(VMT * mph2mps, self.name)
            self._N_eff[n] = VMT / self._speed[n] * self.relativeLength
            n.setN(self.name, self._N_eff[n])

    # def allocateVehicles(self):
    #     """even"""
    #     n_networks = len(self.networks)
    #     for n in self.networks:
    #         n.N_eq[self.name] = self._N_tot / n_networks * self.relativeLength
    #         self._N[n] = self._N_tot / n_networks

    def __str__(self):
        return str([self.name + ': VMT=' + str(self._VMT_tot) + ', L_blocked=' + str(self._L_blocked)])

    def getSpeed(self):
        return self.networks[0].getBaseSpeed()

    def getN(self, network):
        return self._VMT[network] / self._speed[network] / self.relativeLength

    def getNs(self):
        return [self.getN(n) for n in self.networks]

    def getBlockedDistance(self, network):
        return self._L_blocked[network]

    # def updateN(self, demand: TravelDemand):
    #     n_new = self.getLittlesLawN(demand.rateOfPmtPerHour, demand.averageDistanceInSystemInMiles)
    #     self._N_tot = n_new
    #     self.allocateVehicles()

    # def getLittlesLawN(self, rateOfPmtPerHour: float, averageDistanceInSystemInMiles: float):
    #     speedInMilesPerHour = self.getSpeed() * 2.23694
    #     if not (speedInMilesPerHour >= 2.0):
    #         self.__bad = True
    #         speedInMilesPerHour = 2.0
    #     else:
    #         self.__bad = False
    #     return rateOfPmtPerHour / speedInMilesPerHour

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self.networks]):
            return 0.0
        else:
            return self.travelDemand.rateOfPmtPerHour

    def getOperatorCosts(self) -> float:
        return 0.0

    def getOperatorRevenues(self) -> float:
        return 0.0

    def getPortionDedicated(self) -> float:
        return 0.0


class WalkMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str, travelDemandData=None, speedData=None) -> None:
        super(WalkMode, self).__init__(travelDemandData=travelDemandData)
        self.name = "walk"
        self.params = modeParams
        self._idx = idx
        # self._inds = self.initInds(idx)
        self.networks = networks
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._N_eff[n] = 0.0
            self._speed[n] = n.base_speed

    @property
    def speedInMetersPerSecond(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self.params.at[self._idx, "SpeedInMetersPerSecond"]

    def getSpeed(self):
        return self.speedInMetersPerSecond


class BikeMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str, travelDemandData=None, speedData=None) -> None:
        super(BikeMode, self).__init__(travelDemandData=travelDemandData)
        self.name = "bike"
        self.params = modeParams
        self._idx = idx
        # self._inds = self.initInds(idx)
        self.networks = networks
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._N_eff[n] = 0.0
            self._speed[n] = n.base_speed
        self.bikeLanePreference = 2.0

    @property
    def speedInMetersPerSecond(self):
        # return self.params.to_numpy()[self._inds["PerEndCost"]]
        return self.params.at[self._idx, "SpeedInMetersPerSecond"]

    def getSpeed(self):
        return self.speedInMetersPerSecond

    # def allocateVehicles(self):
    #     """by length"""
    #     L_tot = sum([(n.L + n.L * (self.bikeLanePreference - 1) * n.dedicated) for n in self.networks])
    #     for n in self.networks:
    #         n.N_eq[self.name] = (n.L + n.L * (
    #                 self.bikeLanePreference - 1) * n.dedicated) * self._N_tot / L_tot * self.relativeLength
    #         self._N[n] = (n.L + n.L * (self.bikeLanePreference - 1) * n.dedicated) * self._N_tot / L_tot

    def getPortionDedicated(self) -> float:
        if self._VMT_tot > 0:
            tot = 0.0
            tot_dedicated = 0.0
            for key, val in self._VMT.items():
                tot += val
                if key.dedicated:
                    tot_dedicated += val
            return tot_dedicated / tot
        else:
            return 0.0


class RailMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str, travelDemandData=None, speedData=None) -> None:
        super(RailMode, self).__init__(travelDemandData=travelDemandData)
        self.name = "rail"
        self.params = modeParams
        self._idx = idx
        self.networks = networks
        # self.initInds(idx)
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._N_eff[n] = 0.0
            self._speed[n] = n.base_speed

    @property
    def routeAveragedSpeed(self):
        return self.params.at[self._idx, "SpeedInMetersPerSecond"]

    @property
    def vehicleOperatingCostPerHour(self):
        # return self.params.to_numpy()[self._inds["VehicleOperatingCostsPerHour"]]
        return self.params.at[self._idx, "VehicleOperatingCostsPerHour"]

    @property
    def fare(self):
        # return self.params.to_numpy()[self._inds["PerStartCost"]]
        return self.params.at[self._idx, "PerStartCost"]

    @property
    def headwayInSec(self):
        # return self.params.to_numpy()[self._inds["Headway"]]
        return self.params.at[self._idx, "Headway"]

    @property
    def stopSpacingInMeters(self):
        # return self.params.to_numpy()[self._inds["StopSpacing"]]
        return self.params.at[self._idx, "StopSpacing"]

    @property
    def portionAreaCovered(self):
        # return self.params.to_numpy()[self._inds["CoveragePortion"]]
        return self.params.at[self._idx, "CoveragePortion"]

    def updateDemand(self, travelDemand=None):
        if travelDemand is not None:
            self.travelDemand = travelDemand
        self._VMT_tot = self.getRouteLength() / self.headwayInSec

    def getAccessDistance(self) -> float:
        return self.stopSpacingInMeters / 4.0 / self.portionAreaCovered ** 2.0

    def getSpeed(self):
        return self.routeAveragedSpeed

    def getRouteLength(self):
        return sum([n.L for n in self.networks])

    def getOperatorCosts(self) -> float:
        return sum(self.getNs()) * self.vehicleOperatingCostPerHour

    def getOperatorRevenues(self) -> float:
        return self.travelDemand.tripStartRatePerHour * self.fare

    def getDemandForVmtPerHour(self):
        return self.getRouteLength() / self.headwayInSec * 3600.

    def assignVmtToNetworks(self):
        Ltot = sum([n.L for n in self.networks])
        for n in self.networks:
            VMT = self._VMT_tot * n.L / Ltot
            self._VMT[n] = VMT
            n.setVMT(self.name, self._VMT[n])
            self._speed[n] = self.routeAveragedSpeed
            self._N_eff[n] = VMT / self._speed[n] * self.relativeLength
            n.setN(self.name, self._N_eff[n])


class AutoMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str, travelDemandData=None, speedData=None) -> None:
        super(AutoMode, self).__init__(travelDemandData=travelDemandData)
        self.name = "auto"
        self.params = modeParams
        self._idx = idx
        self.networks = networks
        self.MFDmode = "single"
        self.override = False
        self.__speedData = speedData
        for n in networks:
            n.addMode(self)
            # self._N[n] = 0.0
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._N_eff[n] = 0.0
            self._speed[n] = n.base_speed

    @property
    def relativeLength(self):
        return self.params.at[self._idx, "VehicleSize"]

    def getSpeed(self):
        return self.__speedData[0]

    def x0(self):
        return np.array([1. / len(self.networks)] * len(self.networks))

    def constraints(self):
        return dict(type='eq', fun=lambda x: sum(x) - 1.0, jac=lambda x: [1.0] * len(x))

    def bounds(self):
        return [(0.0, 1.0)] * len(self.networks)

    def updateDemand(self, travelDemand=None):  # TODO: Why did I add this?
        if travelDemand is None:
            travelDemand = self.travelDemand
        else:
            self.travelDemand = travelDemand
        self._VMT_tot = travelDemand.rateOfPmtPerHour * self.relativeLength

    def assignVmtToNetworks(self):
        if len(self.networks) == 1:
            if self.MFDmode == "single":
                n = self.networks[0]
                self._VMT[n] = self._VMT_tot
                # self._speed[n] = n.NEF(self._VMT_tot * mph2mps, self.name, self.override)
                n.setVMT(self.name, self._VMT[n])
                self._N_eff[n] = self._VMT_tot / self._speed[n] * self.relativeLength
                n.setN(self.name, self._N_eff[n])
            else:
                n = self.networks[0]
                # self._speed[n] = n.getTransitionMatrixMeanSpeed()  # TODO: Check Units
                self._VMT[n] = self._VMT_tot
                n.setVMT(self.name, self._VMT[n])
                self._N_eff[n] = n.getNetworkStateData().finalAccumulation * self.relativeLength  # TODO: take avg
                n.setN(self.name, self._N_eff[n])
        elif len(self.networks) > 1:
            res = minimize(self.getSpeedDifference, self.x0(), constraints=self.constraints(), bounds=self.bounds())
            for n, a in zip(self.networks, res.x):
                self._VMT[n] = a * self._VMT_tot
                self._speed[n] = n.NEF(a * self._VMT_tot * mph2mps, self.name)
                n.setVMT(self.name, self._VMT[n])
                self._N_eff[n] = self._VMT[n] / self._speed[n]
                n.setN(self.name, self._N_eff[n])
        else:
            print("OH NO!")

    # def allocateVehicles(self):
    #     """for constant car speed"""
    #     current_allocation = []
    #     blocked_lengths = []
    #     lengths = []
    #     other_mode_n_eq = []
    #     jammed = []
    #     for n in self.networks:
    #         other_modes = list(n.N_eq.keys())
    #         if self.name in other_modes:
    #             other_modes.remove(self.name)
    #         current_allocation.append(self._N[n])
    #         blocked_lengths.append(n.getBlockedDistance())
    #         lengths.append(n.L)
    #         other_mode_n_eq.append(sum([n.N_eq[m] for m in other_modes]))
    #         jammed.append(n.isJammed)
    #     n_eq_other = sum(other_mode_n_eq)
    #     L_tot = sum(lengths)
    #     L_blocked_tot = sum(blocked_lengths)
    #     density_av = (self._N_tot + n_eq_other) / (L_tot - L_blocked_tot) * self.relativeLength
    #     if any(jammed):
    #         print(density_av)
    #     if self._N_tot > 0:
    #         n_new = np.nan_to_num(np.array(
    #             [density_av * (lengths[i] - blocked_lengths[i]) - other_mode_n_eq[i] for i in range(len(lengths))]))
    #     else:
    #         n_new = np.array([0.0] * len(lengths))
    #     should_be_empty = (n_new < 0) | np.array(jammed)
    #     to_reallocate = np.sum(n_new[should_be_empty])
    #     n_new[~should_be_empty] += to_reallocate * n_new[~should_be_empty] / np.sum(n_new[~should_be_empty])
    #     n_new[should_be_empty] = 0
    #     for ind, n in enumerate(self.networks):
    #         n.N_eq[self.name] = n_new[ind] * self.relativeLength
    #         self._N[n] = n_new[ind]


class BusMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str, travelDemandData=None, speedData=None) -> None:
        super(BusMode, self).__init__(travelDemandData=travelDemandData)
        self.name = "bus"
        self.params = modeParams
        self._idx = modeParams.index.get_loc(idx)
        self.__params = modeParams.to_numpy()
        self.modeParamsColumnToIdx = {i: modeParams.columns.get_loc(i) for i in modeParams.columns}
        self.networks = networks
        self.__operatingL = dict()
        self.__speedData = speedData
        for n in networks:
            n.addMode(self)
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._N_eff[n] = 0.0
            self._speed[n] = n.base_speed
            self.__operatingL[n] = self.updateOperatingL(n)

        self.__routeLength = self.updateRouteLength()
        self.travelDemand = TravelDemand()
        self.routeAveragedSpeed = self.getSpeed()
        self.occupancy = 0.0
        self.updateModeBlockedDistance()

    @property
    def routeAveragedSpeed(self):
        return self.__speedData[0]

    @routeAveragedSpeed.setter
    def routeAveragedSpeed(self, spd):
        self.__speedData[0] = spd

    @property
    def headwayInSec(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["Headway"]]

    @property
    def passengerWaitInSec(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["PassengerWait"]]

    @property
    def passengerWaitInSecDedicated(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["PassengerWaitDedicated"]]

    @property
    def stopSpacingInMeters(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["StopSpacing"]]

    @property
    def minStopTimeInSec(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["MinStopTime"]]

    @property
    def fare(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["PerStartCost"]]

    @property
    def perStart(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["PerStartCost"]]

    @property
    def perEnd(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["PerEndCost"]]

    @property
    def perMile(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["PerMileCost"]]

    @property
    def vehicleOperatingCostPerHour(self):
        return self.__params[self._idx, self.modeParamsColumnToIdx["VehicleOperatingCostPerHour"]]

    @property
    def routeDistanceToNetworkDistance(self) -> float:
        """
        Changed January 2021: Removed need for car-only subnnetworks.
        Now buses only run on a fixed portion of the bus/car subnetwork
        """
        return self.__params[self._idx, self.modeParamsColumnToIdx["CoveragePortion"]]

    @property
    def relativeLength(self):
        # return self.params.to_numpy()[self._inds["VehicleSize"]]
        return self.__params[self._idx, self.modeParamsColumnToIdx["VehicleSize"]]

    def updateScenarioInputs(self):
        self.__params = self.params.to_numpy()
        for n in self.networks:
            self._L_blocked[n] = 0.0
            self._VMT[n] = 0.0
            self._N_eff[n] = 0.0
            self._speed[n] = n.base_speed
            self.__operatingL[n] = self.updateOperatingL(n)

    def updateDemand(self, travelDemand=None):
        if travelDemand is not None:
            self.travelDemand = travelDemand
        self._VMT_tot = self.getDemandForVmtPerHour()

    def getAccessDistance(self) -> float:
        """Order of magnitude estimate for average walking distance to nearest stop"""
        return self.stopSpacingInMeters / 4.0 / self.routeDistanceToNetworkDistance

    def getDemandForVmtPerHour(self):
        return self.getRouteLength() / self.headwayInSec * 3600. / 1609.34

    def getOperatingL(self, network) -> float:
        return self.__operatingL[network]

    def updateOperatingL(self, network) -> float:
        """Changed January 2021: Buses only operate on a portion of subnetwork"""
        if network.dedicated:
            return network.L
        else:
            dedicatedDistance = sum([n.L for n in self.networks if n.dedicated])
            totalDistance = sum([n.L for n in self.networks])
            return self.routeDistanceToNetworkDistance * totalDistance * network.L / (totalDistance - dedicatedDistance)

    def getN(self, network=None):
        """Changed January 2021: Buses only operate on a portion of subnetwork"""
        if network:
            return self.getOperatingL(network) / self.routeAveragedSpeed / self.headwayInSec
        else:
            return self.getRouteLength() / self.routeAveragedSpeed / self.headwayInSec

    def getRouteLength(self):
        return self.__routeLength

    def updateRouteLength(self):
        return sum([n.L for n in self.networks]) * self.routeDistanceToNetworkDistance

    def getSubNetworkSpeed(self, network):
        # averageStopDuration = self.min_stop_time + self.passenger_wait * (
        #         self.travelDemand.tripStartRate + self.travelDemand.tripEndRate) / (
        #                               self.routeAveragedSpeed / self.stop_spacing * self.N_tot)
        # return base_speed / (1 + averageStopDuration * base_speed / self.stop_spacing)
        if network.dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        numberOfStopsInSubnetwork = self.getOperatingL(network) / self.stopSpacingInMeters
        numberOfStopsInRoute = self.getRouteLength() / self.stopSpacingInMeters
        pass_per_stop = (self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour
                         ) / numberOfStopsInRoute * self.headwayInSec / 3600.
        stopping_time = numberOfStopsInSubnetwork * self.minStopTimeInSec
        stopped_time = perPassenger * pass_per_stop * numberOfStopsInSubnetwork + stopping_time
        driving_time = self.getOperatingL(network) / network.base_speed
        spd = self.getOperatingL(network) / (stopped_time + driving_time)
        if np.isnan(spd):
            spd = 0.1
            self.__bad = True
        else:
            self.__bad = False
        return spd

    def getSpeeds(self):
        speeds = []
        for n in self.networks:
            if n.L == 0:
                speeds.append(np.inf)
            else:
                bus_speed = self.getSubNetworkSpeed(n)
                speeds.append(bus_speed)
        return speeds

    def getSpeed(self):
        meters = np.zeros(len(self.networks), dtype=float)
        seconds = np.zeros(len(self.networks), dtype=float)
        for idx, n in enumerate(self.networks):
            if n.L > 0:
                # n_bus = self.getN(n)
                bus_speed = self.getSubNetworkSpeed(n)
                meters[idx] = self.getOperatingL(n)
                seconds[idx] = self.getOperatingL(n) / bus_speed
        if np.sum(seconds) > 0:
            spd = np.sum(meters) / np.sum(seconds)
            return spd
        else:
            return next(iter(self.networks)).getBaseSpeed()

    def calculateBlockedDistance(self, network) -> float:
        if network.dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        # bs = network.base_speed
        if network.base_speed > 0:
            numberOfStops = self.getRouteLength() / self.stopSpacingInMeters
            # numberOfBuses = self.getN(network)
            meanTimePerStop = (self.minStopTimeInSec + self.headwayInSec * perPassenger * (
                    self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour) / (
                                       numberOfStops * 3600.0))
            portionOfTimeStopped = min([meanTimePerStop * meanTimePerStop / self.headwayInSec, 1.0])
            # TODO: Think through this more fully. Is this the right way to scale up this time to distance?
            out = portionOfTimeStopped * network.avgLinkLength * self.getN(network)
            # portionOfRouteBlocked = out / self.routeLength
        else:
            out = 0
        return out

    def updateModeBlockedDistance(self):
        for n in self.networks:
            L_blocked = self.calculateBlockedDistance(n)
            self._L_blocked[n] = L_blocked
            n.L_blocked[self.name] = L_blocked  # * self.getRouteLength() / n.L
            n.getNetworkStateData().blockedDistance += L_blocked

    def assignVmtToNetworks(self):
        speeds = self.getSpeeds()
        times = []
        lengths = []
        for ind, n in enumerate(self.networks):
            spd = speeds[ind]
            if spd < 0.1:
                # print("Speed to small: ", spd)
                spd = 0.1
            times.append(self.getOperatingL(n) / spd)
            lengths.append(self.getOperatingL(n))
        for ind, n in enumerate(self.networks):
            assert isinstance(n, Network)
            if speeds[ind] >= 0:
                VMT = self._VMT_tot * lengths[ind] / self.getRouteLength()
                self._VMT[n] = VMT
                n.setVMT(self.name, self._VMT[n])
                n.updateBaseSpeed()
                self._speed[n] = self.getSubNetworkSpeed(n)
                self._N_eff[n] = VMT / self._speed[n] * self.relativeLength
                n.setN(self.name, self._N_eff[n])
                n.getNetworkStateData().nonAutoAccumulation += self._N_eff[n]
            else:
                print("BAD WHY IS THIS SPEED NEGATIVE")
        self.updateCommercialSpeed()

    def updateCommercialSpeed(self):
        self.routeAveragedSpeed = self.getRouteLength() / sum(
            [self.getOperatingL(n) / spd for n, spd in self._speed.items()])

    def getOccupancy(self) -> float:
        return self.travelDemand.averageDistanceInSystemInMiles / (
                self.routeAveragedSpeed * 2.23694) * self.travelDemand.tripStartRatePerHour / self.getN()

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self.networks]):
            return 0.0
        elif self.occupancy > 100:
            return np.nan
        else:
            return self.travelDemand.rateOfPmtPerHour

    def getOperatorCosts(self) -> float:
        return sum(self.getNs()) * self.vehicleOperatingCostPerHour

    def getOperatorRevenues(self) -> float:
        return self.travelDemand.tripStartRatePerHour * self.fare


class Network:
    def __init__(self, data, characteristics, idx, diameter=None, microtypeID=None, modeToMicrotypeSpeed=None,
                 modeToIdx=None):
        self.data = data
        self.__data = data.to_numpy()
        self.characteristics = characteristics
        self.charColumnToIdx = {i: characteristics.columns.get_loc(i) for i in characteristics.columns}
        self.dataColumnToIdx = {i: data.columns.get_loc(i) for i in data.columns}
        self.microtypeID = microtypeID
        self._idx = data.index.get_loc(idx)
        self.type = self.characteristics.iat[self._idx, self.charColumnToIdx["Type"]]
        self.L_blocked = dict()
        self._modes = dict()
        self.dedicated = characteristics.loc[idx, "Dedicated"]
        self.isJammed = False
        self._VMT = dict()
        self._N_eff = dict()
        self._networkStateData = NetworkStateData().initFromNetwork(self)
        self._N_init = 0.0
        self._N_final = 0.0
        self._V_mean = self.freeFlowSpeed
        # These are for debugging and can likely be removed
        self._Q_prev = 0.0
        self._Q_curr = 0.0
        self._V_init = 0.0
        self._V_final = self.freeFlowSpeed
        self._V_steadyState = self.freeFlowSpeed
        self.__modeToIdx = modeToIdx
        self.__modeToMicrotypeSpeed = modeToMicrotypeSpeed
        self.base_speed = self.freeFlowSpeed
        if diameter is None:
            self.__diameter = 1.0
        else:
            self.__diameter = diameter

    # @property
    # def type(self):
    #     return self.characteristics.iat[self._idx, self.charColumnToIdx["Type"]]

    @property
    def autoSpeed(self):
        return self.__modeToMicrotypeSpeed[self.__modeToIdx['auto']]

    # @base_speed.setter
    # def base_speed(self, spd):
    #     self.__modeToMicrotypeSpeed[self.__modeToIdx['auto']] = spd

    @property
    def avgLinkLength(self):
        return self.__data[self._idx, self.dataColumnToIdx["avgLinkLength"]]

    @property
    def freeFlowSpeed(self):
        return self.__data[self._idx, self.dataColumnToIdx["vMax"]]

    @property
    def jamDensity(self):
        return self.__data[self._idx, self.dataColumnToIdx["densityMax"]]

    @property
    def L(self):
        return self.__data[self._idx, self.dataColumnToIdx["Length"]]

    @property
    def diameter(self):
        return self.__diameter

    def __str__(self):
        return str(tuple(self._VMT.keys()))

    def __contains__(self, mode):
        return mode in self._modes

    def updateScenarioInputs(self):
        np.copyto(self.__data, self.data.to_numpy())

    def getAccumulationExcluding(self, mode: str):
        return np.sum(acc for m, acc in self._N_eff.items() if m != mode)

    def resetAll(self):
        self.L_blocked = dict()
        self._modes = dict()
        self.base_speed = self.freeFlowSpeed
        self.isJammed = False

    def resetModes(self):
        for mode in self._modes.values():
            # self.N_eq[mode.name] = mode.getN(self) * mode.params.relativeLength
            self._VMT[mode] = mode._VMT[self]
            self._N_eff[mode] = mode._N_eff[self]
            self.L_blocked[mode.name] = mode.getBlockedDistance(self)
        self.isJammed = False
        self.base_speed = self.freeFlowSpeed
        # mode.reset()

    def setVMT(self, mode: str, VMT: float):
        self._VMT[mode] = VMT

    def setN(self, mode: str, N: float):
        self._N_eff[mode] = N

    def updateBaseSpeed(self, override=False):
        # out = self.NEF(overrideMatrix=override)
        self.base_speed = self.NEF(overrideMatrix=override)

    def getSpeedFromMFD(self, N):
        L_tot = self.L - self.getBlockedDistance()
        N_0 = self.jamDensity * L_tot
        return self.freeFlowSpeed * (1. - N / N_0)

    def NEF2(self) -> float:
        if self.type == "Road":
            return self._V_mean
        else:
            return self.freeFlowSpeed

    def NEF(self, Q=None, modeIgnored=None, overrideMatrix=False) -> float:
        if self.type == 'Road':
            if 'auto' in self.getModeNames() and not overrideMatrix:
                # print("THIS WILL BREAK THINGS")
                return self.__modeToMicrotypeSpeed[self.__modeToIdx['auto']]
                # return self._networkStateData.averageSpeed
            else:
                if Q is None:
                    Qtot = sum([VMT for VMT in self._VMT.values()]) * mph2mps
                else:
                    Qtot = Q
                    for mode, Qmode in self._VMT.items():
                        if mode != modeIgnored:
                            Qtot += Qmode * mph2mps
                if Qtot == 0:
                    return self.freeFlowSpeed
                self._Q_curr = Qtot
                L_tot = self.L - self.getBlockedDistance()
                L_0 = 10 * 1609.34  # TODO: Get average distance, don't hardcode
                t = 3 * 3600.  # TODO: Add timestep duration in seconds
                N_0 = self.jamDensity * L_tot
                V_0 = self.freeFlowSpeed
                N_init = self._N_init
                if N_0 ** 2. / 4. >= N_0 * Qtot / V_0:
                    # Stable state
                    A = sqrt(N_0 ** 2. / 4. - N_0 * Qtot / V_0)
                    var = A * V_0 * t / (N_0 * L_0)
                    N_final = N_0 / 2 - A * ((N_0 / 2 - N_init) * cosh(var) + A * sinh(var)) / (
                            (N_0 / 2 - N_init) * sinh(var) + A * cosh(var))
                    V_init = self.getSpeedFromMFD(N_init)
                    V_final = self.getSpeedFromMFD(N_final)
                    V_steadyState = self.getSpeedFromMFD(N_0 / 2 - A)
                else:
                    Aprime = sqrt(N_0 * Qtot / V_0 - N_0 ** 2. / 4.)
                    var = Aprime * V_0 * t / (N_0 * L_0)
                    N_final = N_0 / 2 - Aprime * ((N_0 / 2 - N_init) * cos(var) + Aprime * sin(var)) / (
                            (N_0 / 2 - N_init) * sin(var) + Aprime * cos(var))
                    V_init = self.getSpeedFromMFD(N_init)
                    V_final = self.getSpeedFromMFD(N_final)
                    V_steadyState = 0
                self._networkStateData.N_final = N_final
                self._networkStateData.V_init = V_init
                self._networkStateData.V_final = V_final
                self._networkStateData.V_steadyState = V_steadyState
                if overrideMatrix:
                    self.base_speed = max([0.1, (V_init + V_final) / 2.0])
                return max([0.1, (V_init + V_final) / 2.0])  # TODO: Actually take the integral
        else:
            return self.freeFlowSpeed

    def getBaseSpeed(self):
        if self.base_speed > 0.01:
            return self.base_speed
        else:
            return 0.01

    def updateBlockedDistance(self):
        for mode in self._modes.values():
            mode.updateModeBlockedDistance()

    def containsMode(self, mode: str) -> bool:
        return mode in self._modes.keys()

    def getBlockedDistance(self) -> float:
        if self.L_blocked:
            return sum(list(self.L_blocked.values()))
        else:
            return 0.0

    def addMode(self, mode: Mode):
        self._modes[mode.name] = mode
        self.L_blocked[mode.name] = 0.0
        self._VMT[mode.name] = 0.0
        self._N_eff[mode.name] = 0.0
        return self

    def getModeNames(self) -> list:
        return list(self._modes.keys())

    def getModeValues(self) -> list:
        return list(self._modes.values())

    def updateFromMFD(self, v, n):
        self._networkStateData.finalSpeed = v[-1]
        self._networkStateData.finalAccumulation = n[-1]
        # self._networkStateData.averageSpeed = np.mean(v)

    # def getTransitionMatrixMeanSpeed(self):
    #     return self._networkStateData.averageSpeed

    def getNetworkStateData(self):
        return self._networkStateData

    def setInitialStateData(self, oldNetworkStateData):
        if len(oldNetworkStateData.n > 0):
            self._networkStateData.initialAccumulation = oldNetworkStateData.n[-1]
            self._networkStateData.initialSpeed = oldNetworkStateData.v[-1]
            self._networkStateData.initialTime = oldNetworkStateData.t[-1]
        # self._networkStateData.nonAutoAccumulation = oldNetworkStateData.nonAutoAccumulation
        # self._networkStateData.blockedDistance = oldNetworkStateData.blockedDistance


class NetworkCollection:
    def __init__(self, networksAndModes=None, modeToModeData=None, microtypeID=None, demandData=None, speedData=None,
                 dataToIdx=None, modeToIdx=None, verbose=False):
        self._networks = dict()
        self.modeToNetwork = dict()
        self.__modes = []

        if demandData is None:
            self.__demandData = np.ndarray(0)
            self.__speedData = np.ndarray(0)
            self.__dataToIdx = dict()
            self.__modeToIdx = dict()
        else:
            self.__demandData = demandData
            self.__speedData = speedData
            self.__dataToIdx = dataToIdx
            self.__modeToIdx = modeToIdx

        if isinstance(networksAndModes, Dict) and isinstance(modeToModeData, Dict):
            self.populateNetworksAndModes(networksAndModes, modeToModeData, microtypeID)
        self.modes = dict()
        self.demands = TravelDemands([])
        self.verbose = verbose
        # self.resetModes()

    def populateNetworksAndModes(self, networksAndModes, modeToModeData, microtypeID):
        # modeToNetwork = dict()
        if isinstance(networksAndModes, Dict):
            for (network, modeNames) in networksAndModes.items():
                assert (isinstance(network, Network))
                sortedModeNames = tuple(sorted(modeNames))
                self._networks[sortedModeNames] = network
                for modeName in modeNames:
                    if modeName in self.modeToNetwork:
                        self.modeToNetwork[modeName].append(network)
                    else:
                        self.modeToNetwork[modeName] = [network]

        else:
            print('Bad NetworkCollection Input')
        for (modeName, networks) in self.modeToNetwork.items():
            assert (isinstance(modeName, str))
            assert (isinstance(networks, List))
            params = modeToModeData[modeName]

            if modeName == "bus":
                self.__speedData[self.__modeToIdx[modeName]] = networks[0].base_speed
                self.__modes.append(BusMode(networks, params, microtypeID,
                                            travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                                            speedData=self.__speedData[self.__modeToIdx[modeName], None]))
            elif modeName == "auto":
                self.__speedData[self.__modeToIdx[modeName]] = networks[0].base_speed
                self.__modes.append(AutoMode(networks, params, microtypeID,
                                             travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                                             speedData=self.__speedData[self.__modeToIdx[modeName], None]))
            elif modeName == "walk":
                self.__speedData[self.__modeToIdx[modeName]] = params.loc[microtypeID, 'SpeedInMetersPerSecond']
                self.__modes.append(WalkMode(networks, params, microtypeID,
                                             travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                                             speedData=self.__speedData[self.__modeToIdx[modeName], None]))
            elif modeName == "bike":
                self.__speedData[self.__modeToIdx[modeName]] = params.loc[microtypeID, 'SpeedInMetersPerSecond']
                self.__modes.append(BikeMode(networks, params, microtypeID,
                                             travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                                             speedData=self.__speedData[self.__modeToIdx[modeName], None]))
            elif modeName == "rail":
                self.__speedData[self.__modeToIdx[modeName]] = params.loc[microtypeID, 'SpeedInMetersPerSecond']
                self.__modes.append(RailMode(networks, params, microtypeID,
                                             travelDemandData=self.__demandData[self.__modeToIdx[modeName], :],
                                             speedData=self.__speedData[self.__modeToIdx[modeName], None]))
            else:
                print("BAD!")
                Mode(networks, params, microtypeID, "bad")

    def updateModeData(self):
        for m in self.__modes:
            m.updateScenarioInputs()

    def isJammed(self):
        return np.any([n.isJammed for n in self._networks])

    def resetModes(self):
        allModes = [n.getModeValues() for n in self._networks.values()]
        uniqueModes = set([item for sublist in allModes for item in sublist])
        for n in self._networks.values():
            n.isJammed = False
        self.modes = dict()
        for m in uniqueModes:
            # m.updateN(TravelDemand())
            self.modes[m.name] = m
            self.demands[m.name] = m.travelDemand
        # self.updateNetworks()

    # @profile
    def updateModes(self, nIters: int = 1):
        # allModes = [n.getModeValues() for n in self._networks]
        # uniqueModes = set([item for sublist in allModes for item in sublist])
        oldSpeeds = self.getModeSpeeds()
        for m in self.modes.values():
            m.updateDemand(self.demands[m.name])
        for it in range(nIters):
            for modes, n in self:
                n.getNetworkStateData().resetBlockedDistance()
                n.getNetworkStateData().resetNonAutoAccumulation()
            for m in self.modes.values():  # uniqueModes:
                m.assignVmtToNetworks()
                for n in m.networks:
                    n.updateBaseSpeed()
                m.updateModeBlockedDistance()
                # m.updateCommercialSpeed()
                # self.getModeSpeeds()
                # m.updateN(self.demands[m.name])
            # self.updateNetworks()
            # self.updateMFD()
            # if self.verbose:
            #     print(str(self))
            # if np.any([n.isJammed for n in self._networks]):
            #     break
            newSpeeds = self.getModeSpeeds()
            if np.linalg.norm(oldSpeeds - newSpeeds) < 1e-9:
                break
            else:
                oldSpeeds = newSpeeds

    def updateNetworks(self):
        for n in self._networks:
            n.resetModes()

    def __getitem__(self, item):
        return [n for idx, n in self._networks.items() if item in idx]

    def __str__(self):
        return str([n.base_speed for n in self._networks])

    def __iter__(self):
        return iter(self._networks.items())

    def __contains__(self, mode):
        return mode in self.modeToNetwork

    def getModeNames(self) -> list:
        return list(self.modeToNetwork.keys())

    def getModeSpeeds(self) -> np.array:
        return np.array([m.getSpeed() for m in self.modes.values()])

    def getModeOperatingCosts(self):
        out = TotalOperatorCosts()
        for name, mode in self.modes.items():
            out[name] = (mode.getOperatorCosts(), mode.getOperatorRevenues())
        return out


class NetworkStateData:
    def __init__(self, data=None):
        if data is None:
            self.__data = dict()
            self.finalAccumulation = 0.0
            self.finalProduction = 0.0
            self.initialSpeed = np.inf
            self.finalSpeed = np.inf
            self.steadyStateSpeed = np.inf
            self.initialAccumulation = 0.0
            self.nonAutoAccumulation = 0.0
            self.blockedDistance = 0.0
            # self.averageSpeed = 0.0
            self.initialTime = 0.0
            self.v = np.zeros(0)
            self.n = np.zeros(0)
            self.t = np.zeros(0)
        else:
            self.finalAccumulation = data.finalAccumulation
            self.finalProduction = data.finalProduction
            self.initialSpeed = data.initialSpeed
            self.finalSpeed = data.finalSpeed
            self.steadyStateSpeed = data.steadyStateSpeed
            self.initialAccumulation = data.initialAccumulation
            self.nonAutoAccumulation = data.nonAutoAccumulation
            self.blockedDistance = data.blockedDistance
            # self.averageSpeed = data.averageSpeed
            self.initialTime = data.initialTime
            self.v = data.v
            self.n = data.n
            self.t = data.t

    def initFromNetwork(self, network: Network):
        self.initialSpeed = network.freeFlowSpeed
        self.finalSpeed = network.freeFlowSpeed
        self.steadyStateSpeed = network.freeFlowSpeed
        # self.averageSpeed = network.freeFlowSpeed
        return self

    def resetBlockedDistance(self):
        self.blockedDistance = 0.0

    def resetNonAutoAccumulation(self):
        self.nonAutoAccumulation = 0.0

    def reset(self):
        self.__data = dict()
        self.finalAccumulation = 0.0
        self.finalProduction = 0.0
        self.initialSpeed = np.inf
        self.finalSpeed = np.inf
        self.steadyStateSpeed = np.inf
        self.initialAccumulation = 0.0
        self.nonAutoAccumulation = 0.0
        self.blockedDistance = 0.0
        # self.averageSpeed = 0.0
        self.initialTime = 0.0
        self.v = np.zeros(0)
        self.n = np.zeros(0)
        self.t = np.zeros(0)


class CollectedNetworkStateData:
    def __init__(self):
        self.__data = dict()

    def __setitem__(self, key, value: NetworkStateData):
        self.__data[key] = value

    def __getitem__(self, item) -> NetworkStateData:
        return self.__data[item]

    def getAutoProduction(self):
        prods = []
        for (mID, modes), val in self.__data.items():
            if "auto" in modes:
                if len(val.t) == 0:
                    prods.append(0.)
                else:
                    prods.append(np.sum(val.v * val.n) * (val.t[1] - val.t[0]))
        return np.array(prods)

    def addMicrotype(self, microtype):
        for modes, network in microtype.networks:
            self[(microtype.microtypeID, modes)] = network.getNetworkStateData()

    def adoptPreviousMicrotypeState(self, microtype):
        for modes, network in microtype.networks:
            network.setInitialStateData(self[(microtype.microtypeID, modes)])

    def getAutoSpeeds(self):
        speeds = []
        ns = []
        ts = None
        labels = []
        for (mID, modes), val in self.__data.items():
            if "auto" in modes:
                speeds.append(val.v)
                ns.append(val.n)
                if ts is None:
                    ts = val.t
                labels.append((mID, modes))
        return ts, np.stack(speeds, axis=-1), np.stack(ns, axis=-1), labels

    def __bool__(self):
        return len(self.__data) > 0

    def __iter__(self):
        return iter(self.__data.items())

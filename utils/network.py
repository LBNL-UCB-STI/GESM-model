from typing import List, Dict

import numpy as np
import pandas as pd
from math import isinf, isnan

from utils.supply import TravelDemand, TravelDemands

np.seterr(all='ignore')


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
    def __init__(self, networks: List, params: pd.DataFrame, idx: str, name: str):
        self.name = name
        self._N_tot = 0.0
        self._N = dict()
        self._L_blocked = dict()
        self.N_queued = 0.0
        self.queuing_speed = 0.01
        self.params = params
        self.__idx = idx # TODO: Replace with params.index.get_loc(idx) and use to_numpy() to access
        self._networks = networks
        self._averagePassengerDistanceInSystem = 0.0
        self.__bad = False
        for n in networks:
            n.addMode(self)
            self._N[n] = 0.0
            self._L_blocked[n] = 0.0
        self.travelDemand = TravelDemand()


    @property
    def relativeLength(self):
        return self.params.at[self.__idx, "VehicleSize"]

    @property
    def perStart(self):
        return self.params.at[self.__idx, "PerStartCost"]

    @property
    def perEnd(self):
        return self.params.at[self.__idx, "PerEndCost"]

    @property
    def perMile(self):
        return self.params.at[self.__idx, "PerMileCost"]

    def getAccessDistance(self) -> float:
        return 0.0

    def updateModeBlockedDistance(self):
        for n in self._networks:
            self._L_blocked[n] = n.L_blocked[self.name]

    def addVehicles(self, n):
        self._N_tot += n
        self.allocateVehicles()

    def allocateVehicles(self):
        """even"""
        n_networks = len(self._networks)
        for n in self._networks:
            n.N_eq[self.name] = self._N_tot / n_networks * self.relativeLength
            self._N[n] = self._N_tot / n_networks

    def __str__(self):
        return str([self.name + ': N=' + str(self._N) + ', L_blocked=' + str(self._L_blocked)])

    def getSpeed(self):
        if self.N_queued > 0:
            non_queued_speed = max(self._N, key=self._N.get).getBaseSpeed()
            queued_speed = self.queuing_speed
            return (self._N_tot * non_queued_speed + self.N_queued * queued_speed) / (self._N_tot + self.N_queued)
        else:
            return max(self._N, key=self._N.get).getBaseSpeed()

    def getN(self, network):
        return self._N[network]

    def getNs(self):
        return list(self._N.values())

    def getBlockedDistance(self, network):
        return self._L_blocked[network]

    def getMaxFlow(self):
        return sum([n.getMaxFlow() for n in self._networks])

    def getRateOfVmtPerHour(self):
        return 0.0

    def updateN(self, demand: TravelDemand):
        n_old = self._N_tot
        n_new, n_queued = self.getLittlesLawN(demand.rateOfPmtPerHour, demand.averageDistanceInSystemInMiles)
        self._N_tot = (n_new + n_old) / 2.0
        self.N_queued = n_queued
        self.allocateVehicles()

    def getLittlesLawN(self, rateOfPmtPerHour: float, averageDistanceInSystemInMiles: float):
        # maxPmtPerHour = sum([n.getMaxFlowForMode(self) for n in self._networks]) * 0.3
        speedInMilesPerHour = self.getSpeed() * 2.23694
        # if rateOfPmtPerHour > maxPmtPerHour:
        #     N_normal = maxPmtPerHour / speedInMilesPerHour
        #     self.N_queued = 0.0# (rateOfPmtPerHour - maxPmtPerHour) / 25000
        #     # self.__bad = True
        #     return N_normal + self.N_queued
        # else:
        #     self.__bad = False
        #     self.N_queued = 0.0
        #     return rateOfPmtPerHour / speedInMilesPerHour
        ll_accumulation = rateOfPmtPerHour / speedInMilesPerHour

        maxAccumulation = sum([n.getMaxAccumulationForMode(self) for n in self._networks]) * 1.5

        if ll_accumulation > maxAccumulation * 0.5:
            overPeak = ll_accumulation - maxAccumulation * 0.5
            extra = maxAccumulation * 0.5 * (0.002*overPeak / (1+0.002*overPeak))
            return maxAccumulation * 0.5 + extra, overPeak - extra
        else:
            return ll_accumulation, 0.0


        # if not (speedInMilesPerHour >= 2.0):
        #     self.__bad = True
        #     speedInMilesPerHour = 2.0
        # else:
        #     self.__bad = False
        # averageTimeInSystemInHours = averageDistanceInSystemInMiles / speedInMilesPerHour
        # return rateOfPmtPerHour / averageDistanceInSystemInMiles * averageTimeInSystemInHours

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self._networks]):
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
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super().__init__(networks, modeParams, idx, "walk")
        self.__idx = idx

    @property
    def speedInMetersPerSecond(self):
        return self.params.at[self.__idx, "SpeedInMetersPerSecond"]

    def getSpeed(self):
        return self.speedInMetersPerSecond


class BikeMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super().__init__(networks, modeParams, idx, "bike")
        self.__idx = idx
        self.bikeLanePreference = 2.0

    @property
    def speedInMetersPerSecond(self):
        return self.params.at[self.__idx, "SpeedInMetersPerSecond"]

    def getSpeed(self):
        return self.speedInMetersPerSecond

    def allocateVehicles(self):
        """by length"""
        L_tot = sum([(n.L + n.L * (self.bikeLanePreference - 1) * n.dedicated) for n in self._networks])
        for n in self._networks:
            n.N_eq[self.name] = (n.L + n.L * (
                        self.bikeLanePreference - 1) * n.dedicated) * self._N_tot / L_tot * self.relativeLength
            self._N[n] = (n.L + n.L * (self.bikeLanePreference - 1) * n.dedicated) * self._N_tot / L_tot

    def getPortionDedicated(self) -> float:
        if self._N_tot > 0:
            tot = 0.0
            tot_dedicated = 0.0
            for key, val in self._N.items():
                tot += val
                if key.dedicated:
                    tot_dedicated += val
            return tot_dedicated / tot
        else:
            return 0.0


class RailMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super().__init__(networks, modeParams, idx, "rail")
        self.__idx = idx

    @property
    def routeAveragedSpeed(self):
        return self.params.at[self.__idx, "SpeedInMetersPerSecond"]

    @property
    def vehicleOperatingCostPerHour(self):
        return self.params.at[self.__idx, "VehicleOperatingCostsPerHour"]

    @property
    def fare(self):
        return self.params.at[self.__idx, "PerStartCost"]

    @property
    def headwayInSec(self):
        return self.params.at[self.__idx, "Headway"]

    @property
    def stopSpacingInMeters(self):
        return self.params.at[self.__idx, "StopSpacing"]

    @property
    def portionAreaCovered(self):
        return self.params.at[self.__idx, "CoveragePortion"]

    def getAccessDistance(self) -> float:
        return self.stopSpacingInMeters / 4.0 / self.portionAreaCovered ** 2.0

    def getSpeed(self):
        return self.routeAveragedSpeed

    def getRouteLength(self):
        return sum([n.L for n in self._networks])

    def getOperatorCosts(self) -> float:
        return sum(self.getNs()) * self.vehicleOperatingCostPerHour

    def getOperatorRevenues(self) -> float:
        return self.travelDemand.tripStartRatePerHour * self.fare

    def updateN(self, demand: TravelDemand):
        n_new = self.getRouteLength() / self.routeAveragedSpeed / self.headwayInSec
        self._N_tot = n_new
        self.allocateVehicles()

    def allocateVehicles(self):
        """ Assumes just one subNetwork """
        n = self._networks[0]
        n.N_eq[self.name] = self._N_tot
        self._N[n] = n.N_eq[self.name]


class AutoMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super().__init__(networks, modeParams, idx, "auto")
        self.__idx = idx

    def allocateVehicles(self):
        """for constant car speed"""
        if self._N_tot > 0:
            current_allocation = []
            blocked_lengths = []
            lengths = []
            other_mode_n_eq = []
            jammed = []
            for n in self._networks:
                other_modes = list(n.N_eq.keys())
                if self.name in other_modes:
                    other_modes.remove(self.name)
                current_allocation.append(self._N[n])
                blocked_lengths.append(n.getBlockedDistance())
                lengths.append(n.L)
                other_mode_n_eq.append(sum([n.N_eq[m] for m in other_modes]))
                jammed.append(n.isJammed)
            n_eq_other = sum(other_mode_n_eq)
            L_tot = sum(lengths)
            L_blocked_tot = sum(blocked_lengths)
            density_av = (self._N_tot + n_eq_other) / (L_tot - L_blocked_tot) * self.relativeLength
            if any(jammed):
                print(density_av)
            if self._N_tot > 0:
                n_new = np.nan_to_num(np.array(
                    [density_av * (lengths[i] - blocked_lengths[i]) - other_mode_n_eq[i] for i in range(len(lengths))]))
            else:
                n_new = np.array([0.0] * len(lengths))
            should_be_empty = (n_new < 0) | np.array(jammed)
            to_reallocate = np.sum(n_new[should_be_empty])
            n_new[~should_be_empty] += to_reallocate * n_new[~should_be_empty] / np.sum(n_new[~should_be_empty])
            n_new[should_be_empty] = 0
            if np.any(np.isnan(n_new)):
                print("WTF")
            for ind, n in enumerate(self._networks):
                n.N_eq[self.name] = n_new[ind] * self.relativeLength
                self._N[n] = n_new[ind]
        else:
            for ind, n in enumerate(self._networks):
                n.N_eq[self.name] = 0.0
                self._N[n] = 0.0


class BusMode(Mode):
    def __init__(self, networks, modeParams: pd.DataFrame, idx: str) -> None:
        super().__init__(networks, modeParams, idx, "bus")
        self.__idx = idx
        self.passengerWaitInSec = self.params.at[self.__idx, "PassengerWait"]
        self.passengerWaitInSecDedicated = self.params.at[self.__idx, "PassengerWaitDedicated"]
        self.stopSpacingInMeters = self.params.at[self.__idx, "StopSpacing"]
        self.minStopTimeInSec = self.params.at[self.__idx, "MinStopTime"]
        self.routeAveragedSpeed = super().getSpeed()
        self.routeLength = self.getRouteLength()
        self.addVehicles(self.routeLength / self.routeAveragedSpeed / self.headwayInSec)
        self.routeAveragedSpeed = self.getSpeed()
        self.occupancy = 0.0
        self.updateModeBlockedDistance()


    @property
    def headwayInSec(self):
        return self.params.at[self.__idx, "Headway"]

    # Removing these as properties because the pandas lookup slows things down

    # @property
    # def passengerWaitInSec(self):
    #     return self.params.at[self.__idx, "PassengerWait"]

    # @property
    # def passengerWaitInSecDedicated(self):
    #     return self.params.at[self.__idx, "PassengerWaitDedicated"]

    # @property
    # def stopSpacingInMeters(self):
    #     return self.params.at[self.__idx, "StopSpacing"]

    # @property
    # def minStopTimeInSec(self):
    #     return self.params.at[self.__idx, "MinStopTime"]

    @property
    def fare(self):
        return self.params.at[self.__idx, "PerStartCost"]

    @property
    def vehicleOperatingCostPerHour(self):
        return self.params.at[self.__idx, "VehicleOperatingCostPerHour"]

    @property
    def portionAreaCovered(self):
        return self.params.at[self.__idx, "CoveragePortion"]

    def getAccessDistance(self) -> float:
        return self.stopSpacingInMeters / 4.0 / self.portionAreaCovered ** 2.0

    def updateN(self, demand: TravelDemand):
        n_new = self.routeLength / self.routeAveragedSpeed / self.headwayInSec
        if isnan(n_new):
            print("BAD BUS ACCUMULATION")
        self._N_tot = n_new
        self.allocateVehicles()

    def getRouteLength(self):
        return sum([n.L for n in self._networks])

    def getSubNetworkSpeed(self, car_speed, dedicated):
        # averageStopDuration = self.min_stop_time + self.passenger_wait * (
        #         self.travelDemand.tripStartRate + self.travelDemand.tripEndRate) / (
        #                               self.routeAveragedSpeed / self.stop_spacing * self.N_tot)
        # return car_speed / (1 + averageStopDuration * car_speed / self.stop_spacing)
        if dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        # car_travel_time = self.getRouteLength() / car_speed
        numberOfStops = self.routeLength / self.stopSpacingInMeters
        pass_per_stop = (self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour
                         ) / numberOfStops * self.headwayInSec / 3600.
        stopping_time = numberOfStops * self.minStopTimeInSec
        stopped_time = perPassenger * pass_per_stop + stopping_time
        driving_time = self.routeLength / car_speed
        m = self.routeLength
        s = (stopped_time + self.routeLength / car_speed)
        spd = self.routeLength / (stopped_time + driving_time)
        if np.isnan(spd):
            spd = 0.1
            self.__bad = True
        else:
            self.__bad = False
        return spd

    def getSpeeds(self):
        speeds = []
        for n in self._networks:
            if n.L == 0:
                speeds.append(np.inf)
            else:
                carSpeed = n.getBaseSpeed()
                # if np.isnan(carSpeed):
                #     print("AAAH")
                bus_speed = self.getSubNetworkSpeed(carSpeed, n.dedicated)
                # if np.isnan(bus_speed):
                #     print("AAAH")
                speeds.append(bus_speed)
        return speeds

    def getSpeed(self):
        meters = np.zeros(len(self._networks), dtype=float)
        seconds = np.zeros(len(self._networks), dtype=float)
        # meters = []
        # seconds = []
        for idx, n in enumerate(self._networks):
            if n.L > 0:
                n_bus = self._N[n]
                bus_speed = self.getSubNetworkSpeed(n.getBaseSpeed(), n.dedicated)
                seconds[idx] = n_bus
                meters[idx] = n_bus * bus_speed
        if np.sum(seconds) > 0:
            spd = np.sum(meters) / np.sum(seconds)
            return spd
        else:
            return next(iter(self._networks)).getBaseSpeed()

    def calculateBlockedDistance(self, network) -> float:
        if network.dedicated:
            perPassenger = self.passengerWaitInSecDedicated
        else:
            perPassenger = self.passengerWaitInSec
        if (network.base_speed > 0) & (network.L > 0):
            numberOfStops = self.routeLength / self.stopSpacingInMeters
            meanTimePerStop = (self.minStopTimeInSec + self.headwayInSec * perPassenger * (
                    self.travelDemand.tripStartRatePerHour + self.travelDemand.tripEndRatePerHour) / (
                                       numberOfStops * 3600.0))
            timeToTraverseAtCarSpeed = network.L / network.base_speed
            additionalTimeStopped = network.L / self.stopSpacingInMeters * meanTimePerStop
            portionOfTimeStopped = min([additionalTimeStopped / (additionalTimeStopped + timeToTraverseAtCarSpeed), 1.0])
            out = network.avgLinkLength * portionOfTimeStopped * self._N[network]
            portionOfRouteBlocked = out / self.routeLength
            # busSpeed = self.getSubNetworkSpeed(network.car_speed)
            # out = busSpeed / self.stop_spacing * self.N_tot * self.min_stop_time * network.l
            if out > network.L:
                print("BAD")
        else:
            out = 0
        return out

    def updateModeBlockedDistance(self):
        for n in self._networks:
            L_blocked = self.calculateBlockedDistance(n)
            self._L_blocked[n] = L_blocked
            n.L_blocked[self.name] = L_blocked  # * self.getRouteLength() / n.L

    def allocateVehicles(self):
        "Poisson likelihood"
        speeds = self.getSpeeds()
        times = []
        lengths = []
        for ind, n in enumerate(self._networks):
            spd = speeds[ind]
            times.append(n.L / spd)
            lengths.append(n.L)
        T_tot = sum([lengths[i] / speeds[i] for i in range(len(speeds))])
        for ind, n in enumerate(self._networks):
            if speeds[ind] > 0:
                if isinf(speeds[ind]):
                    n_new = 0.0
                else:
                    n_new = self._N_tot * times[ind] / T_tot
                n.N_eq[self.name] = n_new * self.relativeLength
                self._N[n] = n_new
                if np.isnan(n_new):
                    print('BAD BUS')
            else:
                n_new = self._N_tot / lengths[ind]
                n.N_eq[self.name] = n_new * self.relativeLength
                self._N[n] = n_new
        self.routeAveragedSpeed = self.getSpeed()
        self.occupancy = self.getOccupancy()

        # if np.isnan(self.occupancy):
        #     print("AAAH")

    def getOccupancy(self) -> float:
        return self.travelDemand.averageDistanceInSystemInMiles / (
                self.routeAveragedSpeed * 2.23694) * self.travelDemand.tripStartRatePerHour / self._N_tot

    def getPassengerFlow(self) -> float:
        if np.any([n.isJammed for n in self._networks]):
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
    def __init__(self, data, idx):
        self.data = data
        self.__idx = idx
        self.N_eq = dict()
        self.L_blocked = dict()
        self._modes = dict()
        self.base_speed = self.freeFlowSpeed
        self.dedicated = data.loc[idx, "Dedicated"]
        self.isJammed = False

    @property
    def type(self):
        return self.data.at[self.__idx, "Type"]

    @property
    def avgLinkLength(self):
        return self.data.at[self.__idx, "avgLinkLength"]

    @property
    def freeFlowSpeed(self):
        return self.data.at[self.__idx, "vMax"]

    @property
    def jamDensity(self):
        return self.data.at[self.__idx, "densityMax"]

    @property
    def L(self):
        return self.data.at[self.__idx, "Length"]

    @L.setter
    def L(self, L):
        self.data.at[self.__idx, "Length"] = L

    def __str__(self):
        return str(list(self.N_eq.keys()))

    def resetAll(self):
        self.N_eq = dict()
        self.L_blocked = dict()
        self._modes = dict()
        self.base_speed = self.freeFlowSpeed
        self.isJammed = False

    def resetModes(self):
        for mode in self._modes.values():
            self.N_eq[mode.name] = mode.getN(self) * mode.params.relativeLength
            self.L_blocked[mode.name] = mode.getBlockedDistance(self)
        self.isJammed = False
        self.base_speed = self.freeFlowSpeed
        # mode.reset()

    def getBaseSpeed(self):
        if self.base_speed > 0.01:
            return self.base_speed
        else:
            return 0.01

    def updateBlockedDistance(self):
        for mode in self._modes.values():
            # assert(isinstance(mode, Mode) | issubclass(mode, Mode))
            mode.updateModeBlockedDistance()

    def getMaxFlow(self) -> float:
        if self.type == "Road":
            if self.L > 0:
                L_eq = self.L - self.getBlockedDistance()
                if np.isnan(self.freeFlowSpeed * self.jamDensity / 2.0 * L_eq):
                    print("OH no bad max flow")
                return self.freeFlowSpeed * self.jamDensity / 2.0 * L_eq
            else:
                return 0.0
        else:
            return np.inf

    def getPeakAccumulation(self):
        if self.type == "Road":
            if self.L > 0:
                L_eq = self.L - self.getBlockedDistance()
                return self.jamDensity * L_eq
            else:
                return 0.0
        else:
            return np.inf

    def getMaxFlowForMode(self, mode: Mode) -> float:
        totalMaxFlow = self.getMaxFlow()
        otherModeFlows = 0.0
        for otherModeName in self._modes.keys():
            if otherModeName != mode.name:
                otherModeFlows += self.N_eq[otherModeName] * self.base_speed
        net = totalMaxFlow - otherModeFlows
        if np.isnan(net):
            print("WTF")
        return net / mode.relativeLength /1609.34 * 3600.0 # convert from vehicle meters / sec to vehicle miles / hr

    def getMaxAccumulationForMode(self, mode: Mode) -> float:
        totalAccumulationFlow = self.getPeakAccumulation()
        otherModeAccumulation = 0.0
        for otherModeName in self._modes.keys():
            if otherModeName != mode.name:
                otherModeAccumulation += self.N_eq[otherModeName]
        net = totalAccumulationFlow - otherModeAccumulation
        if np.isnan(net):
            print("WTF")
        return net

    def MFD(self):
        if self.type == "Road":
            if self.L > 0:
                L_eq = self.L - self.getBlockedDistance()
                N_eq = self.getN_eq()
                k = N_eq / L_eq
                if not (L_eq >= 0):
                    print("OH NO, N LESS THAN 0")
                v = self.freeFlowSpeed * (self.jamDensity - k) / self.jamDensity
                v = max((v, 0.01))
            else:
                v = self.freeFlowSpeed
        else:
            v = self.freeFlowSpeed
        self.base_speed = v

    # def MFD(self):
    #     if self.L <= 0:
    #         self.car_speed = 1.0
    #         return
    #     L_eq = self.L - self.getBlockedDistance()
    #     N_eq = self.getN_eq()
    #     maxDensity = 0.25
    #     if (N_eq / L_eq < maxDensity) & (N_eq / L_eq >= 0.0):
    #         noCongestionN = (self.kappa * L_eq * self.w - L_eq * self.Q) / (self.u_f + self.w)
    #         if N_eq <= noCongestionN:
    #             peakFlowSpeed = - L_eq * self.lam / noCongestionN * np.log(
    #                 np.exp(- self.u_f * noCongestionN / (L_eq * self.lam)) +
    #                 np.exp(- self.Q / self.lam) +
    #                 np.exp(-(self.kappa - noCongestionN / L_eq) * self.w / self.lam))
    #             v = self.u_f - N_eq / noCongestionN * (self.u_f - peakFlowSpeed)
    #         else:
    #             v = - L_eq * self.lam / N_eq * np.log(
    #                 np.exp(- self.u_f * N_eq / (L_eq * self.lam)) +
    #                 np.exp(- self.Q / self.lam) +
    #                 np.exp(-(self.kappa - N_eq / L_eq) * self.w / self.lam))
    #         self.car_speed = np.maximum(v, 0.25)
    #     else:
    #         self.car_speed = np.nan

    def containsMode(self, mode: str) -> bool:
        return mode in self._modes.keys()

    def addDensity(self, mode, N_eq):
        self._modes[mode].addVehicles(N_eq)

    def getBlockedDistance(self) -> float:
        if self.L_blocked:
            return sum(list(self.L_blocked.values()))
        else:
            return 0.0

    def getN_eq(self) -> float:
        if self.N_eq:
            return sum(list(self.N_eq.values()))
        else:
            return 0.0

    def addMode(self, mode: Mode):
        self._modes[mode.name] = mode
        self.L_blocked[mode.name] = 0.0
        self.N_eq[mode.name] = 0.0
        return self

    def getModeNames(self) -> list:
        return list(self._modes.keys())

    def getModeValues(self) -> list:
        return list(self._modes.values())


class VehicleMilesTraveledByNetwork:
    def __init__(self, inputData=None):
        if isinstance(inputData, dict):
            self.__data = inputData
        else:
            self.__data = dict()

    def __iadd__(self, other):
        for key, value in other.items():
            if key in self.__data:
                self.__data[key] += value
            else:
                self.__data[key] = value
        return self


class NetworkCollection:
    def __init__(self, networksAndModes=None, modeToModeData=None, microtypeID=None, verbose=False):
        self._networks = list()
        self.modes = dict()
        if isinstance(networksAndModes, Dict) and isinstance(modeToModeData, Dict):
            self.populateNetworksAndModes(networksAndModes, modeToModeData, microtypeID)
        self.demands = TravelDemands([])
        self.verbose = verbose
        self.resetModes()

    def populateNetworksAndModes(self, networksAndModes, modeToModeData, microtypeID):
        modeToNetwork = dict()
        if isinstance(networksAndModes, Dict):
            for (network, modeNames) in networksAndModes.items():
                assert (isinstance(network, Network))
                self._networks.append(network)
                for modeName in modeNames:
                    if modeName in modeToNetwork:
                        modeToNetwork[modeName].append(network)
                    else:
                        modeToNetwork[modeName] = [network]
        else:
            print('Bad NetworkCollection Input')
        for (modeName, networks) in modeToNetwork.items():
            assert (isinstance(modeName, str))
            assert (isinstance(networks, List))
            params = modeToModeData[modeName]
            if modeName == "bus":
                self.modes["bus"] = BusMode(networks, params, microtypeID)
            elif modeName == "auto":
                self.modes["auto"] = AutoMode(networks, params, microtypeID)
            elif modeName == "walk":
                self.modes["walk"] = WalkMode(networks, params, microtypeID)
            elif modeName == "bike":
                self.modes["bike"] = BikeMode(networks, params, microtypeID)
            elif modeName == "rail":
                self.modes["rail"] = RailMode(networks, params, microtypeID)
            else:
                print("BAD!")
                Mode(networks, params, microtypeID, "bad")

    def isJammed(self):
        return np.any([n.isJammed for n in self._networks])

    def resetModes(self):
        # allModes = [n.getModeValues() for n in self._networks]
        # uniqueModes = set([item for sublist in allModes for item in sublist])
        for n in self._networks:
            n.isJammed = False
        self.modes = dict()
        for m in self.modes.values():
            m.updateN(TravelDemand())
            self.modes[m.name] = m
            self.demands[m.name] = m.travelDemand
        # self.updateNetworks()

    def updateModes(self, n: int = 50):
        allModes = [n.getModeValues() for n in self._networks]
        uniqueModes = set([item for sublist in allModes for item in sublist])
        oldSpeeds = self.getModeSpeeds()
        for it in range(n):
            for m in uniqueModes:#self.modes.values():
                m.updateN(self.demands[m.name])
            # self.updateNetworks()
            # vmtByNetwork =
            self.updateMFD()
            if self.verbose:
                print(str(self))
            if np.any([n.isJammed for n in self._networks]):
                break
            newSpeeds = self.getModeSpeeds()
            if np.sum(np.power(oldSpeeds - newSpeeds, 2)) < 0.000001:
                break
            else:
                oldSpeeds = newSpeeds

    def updateNetworks(self):
        for n in self._networks:
            n.resetModes()

    def append(self, network: Network):
        self._networks.append(network)
        self.resetModes()

    def __getitem__(self, item):
        return [n for n in self._networks if item in n.getModeNames()]

    # def addVehicles(self, mode: str, N: float, n=5):
    #     self[mode].addVehicles(N)
    #     self.updateMFD(n)

    def updateMFD(self, iters=1):
        for i in range(iters):
            for n in self._networks:
                n.updateBlockedDistance()
                n.MFD()
                if np.isnan(n.base_speed):
                    n.isJammed = True
            # for m in self.modes.values():
            #     m.updateN(self.demands[m.name])

    def __str__(self):
        return str([n.base_speed for n in self._networks])

    # def addMode(self, networks: list, mode: Mode):
    #     for network in networks:
    #         assert (isinstance(network, Network))
    #         if network not in self._networks:
    #             self.append(network)
    #         self.modes[mode.name] = mode
    #     return self

    def getModeNames(self) -> list:
        return list(self.modes.keys())

    def getModeSpeeds(self) -> np.array:
        return np.array([m.getSpeed() for m in self.modes.values()])

    def getModeOperatingCosts(self):
        out = TotalOperatorCosts()
        for name, mode in self.modes.items():
            out[name] = (mode.getOperatorCosts(), mode.getOperatorRevenues())
        return out

    # def getMaxFlowForMode(self, mode: str) -> float:
    #     return sum([n.getMaxFlowForMode(mode) for n in self._networks])

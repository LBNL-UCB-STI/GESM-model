# -*- coding: utf-8 -*-
import numpy as np


class TravelDemand:
    def __init__(self, data=None, dataToIdx=None):
        if data is None:
            self.__data = np.zeros(4)
        else:
            self.__data = data
        if dataToIdx is None:
            self.__dataToIdx = {'tripStarts': 0, 'tripEnds': 1, 'throughTrips': 2, 'throughDistance': 3}
        else:
            self.__dataToIdx = dataToIdx

    @property
    def tripStartRatePerHour(self):
        return self.__data[self.__dataToIdx['tripStarts']]

    @tripStartRatePerHour.setter
    def tripStartRatePerHour(self, value):
        self.__data[self.__dataToIdx['tripStarts']] = value

    @property
    def tripEndRatePerHour(self):
        return self.__data[self.__dataToIdx['tripEnds']]

    @tripEndRatePerHour.setter
    def tripEndRatePerHour(self, value):
        self.__data[self.__dataToIdx['tripEnds']] = value

    @property
    def rateOfPmtPerHour(self):
        return self.__data[self.__dataToIdx['throughDistance']]

    @rateOfPmtPerHour.setter
    def rateOfPmtPerHour(self, value):
        self.__data[self.__dataToIdx['throughDistance']] = value

    @property
    def throughTrips(self):
        return self.__data[self.__dataToIdx['throughTrips']]

    @throughTrips.setter
    def throughTrips(self, value):
        self.__data[self.__dataToIdx['throughTrips']] = value

    @property
    def averageDistanceInSystemInMiles(self):
        return self.rateOfPmtPerHour / self.throughTrips

    def reset(self):
        self.tripStartRatePerHour = 0.0
        self.tripEndRatePerHour = 0.0
        self.rateOfPmtPerHour = 0.0


class TravelDemands:
    def __init__(self, modes: list):
        self._modes = modes
        self._demands = dict()
        for mode in modes:
            self._demands[mode] = TravelDemand()

    def __setitem__(self, key: str, value: TravelDemand):
        self._demands[key] = value
        if key not in self._modes:
            self._modes.append(key)

    def __getitem__(self, item: str) -> TravelDemand:
        return self._demands[item]

    def setEndRate(self, mode: str, rate: float):
        self._demands[mode].tripEndRatePerHour = rate

    def setStartRate(self, mode: str, rate: float):
        self._demands[mode].tripStartRatePerHour = rate

    def getEndRate(self, mode: str):
        if mode in self._demands:
            return self._demands[mode].tripEndRatePerHour
        else:
            print("Should this really be happening?")
            return 0.0

    def getStartRate(self, mode: str):
        if mode in self._demands:
            return self._demands[mode].tripStartRatePerHour
        else:
            print("Should this really be happening?")
            return 0.0

    def getRateOfPMT(self, mode: str):
        if mode in self._demands:
            return self._demands[mode].rateOfPmtPerHour
        else:
            print("Should this really be happening?")
            return 0.0

    def getAverageDistance(self, mode: str):
        if mode in self._demands:
            return self._demands[mode].averageDistanceInSystemInMiles
        else:
            print("Should this really be happening?")
            return 0.0

    def resetDemand(self):
        for mode in self._modes:
            self._demands[mode].reset()

    # def setSingleDemand(self, mode, demand: float, trip_distance_in_miles: float):
    #     self._demands[mode].tripStartRatePerHour = demand
    #     self._demands[mode].tripEndRatePerHour = demand
    #     self._demands[mode].rateOfPmtPerHour = demand * trip_distance_in_miles
    #     self._demands[mode].averageDistanceInSystemInMiles = trip_distance_in_miles

    # def addSingleDemand(self, mode, demand: float, trip_distance_in_meters: float):
    #     self._demands[mode].tripStartRatePerHour += demand
    #     self._demands[mode].tripEndRatePerHour += demand
    #     self._demands[mode].rateOfPmtPerHour += demand * trip_distance_in_meters / 1609.34
    #     self._demands[mode].averageDistanceInSystemInMiles += trip_distance_in_meters / 1609.34

    def addModeStarts(self, mode: str, demand: float):
        if demand > 0:
            if mode not in self._demands:
                print("AAAAAAH")
            self._demands[mode].tripStartRatePerHour += demand

    def addModeEnds(self, mode: str, demand: float):
        if demand > 0:
            self._demands[mode].tripEndRatePerHour += demand

    def addModeThroughTrips(self, mode: str, demand: float, trip_distance_in_miles: float):
        if demand > 0:
            # current_demand = self._demands[mode].rateOfPmtPerHour
            # current_distance = self._demands[mode].averageDistanceInSystemInMiles
            self._demands[mode].rateOfPmtPerHour += demand * trip_distance_in_miles
            # self._demands[mode].averageDistanceInSystemInMiles = (
            #                                                              current_demand * current_distance + demand * trip_distance_in_meters / 1609.34) / (
            #                                                              current_demand + demand)

    def __str__(self):
        return 'Start Rate: ' + str(
            [mode + ' ' + str(self.getStartRate(mode)) for mode in self._modes]) + '; Dist: ' + str(
            [mode + ' ' + str(self.getAverageDistance(mode)) for mode in self._modes])

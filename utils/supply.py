# -*- coding: utf-8 -*-


class TravelDemand:
    def __init__(self):
        self.tripStartRatePerHour = 0.0
        self.tripEndRatePerHour = 0.0
        self.rateOfPmtPerHour = 0.0
        self.averageDistanceInSystemInMiles = 2.0

    def reset(self):
        self.tripStartRatePerHour = 0.0
        self.tripEndRatePerHour = 0.0
        self.rateOfPmtPerHour = 0.0
        self.averageDistanceInSystemInMiles = 2.0


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
        if item in self._demands:
            return self._demands[item]
        else:
            print("Is this supposed to happen")
            return TravelDemand()

    def setEndRate(self, mode: str, rate: float):
        self._demands[mode].tripEndRatePerHour = rate

    def setStartRate(self, mode: str, rate: float):
        self._demands[mode].tripStartRatePerHour = rate

    def getEndRate(self, mode: str):
        return self._demands[mode].tripEndRatePerHour

    def getStartRate(self, mode: str):
        return self._demands[mode].tripStartRatePerHour

    def getRateOfPMT(self, mode: str):
        return self._demands[mode].rateOfPmtPerHour

    def getAverageDistance(self, mode: str):
        return self._demands[mode].averageDistanceInSystemInMiles

    def resetDemand(self):
        for mode in self._modes:
            self._demands[mode].reset()

    def setSingleDemand(self, mode, demand: float, trip_distance_in_miles: float):
        self._demands[mode].tripStartRatePerHour = demand
        self._demands[mode].tripEndRatePerHour = demand
        self._demands[mode].rateOfPmtPerHour = demand * trip_distance_in_miles
        self._demands[mode].averageDistanceInSystemInMiles = trip_distance_in_miles

    # def addSingleDemand(self, mode, demand: float, trip_distance_in_meters: float):
    #     self._demands[mode].tripStartRatePerHour += demand
    #     self._demands[mode].tripEndRatePerHour += demand
    #     self._demands[mode].rateOfPmtPerHour += demand * trip_distance_in_meters / 1609.34
    #     self._demands[mode].averageDistanceInSystemInMiles += trip_distance_in_meters / 1609.34

    def addModeStarts(self, mode: str, demand: float):
        self._demands[mode].tripStartRatePerHour += demand

    def addModeEnds(self, mode: str, demand: float):
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

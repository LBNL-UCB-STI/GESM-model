import pandas as pd

from .OD import TripCollection, OriginDestination, TripGeneration, DemandIndex, ODindex, ModeSplit, TransitionMatrices
from .choiceCharacteristics import CollectedChoiceCharacteristics, filterAllocation
from .microtype import MicrotypeCollection
from .misc import DistanceBins
from .population import Population


class TotalUserCosts:
    def __init__(self, total=0., totalEqualVOT=0., totalIVT=0., totalOVT=0., demandForTripsPerHour=0.,
                 demandForPMTPerHour=0.):
        self.total = total
        self.totalEqualVOT = totalEqualVOT
        self.totalIVT = totalIVT
        self.totalOVT = totalOVT
        self.demandForTripsPerHour = demandForTripsPerHour
        self.demandForPMTPerHour = demandForPMTPerHour

    def __str__(self):
        return str(self.total) + ' ' + str(self.totalEqualVOT)

    def __mul__(self, other):
        result = TotalUserCosts()
        result.total = self.total * other
        result.totalEqualVOT = self.totalEqualVOT * other
        result.totalIVT = self.totalIVT * other
        result.totalOVT = self.totalOVT * other
        result.demandForTripsPerHour = self.demandForTripsPerHour * other
        result.demandForPMTPerHour = self.demandForPMTPerHour * other
        return result

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        self.total *= other
        self.totalEqualVOT *= other
        self.totalIVT *= other
        self.totalOVT *= other
        self.demandForTripsPerHour *= other
        self.demandForPMTPerHour *= other
        return self

    def copy(self):
        return TotalUserCosts(self.total, self.totalEqualVOT, self.totalIVT, self.totalOVT, self.demandForTripsPerHour,
                              self.demandForPMTPerHour)

    def __add__(self, other):
        out = self.copy()
        out.total += other.total
        out.totalEqualVOT += other.totalEqualVOT
        out.totalIVT += other.totalIVT
        out.totalOVT += other.totalOVT
        out.demandForTripsPerHour += other.demandForTripsPerHour
        out.demandForPMTPerHour += other.demandForPMTPerHour
        return out

    def toDataFrame(self, index=None):
        return pd.DataFrame({"totalCost": self.total, "demandForTripsPerHour": self.demandForTripsPerHour,
                             "inVehicleTime": self.totalIVT, "outOfVehicleTime": self.totalOVT,
                             "demandForPMTPerHour": self.demandForPMTPerHour}, index=index)


class CollectedTotalUserCosts:
    def __init__(self):
        self.__costsByPopulation = dict()
        self.__costsByMode = dict()
        self.__costsByPopulationAndMode = dict()
        self.total = 0.
        self.totalEqualVOT = 0.
        self.demandForTripsPerHour = 0.
        self.demandForPMTPerHour = 0.

    def __setitem__(self, key: (DemandIndex, str), value: TotalUserCosts):
        if key[0] in self.__costsByPopulation:
            self.__costsByPopulation[key[0]] += value
        else:
            self.__costsByPopulation[key[0]] = value
        if key[1] in self.__costsByMode:
            self.__costsByMode[key[1]] += value
        else:
            self.__costsByMode[key[1]] = value
        self.__costsByPopulationAndMode[key] = value
        self.updateTotals(value)

    def __getitem__(self, item) -> TotalUserCosts:
        if isinstance(item, DemandIndex):
            return self.__costsByPopulation[item]
        elif isinstance(item, str):
            return self.__costsByMode[item]
        elif isinstance(item, tuple):
            return self.__costsByPopulationAndMode[item]
        else:
            print("BADDDDD")
            return TotalUserCosts()

    def __iter__(self):
        return iter(self.__costsByPopulationAndMode.items())

    def updateTotals(self, value: TotalUserCosts):
        self.total = sum([c.total for c in self.__costsByPopulation.values()])
        self.totalEqualVOT = sum([c.totalEqualVOT for c in self.__costsByPopulation.values()])
        self.demandForTripsPerHour = sum([c.demandForTripsPerHour for c in self.__costsByPopulation.values()])
        self.demandForPMTPerHour = sum([c.demandForPMTPerHour for c in self.__costsByPopulation.values()])

    def copy(self):
        out = CollectedTotalUserCosts()
        out.total = self.total
        out.totalEqualVOT = self.totalEqualVOT
        out.demandForTripsPerHour = self.demandForTripsPerHour
        out.demandForPMTPerHour = self.demandForPMTPerHour
        out.__costsByPopulation = self.__costsByPopulation.copy()
        out.__costsByMode = self.__costsByMode.copy()
        out.__costsByPopulationAndMode = self.__costsByPopulationAndMode.copy()
        return out

    def __imul__(self, other):
        for di in self.__costsByPopulation.keys():
            self.__costsByPopulation[di] = self.__costsByPopulation[di] * other
        for mode in self.__costsByMode.keys():
            self.__costsByMode[mode] = self.__costsByMode[mode] * other
        for item in self.__costsByPopulationAndMode.keys():
            self.__costsByPopulationAndMode[item] = self.__costsByPopulationAndMode[item] * other
        return self

    def __mul__(self, other):
        out = self.copy()
        for di in out.__costsByPopulation.keys():
            out.__costsByPopulation[di] = out.__costsByPopulation[di] * other
        for mode in out.__costsByMode.keys():
            out.__costsByMode[mode] = out.__costsByMode[mode] * other
        for item in out.__costsByPopulationAndMode.keys():
            out.__costsByPopulationAndMode[item] = out.__costsByPopulationAndMode[item] * other
        return out

    def __rmul__(self, other):
        return self * other

    def __iadd__(self, other):
        for item, cost in other:
            self.__costsByPopulation[item[0]] = self.__costsByPopulation.setdefault(item[0], TotalUserCosts()) + cost
            self.__costsByMode[item[1]] = self.__costsByMode.setdefault(item[1], TotalUserCosts()) + cost
            self.__costsByPopulationAndMode[item] = self.__costsByPopulationAndMode.setdefault(item,
                                                                                               TotalUserCosts()) + cost
        return self

    def toDataFrame(self, index=None) -> pd.DataFrame:
        muc = pd.concat([val.toDataFrame(pd.MultiIndex.from_tuples([key[0].toTupleWith(key[1])])) for key, val in
                         self.__costsByPopulationAndMode.items()])
        muc.index.set_names(['homeMicrotype', 'populationGroupType', 'tripPurpose', 'mode'], inplace=True)
        return muc.swaplevel(0, -1)
        # return pd.concat([val.toDataFrame([key.toIndex()]) for key, val in self.__costs.items()])

    def groupBy(self, vals) -> pd.DataFrame:
        df = self.toDataFrame()
        return df.groupby(level=vals).agg(sum)


class Demand:
    def __init__(self):
        self.__modeSplit = dict()
        self.tripRate = 0.0
        self.demandForPMT = 0.0
        self.pop = 0.0
        self.timePeriodDuration = 0.0
        self.__population = Population()
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()
        self.__transitionMatrices = TransitionMatrices()

    def __setitem__(self, key: (DemandIndex, ODindex), value: ModeSplit):
        self.__modeSplit[key] = value

    def __getitem__(self, item: (DemandIndex, ODindex)) -> ModeSplit:
        if item in self:
            return self.__modeSplit[item]
        else:  # else return empty mode split
            (demandIndex, odi) = item
            print("WTF")

    def __contains__(self, item):
        """ Return true if the correct value"""
        if item in self.__modeSplit:
            return True
        else:
            return False

    def initializeDemand(self, population: Population, originDestination: OriginDestination,
                         tripGeneration: TripGeneration, trips: TripCollection, microtypes: MicrotypeCollection,
                         distanceBins: DistanceBins, transitionMatrices: TransitionMatrices, timePeriodDuration: float,
                         multiplier=1.0):
        self.__population = population
        self.__trips = trips
        self.__distanceBins = distanceBins
        self.__transitionMatrices = transitionMatrices
        self.timePeriodDuration = timePeriodDuration
        newTransitionMatrix = microtypes.emptyTransitionMatrix()
        for demandIndex, utilityParams in population:
            od = originDestination[demandIndex]
            ratePerHourPerCapita = tripGeneration[demandIndex.populationGroupType, demandIndex.tripPurpose] * multiplier
            pop = population.getPopulation(demandIndex.homeMicrotype, demandIndex.populationGroupType)
            for odi, portion in od.items():
                trip = trips[odi]
                common_modes = [microtypes[trip.odIndex.o].mode_names, microtypes[trip.odIndex.d].mode_names]
                # # Now we're switching over to only looking at origin and destination modes
                # common_modes = []
                # for microtypeID, allocation in trip.allocation:
                #     if allocation > 0:
                #         common_modes.append(microtypes[microtypeID].mode_names)
                modes = set.intersection(*common_modes)
                tripRatePerHour = ratePerHourPerCapita * pop * portion
                self.tripRate += tripRatePerHour
                demandForPMT = ratePerHourPerCapita * pop * portion * distanceBins[odi.distBin]
                newTransitionMatrix.addAndMultiply(transitionMatrices[odi],
                                                   tripRatePerHour)  # += transitionMatrices[odi] * tripRatePerHour
                self.demandForPMT += demandForPMT
                self.pop += pop
                modeSplit = dict()
                for mode in modes:
                    if mode == "auto":
                        modeSplit[mode] = 1.0
                    else:
                        modeSplit[mode] = 0.0
                self[demandIndex, odi] = ModeSplit(modeSplit, tripRatePerHour, demandForPMT)
                # dist, alloc = transitionMatrices[odi].getSteadyState()
                # distReal = self.__distanceBins[odi.distBin] * 1609.34
                # allocReal = trip.allocation.sortedValueArray()
                # diff = alloc - allocReal
                # print("WHAT")
        microtypes.transitionMatrix.updateMatrix(newTransitionMatrix * (1.0 / self.tripRate))

    def updateMFD(self, microtypes: MicrotypeCollection, nIters=3):
        for microtypeID, microtype in microtypes:
            microtype.resetDemand()
        newTransitionMatrix = microtypes.emptyTransitionMatrix()
        totalDemandForTrips = 0.0
        for (di, odi), ms in self.__modeSplit.items():
            # assert (isinstance(ms, ModeSplit))
            # assert (isinstance(odi, ODindex))
            # assert (isinstance(di, DemandIndex))
            for mode, split in ms:
                microtypes[odi.o].addModeStarts(mode, ms.demandForTripsPerHour * split)
                microtypes[odi.d].addModeEnds(mode, ms.demandForTripsPerHour * split)
                if mode == "auto":
                    newTransitionMatrix.addAndMultiply(self.__transitionMatrices[odi], ms.demandForTripsPerHour * split)
                    totalDemandForTrips += ms.demandForTripsPerHour * split
                else:
                    newAllocation = filterAllocation(mode, self.__trips[odi].allocation, microtypes)
                    for k, portion in newAllocation.items():
                        microtypes[k].addModeDemandForPMT(mode, ms.demandForTripsPerHour * split,
                                                          self.__distanceBins[odi.distBin])
        microtypes.transitionMatrix = newTransitionMatrix * (1.0 / totalDemandForTrips)

        for it in range(nIters):
            microtypes.transitionMatrixMFD(self.timePeriodDuration)
            for microtypeID, microtype in microtypes:
                microtype.updateNetworkSpeeds(1)

    def updateModeSplit(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
                        originDestination: OriginDestination, oldModeSplit: ModeSplit):
        for demandIndex, utilityParams in self.__population:
            od = originDestination[demandIndex]
            for odi, portion in od.items():
                # dg = self.__population[demandIndex]
                ms = self.__population[demandIndex].updateModeSplit(collectedChoiceCharacteristics[odi])
                self[demandIndex, odi].updateMapping(ms)
                self[demandIndex, odi] *= oldModeSplit
        newModeSplit = self.getTotalModeSplit()
        print(newModeSplit)
        diff = oldModeSplit - newModeSplit
        return diff

    def getTotalModeSplit(self, userClass=None, microtypeID=None, distanceBin=None, otherModeSplit=None) -> ModeSplit:
        demandForTrips = 0
        demandForDistance = 0
        trips = dict()
        for (di, odi), ms in self.__modeSplit.items():
            relevant = ((userClass is None) or (di.populationGroupType == userClass)) & (
                    (microtypeID is None) or (di.homeMicrotype == microtypeID)) & (
                               (distanceBin is None) or (odi.distBin == distanceBin))
            if relevant:
                for mode, split in ms:
                    new_demand = trips.setdefault(mode, 0) + split * ms.demandForTripsPerHour
                    trips[mode] = new_demand
                demandForTrips += ms.demandForTripsPerHour
                demandForDistance += ms.demandForPmtPerHour
        for mode in trips.keys():
            if otherModeSplit is not None:
                trips[mode] /= (demandForTrips * 2.)
                trips[mode] += otherModeSplit[mode] / 2.
            else:
                trips[mode] /= demandForTrips
        return ModeSplit(trips, demandForTrips, demandForDistance)

    def getUserCosts(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
                     originDestination: OriginDestination, modes=None) -> CollectedTotalUserCosts:
        out = CollectedTotalUserCosts()
        for demandIndex, utilityParams in self.__population:
            totalCost = 0.
            totalCostDefault = 0.
            totalDemandForTripsPerHour = 0.
            totalDemandForPMTPerHour = 0.
            totalInVehicle = 0.
            totalOutVehicle = 0.
            od = originDestination[demandIndex]
            demandClass = self.__population[demandIndex]
            for odi, portion in od.items():
                ms = self[(demandIndex, odi)]
                mcc = collectedChoiceCharacteristics[odi]
                for mode in ms.keys():
                    cost, inVehicle, outVehicle, demandForTripsPerHour, distance = demandClass.getCostPerCapita(mcc, ms,
                                                                                                                [mode])
                    if demandForTripsPerHour > 0:
                        out[demandIndex, mode] = TotalUserCosts(cost * demandForTripsPerHour, 0.0,
                                                                inVehicle * demandForTripsPerHour,
                                                                outVehicle * demandForTripsPerHour,
                                                                demandForTripsPerHour,
                                                                demandForTripsPerHour * distance)
                # cost, inVehicle, outVehicle, demandForTripsPerHour, distance = demandClass.getCostPerCapita(mcc, ms,
                #                                                                                             modes)
                # # costDefault = demandClass.getCostPerCapita(mcc, ms, modes) * ms.demandForTripsPerHour  # TODO: Add default
                # totalCost -= cost * demandForTripsPerHour
                # totalInVehicle += inVehicle * demandForTripsPerHour
                # totalOutVehicle += outVehicle * demandForTripsPerHour
                # totalCostDefault -= 0.0
                # totalDemandForTripsPerHour += demandForTripsPerHour
                # totalDemandForPMTPerHour += distance * demandForTripsPerHour
            # out[demandIndex] = TotalUserCosts(totalCost, totalCostDefault, totalInVehicle, totalOutVehicle,
            # totalDemandForTripsPerHour, totalDemandForPMTPerHour)
        return out

    def __str__(self):
        return "Trips: " + str(self.tripRate) + ", PMT: " + str(self.demandForPMT)

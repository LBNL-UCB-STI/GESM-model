from .OD import TripCollection, OriginDestination, TripGeneration, DemandIndex, ODindex, ModeSplit
from .choiceCharacteristics import CollectedChoiceCharacteristics, filterAllocation
from .microtype import MicrotypeCollection
from .misc import DistanceBins
from .population import Population


class TotalUserCosts:
    def __init__(self, total=0., totalEqualVOT=0.):
        self.total = total
        self.totalEqualVOT = totalEqualVOT

    def __str__(self):
        return str(self.total) + ' ' + str(self.totalEqualVOT)

    def __mul__(self, other):
        result = TotalUserCosts()
        result.total = self.total * other
        result.totalEqualVOT = self.totalEqualVOT * other
        return result

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        self.total *= other
        self.totalEqualVOT *= other
        return self

    def copy(self):
        return TotalUserCosts(self.total, self.totalEqualVOT)

    def __add__(self, other):
        out = self.copy()
        out.total += other.total
        out.totalEqualVOT += other.totalEqualVOT
        return out


class CollectedTotalUserCosts:
    def __init__(self):
        self.__costs = dict()
        self.total = 0.
        self.totalEqualVOT = 0.

    def __setitem__(self, key: DemandIndex, value: TotalUserCosts):
        self.__costs[key] = value
        self.updateTotals(value)

    def __getitem__(self, item: DemandIndex) -> TotalUserCosts:
        return self.__costs[item]

    def __iter__(self):
        return iter(self.__costs.items())

    def updateTotals(self, value: TotalUserCosts):
        self.total = sum([c.total for c in self.__costs.values()])
        self.totalEqualVOT = sum([c.totalEqualVOT for c in self.__costs.values()])

    def copy(self):
        out = CollectedTotalUserCosts()
        out.total = self.total
        out.totalEqualVOT = self.totalEqualVOT
        out.__costs = self.__costs.copy()
        return out

    def __imul__(self, other):
        for di in self.__costs.keys():
            self[di] = self[di] * other
        return self

    def __mul__(self, other):
        out = self.copy()
        for di in self.__costs.keys():
            out[di] = out[di] * other
        return out

    def __rmul__(self, other):
        return self * other

    def __iadd__(self, other):
        for di, cost in other:
            self[di] = self.__costs.setdefault(di, TotalUserCosts()) + cost
        return self


class Demand:
    def __init__(self):
        self.__modeSplit = dict()
        self.tripRate = 0.0
        self.demandForPMT = 0.0
        self.__population = Population()
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()

    def __setitem__(self, key: (DemandIndex, ODindex), value: ModeSplit):
        self.__modeSplit[key] = value

    def __getitem__(self, item: (DemandIndex, ODindex)) -> ModeSplit:
        return self.__modeSplit[item]

    def initializeDemand(self, population: Population, originDestination: OriginDestination,
                         tripGeneration: TripGeneration, trips: TripCollection, microtypes: MicrotypeCollection,
                         distanceBins: DistanceBins, multiplier=1.0):
        self.__population = population
        self.__trips = trips
        self.__distanceBins = distanceBins
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
                demandForPMT = ratePerHourPerCapita * pop * distanceBins[odi.distBin]
                self.demandForPMT += demandForPMT
                modeSplit = dict()
                for mode in modes:
                    if mode == "auto":
                        modeSplit[mode] = 1.0
                    else:
                        modeSplit[mode] = 0.0
                self[demandIndex, odi] = ModeSplit(modeSplit, tripRatePerHour, demandForPMT)

    def updateMFD(self, microtypes: MicrotypeCollection, nIters=10):
        for microtypeID, microtype in microtypes:
            microtype.resetDemand()

        for (di, odi), ms in self.__modeSplit.items():
            assert (isinstance(ms, ModeSplit))
            assert (isinstance(odi, ODindex))
            assert (isinstance(di, DemandIndex))
            for mode, split in ms:
                microtypes[odi.o].addModeStarts(mode, ms.demandForTripsPerHour * split)
                microtypes[odi.d].addModeEnds(mode, ms.demandForTripsPerHour * split)
                newAllocation = filterAllocation(mode, self.__trips[odi].allocation, microtypes)
                # through_microtypes = []
                # allocation = []
                # for m, a in self.__trips[odi].allocation:
                #     if (a > 0) & (mode in microtypes[m].mode_names):
                #         through_microtypes.append(m)
                #         allocation.append(a)
                # allocation = np.array(allocation)
                # allocation /= np.sum(allocation)
                for k, portion in newAllocation.items():
                    microtypes[k].addModeDemandForPMT(mode, ms.demandForTripsPerHour * split,
                                                      self.__distanceBins[odi.distBin])

        for microtypeID, microtype in microtypes:
            microtype.updateNetworkSpeeds(5)

    def updateModeSplit(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
                        originDestination: OriginDestination, oldModeSplit: ModeSplit):
        for demandIndex, utilityParams in self.__population:
            od = originDestination[demandIndex]
            for odi, portion in od.items():
                # dg = self.__population[demandIndex]
                ms = self.__population[demandIndex].updateModeSplit(collectedChoiceCharacteristics[odi])
                self[demandIndex, odi].updateMapping(ms)
        newModeSplit = self.getTotalModeSplit()
        diff = oldModeSplit - newModeSplit
        return diff

    def getTotalModeSplit(self) -> ModeSplit:
        demand = 0
        trips = dict()
        for ms in self.__modeSplit.values():
            for mode, split in ms:
                new_demand = trips.setdefault(mode, 0) + split * ms.demandForTripsPerHour
                trips[mode] = new_demand
            demand += ms.demandForTripsPerHour
        for mode in trips.keys():
            trips[mode] /= demand
        return ModeSplit(trips)

    def getUserCosts(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
                     originDestination: OriginDestination, defaultParams=None) -> CollectedTotalUserCosts:
        if defaultParams is None:
            defaultParams = {"ASC": 0.0, "VOT": 15.0, "VOM": 1.0}
        out = CollectedTotalUserCosts()
        for demandIndex, utilityParams in self.__population:
            totalCost = 0.
            totalCostDefault = 0.
            od = originDestination[demandIndex]
            demandClass = self.__population[demandIndex]
            for odi, portion in od.items():
                ms = self[(demandIndex, odi)]
                mcc = collectedChoiceCharacteristics[odi]
                cost = demandClass.getCostPerCapita(mcc, ms) * ms.demandForTripsPerHour
                costDefault = demandClass.getCostPerCapita(mcc, ms, defaultParams) * ms.demandForTripsPerHour
                totalCost += cost
                totalCostDefault += costDefault
            out[demandIndex] = TotalUserCosts(totalCost, totalCostDefault)
        return out

    def __str__(self):
        return "Trips: " + str(self.tripRate) + ", PMT: " + str(self.demandForPMT)

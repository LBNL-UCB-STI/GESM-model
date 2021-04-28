# from .microtype import MicrotypeCollection
import numpy as np

from .misc import DistanceBins


class ChoiceCharacteristics:
    """
    UNITS ARE IN HOURS
    """

    def __init__(self, travel_time=0., cost=0., wait_time=0., access_time=0, protected_distance=0, distance=0,
                 data=None):
        self.__parameterToIdx = {'intercept': 0, 'travel_time': 1, 'cost': 2, 'wait_time': 3, 'access_time': 4,
                                 'protected_distance': 5, 'distance': 6}
        if data is None:
            self.__numpy = np.array([1.0, travel_time, cost, wait_time, access_time, protected_distance, distance],
                                    dtype=float)
        else:
            self.__numpy = data

    @property
    def travel_time(self):
        return self.__numpy[self.__parameterToIdx['travel_time']]

    @travel_time.setter
    def travel_time(self, val: float):
        self.__numpy[self.__parameterToIdx['travel_time']] = val

    @property
    def cost(self):
        return self.__numpy[self.__parameterToIdx['cost']]

    @cost.setter
    def cost(self, val: float):
        self.__numpy[self.__parameterToIdx['cost']] = val

    @property
    def wait_time(self):
        return self.__numpy[self.__parameterToIdx['wait_time']]

    @wait_time.setter
    def wait_time(self, val: float):
        self.__numpy[self.__parameterToIdx['wait_time']] = val

    # @property
    # def wait_time_squared(self):
    #     return self.__numpy[self.__parameterToIdx['wait_time_squared']]
    #
    # @wait_time_squared.setter
    # def wait_time_squared(self, val: float):
    #     self.__numpy[self.__parameterToIdx['wait_time_squared']] = val

    @property
    def access_time(self):
        return self.__numpy[self.__parameterToIdx['access_time']]

    @access_time.setter
    def access_time(self, val: float):
        self.__numpy[self.__parameterToIdx['access_time']] = val

    @property
    def protected_distance(self):
        return self.__numpy[self.__parameterToIdx['protected_distance']]

    @protected_distance.setter
    def protected_distance(self, val: float):
        self.__numpy[self.__parameterToIdx['protected_distance']] = val

    @property
    def distance(self):
        return self.__numpy[self.__parameterToIdx['distance']]

    @distance.setter
    def distance(self, val: float):
        self.__numpy[self.__parameterToIdx['distance']] = val

    @property
    def data(self):
        return self.__numpy

    def __len__(self):
        return len(self.__numpy)

    def idx(self):
        return self.__parameterToIdx

    def __add__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            self.wait_time_squared = self.wait_time ** 2.0
            self.access_time += other.access_time
            self.protected_distance += other.protected_distance
            self.distance += other.distance
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self

    def __iadd__(self, other):
        if isinstance(other, ChoiceCharacteristics):
            self.travel_time += other.travel_time
            self.cost += other.cost
            self.wait_time += other.wait_time
            self.wait_time_squared = self.wait_time ** 2.0
            self.access_time += other.access_time
            self.protected_distance += other.protected_distance
            self.distance += other.distance
            return self
        else:
            print('TOUGH LUCK, BUDDY')
            return self


class ModalChoiceCharacteristics:
    def __init__(self, modeToIdx=None, distanceInMiles=0.0, data=None):
        if modeToIdx is None:
            modeToIdx = dict()
        self.__modalChoiceCharacteristics = dict()
        if data is None:
            self.__numpy = np.zeros((len(modeToIdx), len(ChoiceCharacteristics())))
        else:
            self.__numpy = data
        self.distanceInMiles = distanceInMiles
        self.__modeToIdx = modeToIdx
        for mode, idx in modeToIdx.items():
            self.__modalChoiceCharacteristics[mode] = ChoiceCharacteristics(data=self.__numpy[idx, :])

    def __getitem__(self, item: str) -> ChoiceCharacteristics:
        return self.__modalChoiceCharacteristics[item]  # .setdefault(item, ChoiceCharacteristics())

    def __setitem__(self, key: str, value: ChoiceCharacteristics):
        self.__modalChoiceCharacteristics[key] = value

    def modes(self):
        return list(self.__modalChoiceCharacteristics.keys())

    def reset(self):
        for mode in self.modes():
            self[mode] = ChoiceCharacteristics()

    def __contains__(self, item):
        return item in self.__modalChoiceCharacteristics


class CollectedChoiceCharacteristics:
    def __init__(self, scenarioData, demand):
        self.__scenarioData = scenarioData
        self.__demand = demand
        self.modes = scenarioData.getModes()
        self.__choiceCharacteristics = dict()
        self.__distanceBins = DistanceBins()
        self.__numpy = np.zeros((len(scenarioData.odiToIdx), len(scenarioData.modeToIdx), len(scenarioData.paramToIdx)),
                                dtype=float)

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
    def paramToIdx(self):
        return self.__scenarioData.paramToIdx

    @property
    def numpy(self) -> np.ndarray:
        return self.__numpy

    def __setitem__(self, key, value: ModalChoiceCharacteristics):
        self.__choiceCharacteristics[key] = value

    def __getitem__(self, item) -> ModalChoiceCharacteristics:
        return self.__choiceCharacteristics[item]

    def initializeChoiceCharacteristics(self, trips, microtypes, distanceBins: DistanceBins):
        self.__distanceBins = distanceBins
        self.__numpy[:, :, self.paramToIdx['intercept']] = 1
        for odIndex, trip in trips:
            common_modes = [microtypes[odIndex.o].mode_names, microtypes[odIndex.d].mode_names]
            modes = set.intersection(*common_modes)
            for mode in self.modes:
                if mode not in modes:
                    print("Excluding mode ", mode, "in ODI", odIndex)
                    self.__numpy[self.odiToIdx[odIndex], self.modeToIdx[mode], :] = np.nan
            self[odIndex] = ModalChoiceCharacteristics(self.modeToIdx, distanceBins[odIndex.distBin],
                                                       data=self.__numpy[self.odiToIdx[odIndex], :, :])

    def resetChoiceCharacteristics(self):
        self.__numpy[~np.isnan(self.__numpy)] *= 0.0
        self.__numpy[:, :, self.paramToIdx['intercept']] = 1

    def updateChoiceCharacteristics(self, microtypes, trips):
        self.resetChoiceCharacteristics()
        travelTimeInHours = speedToTravelTime(microtypes.numpySpeed, self.__demand.toThroughDistance)

        for odIndex, trip in trips:
            common_modes = [microtypes[odIndex.o].mode_names, microtypes[odIndex.d].mode_names]
            modes = set.intersection(*common_modes)
            for mode in modes:
                microtypes[odIndex.o].addStartTimeCostWait(mode, self[odIndex][mode])
                microtypes[odIndex.d].addEndTimeCostWait(mode, self[odIndex][mode])
        #         newAllocation = microtypes.filterAllocation(mode, trip.allocation)
        #         for microtypeID, allocation in newAllocation.items():
        #             microtypes[microtypeID].addThroughTimeCostWait(mode,
        #                                                            self.__distanceBins[odIndex.distBin] * allocation,
        #                                                            self[odIndex][mode])
        # otherTravelTime = self.__numpy[:,:, self.paramToIdx['travel_time']]
        # print(travelTimeInHours - otherTravelTime)
        self.__numpy[:, :, self.paramToIdx['travel_time']] = travelTimeInHours


def speedToTravelTime(modeSpeed: np.ndarray, toThroughDistance: np.ndarray) -> np.ndarray:
    modeSecondsPerMeter = (1 / modeSpeed)
    modeSecondsPerMeter[np.isinf(modeSecondsPerMeter)] = 0.0
    assignmentMatrix = np.max(toThroughDistance, axis=0) * 1609.34
    throughTravelTimeInSeconds = assignmentMatrix @ modeSecondsPerMeter
    return throughTravelTimeInSeconds / 3600.0


def filterAllocation(mode: str, inputAllocation, microtypes):
    through_microtypes = []
    allocation = []
    tot = 0.0
    for m, a in inputAllocation:
        if (a > 0) & (mode in microtypes[m].mode_names):
            through_microtypes.append(m)
            allocation.append(a)
            tot += a
    return {m: a / tot for m, a in zip(through_microtypes, allocation)}  # dict(zip(through_microtypes, allocation))

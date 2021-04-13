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
    def __init__(self, modes, distanceInMiles=0.0, data=None):
        self.__modalChoiceCharacteristics = dict()
        if data is None:
            self.__numpy = np.zeros((len(modes), len(ChoiceCharacteristics())))
        else:
            self.__numpy = data
        self.distanceInMiles = distanceInMiles
        self.__modeToIdx = {val: ind for ind, val in enumerate(modes)}
        for ind, mode in enumerate(modes):
            self.__modalChoiceCharacteristics[mode] = ChoiceCharacteristics(data=self.__numpy[ind, :])

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
    def __init__(self, modes: set):
        self.modes = modes
        self.__choiceCharacteristics = dict()
        self.__distanceBins = DistanceBins()
        self.__numpy = np.ndarray(0)
        self.__idx = dict()
        self.__characteristicToIdx = ChoiceCharacteristics().idx()
        self.__modeToIdx = {val: ind for ind, val in enumerate(modes)}
        self.__odiToIdx = dict()

    @property
    def numpy(self) -> np.ndarray:
        return self.__numpy

    def __setitem__(self, key, value: ModalChoiceCharacteristics):
        self.__choiceCharacteristics[key] = value

    def __getitem__(self, item) -> ModalChoiceCharacteristics:
        return self.__choiceCharacteristics[item]

    def initializeChoiceCharacteristics(self, trips,
                                        microtypes, distanceBins: DistanceBins, odiToOdx: dict):
        self.__odiToIdx = odiToOdx
        self.__distanceBins = distanceBins
        self.__numpy = np.zeros((len(trips), len(self.modes), len(self.__characteristicToIdx)), dtype=float)
        self.__numpy[:, :, self.__characteristicToIdx['intercept']] = 1
        for odIndex, trip in trips:
            common_modes = [microtypes[odIndex.o].mode_names, microtypes[odIndex.d].mode_names]
            # common_modes = []
            # for microtypeID, allocation in trip.allocation:
            #     if allocation > 0:
            #         common_modes.append(microtypes[microtypeID].mode_names)
            modes = set.intersection(*common_modes)
            for mode in self.modes:
                if mode not in modes:
                    print("Excluding mode ", mode, "in ODI", odIndex)
                    self.__numpy[odiToOdx[odIndex], self.__modeToIdx[mode], :] = np.nan
            self[odiToOdx[odIndex]] = ModalChoiceCharacteristics(self.modes, distanceBins[odIndex.distBin],
                                                                 data=self.__numpy[odiToOdx[odIndex], :, :])

    def resetChoiceCharacteristics(self):
        self.__numpy[~np.isnan(self.__numpy)] *= 0.0
        self.__numpy[:, :, self.__characteristicToIdx['intercept']] = 1
        # for mcc in self.__choiceCharacteristics.values():
        #     mcc.reset()

    def updateChoiceCharacteristics(self, microtypes, trips):
        self.resetChoiceCharacteristics()
        for odIndex, trip in trips:
            common_modes = [microtypes[odIndex.o].mode_names, microtypes[odIndex.d].mode_names]
            modes = set.intersection(*common_modes)
            for mode in modes:
                microtypes[odIndex.o].addStartTimeCostWait(mode, self[self.__odiToIdx[odIndex]][mode])
                microtypes[odIndex.d].addEndTimeCostWait(mode, self[self.__odiToIdx[odIndex]][mode])
                newAllocation = microtypes.filterAllocation(mode, trip.allocation)
                for microtypeID, allocation in newAllocation.items():
                    microtypes[microtypeID].addThroughTimeCostWait(mode,
                                                                   self.__distanceBins[odIndex.distBin] * allocation,
                                                                   self[self.__odiToIdx[odIndex]][mode])


def filterAllocation(mode: str, inputAllocation, microtypes):
    through_microtypes = []
    allocation = []
    tot = 0.0
    for m, a in inputAllocation:
        if (a > 0) & (mode in microtypes[m].mode_names):
            through_microtypes.append(m)
            allocation.append(a)
            tot += a
    # allocation = np.array(allocation) / tot
    # allocation /= np.sum(allocation)
    return {m: a / tot for m, a in zip(through_microtypes, allocation)}  # dict(zip(through_microtypes, allocation))

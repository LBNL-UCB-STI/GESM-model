import pandas as pd
import matplotlib.pyplot as plt
import os
from model import Model
import pytest
import numpy as np


def test_find_equilibrium():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    a = Model(ROOT_DIR + "/../input-data")
    a.initializeTimePeriod("AM-Peak")
    a.findEquilibrium()
    busLaneDistance = np.arange(0, 3400, 200)
    busSpeed = []
    carSpeedA = []
    carSpeedB = []
    carSpeedC = []
    carSpeedD = []
    busModeShare = []
    carModeShare = []
    userCosts = []
    operatorCosts = []
    ldCosts = []
    allCosts = []
    for dist in busLaneDistance:
        a.scenarioData['subNetworkData'].at[15, "Length"] = dist
        a.scenarioData['subNetworkData'].at[6, "Length"] = 4000 - dist
        a.findEquilibrium()
        ms = a.getModeSplit()
        speeds = pd.DataFrame(a.microtypes.getModeSpeeds())
        busSpeed.append(speeds.loc["bus", "C"])
        carSpeedA.append(speeds.loc["auto", "A"])
        carSpeedB.append(speeds.loc["auto", "B"])
        carSpeedC.append(speeds.loc["auto", "C"])
        carSpeedD.append(speeds.loc["auto", "D"])
        busModeShare.append(ms["bus"])
        carModeShare.append(ms["auto"])
        uc = a.getUserCosts()
        userCosts.append(a.getUserCosts().totalEqualVOT)
        operatorCosts.append(a.getOperatorCosts().total)
        ldCosts.append(0.025*dist)
        allCosts.append(a.getUserCosts().totalEqualVOT + a.getOperatorCosts().total + 0.025*dist)

    plt.scatter(busLaneDistance, busSpeed, marker='<', label="Bus")
    plt.xlabel("Bus Lane Distance In Microtype A")
    plt.ylabel("Bus Speed In Microtype A")


    plt.scatter(busLaneDistance, carSpeedA, label="A")
    plt.scatter(busLaneDistance, carSpeedB, label="B")
    plt.scatter(busLaneDistance, carSpeedC, label="C")
    plt.scatter(busLaneDistance, carSpeedD, label="D")
    plt.legend()
    # plt.xlabel("Bus Lane Distance In Microtype A")
    # plt.ylabel("Bus Speed In Microtype A")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/buslanevsspeed.png")

    plt.clf()

    plt.scatter(busLaneDistance, busModeShare)
    plt.xlabel("Bus Lane Distance In Microtype A")
    plt.ylabel("Bus Mode Share")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/buslanevsmodeshare.png")

    plt.clf()
    plt.scatter(busLaneDistance, allCosts)
    # plt.scatter(busLaneDistance, userCosts, label="user")
    # plt.scatter(busLaneDistance, operatorCosts, label="operator")
    # plt.scatter(busLaneDistance, ldCosts, label="lane dedication")
    plt.xlabel("Bus Lane Distance In Microtype A")
    plt.ylabel("User costs")
    plt.legend()
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/buslanevscost.png")

    assert busSpeed[-1] / busSpeed[0] > 1.005  # bus lanes speed up bus traffic by a real amount

    a = Model(ROOT_DIR + "/../input-data")
    a.initializeTimePeriod("AM-Peak")
    a.findEquilibrium()
    headways = np.arange(60, 900, 60)
    busSpeed = []
    carSpeedA = []
    carSpeedB = []
    carSpeedC = []
    carSpeedD = []
    busModeShare = []
    carModeShare = []
    userCosts = []
    operatorCosts = []
    for hw in headways:
        a.scenarioData["modeData"]["bus"].loc["A", "Headway"] = hw
        a.findEquilibrium()
        ms = a.getModeSplit()
        speeds = pd.DataFrame(a.microtypes.getModeSpeeds())
        busSpeed.append(speeds.loc["bus", "A"])
        carSpeedA.append(speeds.loc["auto", "A"])
        carSpeedB.append(speeds.loc["auto", "B"])
        carSpeedC.append(speeds.loc["auto", "C"])
        carSpeedD.append(speeds.loc["auto", "D"])
        busModeShare.append(ms["bus"])
        carModeShare.append(ms["auto"])
        uc = a.getUserCosts()
        oc = a.getOperatorCosts()
        userCosts.append(a.getUserCosts().totalEqualVOT + oc.total)
        operatorCosts.append(a.getOperatorCosts().total)

    plt.clf()
    plt.scatter(headways, busSpeed, marker='<', label="Bus")
    plt.xlabel("Bus Headway In Microtype A")
    plt.ylabel("Bus Headway In Microtype A")


    plt.scatter(headways, carSpeedA, label="A")
    plt.scatter(headways, carSpeedB, label="B")
    plt.scatter(headways, carSpeedC, label="C")
    plt.scatter(headways, carSpeedD, label="D")
    plt.legend()
    # plt.xlabel("Bus Lane Distance In Microtype A")
    # plt.ylabel("Bus Speed In Microtype A")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/headwayvsspeed.png")

    plt.clf()

    plt.scatter(headways, busModeShare)
    plt.xlabel("Bus Headway In Microtype A")
    plt.ylabel("Bus Mode Share")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/headwayvsmodeshare.png")

    plt.clf()
    plt.scatter(headways, userCosts)
    #plt.scatter(headways, operatorCosts)
    plt.xlabel("Bus Headway In Microtype A")
    plt.ylabel("User costs")
    if not os.path.exists(ROOT_DIR + "/../plots"):
        os.mkdir(ROOT_DIR + "/../plots")
    plt.savefig(ROOT_DIR + "/../plots/headwayvscost.png")
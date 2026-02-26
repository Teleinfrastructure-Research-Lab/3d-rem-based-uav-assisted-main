#%%
import numpy as np
import matplotlib.pyplot as plt
import PathLossCalculationFast_
from ChannelAllocation import ChannelAllocation
from MovementSimulation import MovementSimulation
from PerformanceMetricsCalculation import PerformanceMetricsCalculation
from time import time

import tensorflow as tf
from utils import radiomap_placement_maxmin_rate, exhaustiveSearch, weightedAverage, generate_3d_grid_encoding, estimateEmpiricalVariogram, FindVariogramParameters, KrigingInterpolation


#%%
# Experiments:
# 1. Determine resolution for min UAV channels and min numberOfUEs, and remSamplingRate = 10%
# 2. (Maybe, decide later) Evaluate the remSamplingRate = {1, 5, 10, 15, 20} %
# 3. Evaluate the UAV Location method capacity compared to baselines for the best resolution, 
# fixed UAV channels, and fixed numberOfUEs
#Input Parameters
numberOfMovementIterations = int(100e3)
numberOfUEs = 10 # For Uniform distribution, we set the exact number of UEs
percentageOfChannelsForGBS = 0.5
uavServingLocationCase = 'L_REM' # L_REM L_GP L_MM
updateNumberOfIterations = 30 # REM/User association update period                                                
numberOfEpisodes = 1
numberOfGBSs = 1 # The number of GBSs in the simulation
numberOfUAVs = 1 # The number of UAVs in the simulation
meanNumberOfGBSs = numberOfGBSs # For Poisson or Matern distribtion, we define the mean around which 
                    # the Poisson (Matern) distributed number of GBSs will vary
overallBandwidth = 10e6# # 1 MHz - System BW - check if appropriate
channelBandwidth = 180e3 # 180 kHz per channel  
numberOfChannels = int(overallBandwidth / channelBandwidth) # Number of frequency channels (subbands)
# UE and GBS Tx power
powerLevel = np.power(10, (24 - 30) / 10) # 24 dBm
uplinkPowerLevelOfUE = np.power(10, (23 - 30) / 10) / numberOfChannels # 23 dBm, first converted
                                                # into dBW and then into absolute units and
                                                # divided equally by the number of channels
# The GBS and the UAV have the same transmission power
downlinkPowerLevelOfGBSorUAV = np.power(10, (24 - 30) / 10) / numberOfChannels # 24 dBm, first converted
                                                # into dBW and then into absolute units and
                                                # divided equally by the number of channels
ueNoiseFigure_dB = 9 # according to 3GPP TR 38.901 
ueNoiseFigure = np.power(10, ueNoiseFigure_dB / 10) # 9 dB for UE according to TR 36.828, converted
                                    # into absolute units
apNoiseFigure_dB = 9 # The same as for the UE
apNoiseFigure = np.power(10, apNoiseFigure_dB / 10) 
uavNoiseFigure = apNoiseFigure
uavNoiseFigure_dB = apNoiseFigure_dB

numberPotentialServingLocations = 30 # N_{L}

# REM-related parameters
resolution = 12 #Number of points along each axis of the REM grid
numberOfPlanes = 4
proprietyCoefficient = 1 #Determines the number of cycles for which
                         # the REM will be relevant until it is updated
KriPoints = 5 # Kriging Variogram Points
nu = 1.55 # Smoothing parameter

overallNumber = int((resolution**2)*numberOfPlanes)    
remSamplingRate = updateNumberOfIterations / overallNumber # Number of iterations after 
                                                   # which, the simulation's status is updated
                                                   # 410 is 10% of the resolution = 16 
numberOfMeasurements = updateNumberOfIterations
remUpdateIterations = proprietyCoefficient * updateNumberOfIterations

# Parameters for UE positions generation
LengthOfSimulatedRegion = 150 # Lenght of the simulated region (a square), measured in m
HeightOfSimulatedRegion = 20 # Height of the simulated region, measured in m

# Generate the locations of all possible measurements points for the UAV
# for a specific resolution (i.e. points per axis)
heightMin = 0.1 # in m
heightMax = HeightOfSimulatedRegion - 9 # Max height of the UE mobility (terrestrial Het-Net) region
# Generate the 3D grid ofpositions of the UAV operational region
xCoordinatesSpan = np.round(np.linspace(-LengthOfSimulatedRegion/2, LengthOfSimulatedRegion/2, resolution), 2)
yCoordinatesSpan = np.round(np.linspace(-LengthOfSimulatedRegion/2, LengthOfSimulatedRegion/2, resolution), 2)
zCoordinatesSpan = np.round(np.linspace(heightMax, HeightOfSimulatedRegion, resolution), 2)
allPossibleMeasurementPoints = generate_3d_grid_encoding(LengthOfSimulatedRegion, HeightOfSimulatedRegion, 
                                                         numberOfPlanes, resolution, heightMax)
allPossibleMeasurementPoints = np.reshape(allPossibleMeasurementPoints,
                                    (overallNumber, allPossibleMeasurementPoints.shape[2]))
allPossibleMeasurementPoints = np.round(allPossibleMeasurementPoints, 2)

device = "/CPU:0"
if tf.test.is_gpu_available():
    device = "/GPU:0"

distribution = 'Matern' # Desired distribution for the UEs in the grid
                        # Options:
                        # 'Poisson'
                        # 'Uniform'
                        # 'Matern' (for Mattern hard core process which considers the minimum distance
                        # between the points generated by a Poisson process)

meanNumberOfUEs = numberOfUEs # or Poisson or Matern distribtion, we define the mean around which  
                    # the Poisson (Matern) distributed number of UEs will vary

uavPathlossCase = 'U_2RM'#'U_AB' #'U_2R'#'U_DS'#


# Generate the number of UEs and GBSs (for Poisson distributed points, only) 
# and their locations for this movementIterationIdx
# Determine the initial user positions and the GBSs positions 
seed = 1052
# Generate the GBS
xPositionsGBS = np.array([0])
yPositionsGBS = np.array([0])
zPositionsGBS = np.array([10])

# Initial placements of the GBS and UEs
if(distribution == 'Matern'):
    # Generate the GBSs - This is not needed here - the GBS is only one and its position is fixed.
    # The density describes the number of GBSs per square area (i.e. the squared length of the simulation area)
    # densityOfGBSs = meanNumberOfGBSs / np.power(LengthOfSimulatedRegion, 2)  
    # apLocations = PathLossCalculationFast_.DefineLocationsOfGBSsAndUEsIn3D(LengthOfSimulatedRegion, 
    #                                                                  HeightOfSimulatedRegion-5,
    #                                                                  seed + 10) # The movementIterationIdx's
    #                                                                             # number is used as
    #                                                                             # seed for the GBS
    #                                                                             # locations generators. An arbitrary number
    #                                                                             # is added to avoid generating the GBSs
    #                                                                             # on the same locations as the UEs.
    
    # Determine the number of GBSs according to the Matern distribution, as well as their locations
    # Remove GBSs which are within 30 m from each other
    # apLocations.MaternDistributedPoints(densityOfGBSs, 30)
    # numberOfGBSs = apLocations.NumberOfThinnedPoints
    # xPositionsGBS = apLocations.xPositions
    # yPositionsGBS = apLocations.yPositions
    # zPositionsGBS = np.array([1.5])

    # Generate the UEs
    # The density describes the number of UEs per square area (i.e. the squared length of the simulation area)
    densityOfUEs = meanNumberOfUEs / np.power(LengthOfSimulatedRegion, 2)  
    ueLocations = PathLossCalculationFast_.DefineLocationsOfGBSsAndUEsIn3D(LengthOfSimulatedRegion, 
                                                                     HeightOfSimulatedRegion=2,
                                                                     seed=seed + 123) # The movementIterationIdx's
                                                                                # number is used as
                                                                                # seed for the UE
                                                                                # locations generators.
    
    # Determine the number of UEs according to the Poisson distribution, as well as their locations
    ueLocations.PoissonDistributedPoints(densityOfUEs)
    # Utilize the thinning process to remove UEs which are within 0.3 m from any GBS
    ueLocations.MaternDistributedPoints(densityOfUEs, 0.3, xPositionsGBS, yPositionsGBS, zPositionsGBS)
    numberOfUEs = ueLocations.NumberOfThinnedPoints
    xStartingPositionsUE = ueLocations.xPositions 
    yStartingPositionsUE = ueLocations.yPositions
    zStartingPositionsUE = ueLocations.zPositions   

# Scenario specific variables (for debug mostly)
numberOfUpdates = int(numberOfMovementIterations/updateNumberOfIterations) + 1
numberOfREMUpdates = int(numberOfMovementIterations/(updateNumberOfIterations*proprietyCoefficient)) + 1
meanTotalCapacity = np.zeros((numberOfEpisodes, numberOfUpdates))
meanTotalInterference = np.zeros((numberOfEpisodes, numberOfUpdates))
currentUAVMeasurements = np.zeros((numberOfUEs, numberOfMeasurements))
currentUAVMeasurementPositions = np.zeros((numberOfMeasurements, 3))
allUAVMeasurements = np.zeros((numberOfUEs, numberOfREMUpdates, numberOfMeasurements))
allUAVMeasurementPositions = np.zeros((numberOfUEs, numberOfREMUpdates, numberOfMeasurements, 3))
allReconstructedREMs = np.zeros((numberOfREMUpdates, overallNumber))
allReconstructedREMsEachUE = np.zeros((numberOfUEs, numberOfREMUpdates, overallNumber))
numberOfUEsAssociatedToUAV = np.zeros((numberOfEpisodes, numberOfUpdates))
activeABS = np.zeros(numberOfMovementIterations)
idsOfABSPositions = []
idsOfUEsAssociatedToABS = []
listOfUpdateIds = []

# Generate all UE positions though Random Walk for the numberOfMovementIterations 
# (not needed if loaded from file)
# Generate the Moving UEs 
# xPositionsMovingUEs = []
# yPositionsMovingUEs = []
# zPositionsMovingUEs = []
# for currentUEIdx in range(xStartingPositionsUE.size):
#     movementSimulation = MovementSimulation()
#     movementSimulation.RandomWalkGeneratedPoints(currentUEIdx * 21,
#                                 xStartingPositionsUE[currentUEIdx],
#                                 yStartingPositionsUE[currentUEIdx],
#                                 zStartingPositionsUE[currentUEIdx],
#                                 heightMin, heightMax, LengthOfSimulatedRegion,
#                                 numberOfMovementIterations)
#     xPositionsMovingUEs.append(movementSimulation.xPositionsMovingUEs)
#     yPositionsMovingUEs.append(movementSimulation.yPositionsMovingUEs)
#     zPositionsMovingUEs.append(movementSimulation.zPositionsMovingUEs)

# xPositionsMovingUEs = np.array(xPositionsMovingUEs).T
# yPositionsMovingUEs = np.array(yPositionsMovingUEs).T
# zPositionsMovingUEs = np.array(zPositionsMovingUEs).T

# np.save('xPositionsMovingUEs_250k.npy', xPositionsMovingUEs)
# np.save('yPositionsMovingUEs_250k.npy', yPositionsMovingUEs)
# np.save('zPositionsMovingUEs_250k.npy', zPositionsMovingUEs)

xPositionsMovingUEs = np.load('xPositionsMovingUEs_new_100k.npy')
yPositionsMovingUEs = np.load('yPositionsMovingUEs_new_100k.npy')
zPositionsMovingUEs = np.load('zPositionsMovingUEs_new_100k.npy')

# positionsMovingUEs = np.array([
#     xPositionsMovingUEs.T, yPositionsMovingUEs.T, zPositionsMovingUEs.T
# ]).T

defaultScenario = False
plotting = False

# %%
# Smulation
startingTime = time()
for episodeIdx in range(0, numberOfEpisodes):    
    apOnlyCapacityOverall = []
    uavCapacityOverall = []    
    apOnlyInterferenceOverall = []
    uavInterferenceOverall = []        
    totalCapacityPerUpdate = 0
    totalInterferencePerUpdate = 0
    simulationUpdateIdx = 0
    counterOfUpdates = 0
    pathLossBetweenUEandGBS_or_UAV = np.zeros(numberOfUEs)
    uavMeasurementsIdx = 0
    uavAssociationsDone = False
    absAndGBSSessionCounter = 0
    lastREMIdx = 0
    idxOfRemUpdates = 0
    idsOfGBSonly = np.zeros(0, dtype=int)
    idsOfGBS_UAV = np.zeros(0, dtype=int)    
    # Received power on the allocated channels for DL (at UE) 
    receivedUsefulSignalPowerAtUE = np.zeros(numberOfUEs)
    receivedUsefulSignalPowerAtABS = np.zeros(numberOfUEs)
    # Received interference power on the allocated channels for DL (at UE) 
    receivedPowerSumForInterferingGBSsAtUE = np.zeros(numberOfUEs)
    currentNumberOfAllocatedChannels = np.zeros(numberOfUEs, dtype=int)
    performanceMetrics_ = PerformanceMetricsCalculation(channelBandwidth)
    for movementIterationIdx in range(0, numberOfMovementIterations): 
        if( simulationUpdateIdx == 0 ):   
            # Calculates the PathLoss for each UE to the GBS. The lowest loss
            # will determine which GBS the UE will associate with (association scheme 1).
            # Once associated with a UE, change of this association will not be performed.               
            seed = len(xPositionsMovingUEs) + movementIterationIdx + episodeIdx + 25
            pathLossObject = PathLossCalculationFast_.PathLossCalculation(seed)
            pathLossObject.traditionalModel(xPositionsGBS,
                                            yPositionsGBS,
                                            zPositionsGBS,
                                            xPositionsMovingUEs[movementIterationIdx],
                                            yPositionsMovingUEs[movementIterationIdx],
                                            zPositionsMovingUEs[movementIterationIdx])
            pathLossForUE = 10 * np.log10( pathLossObject.pathLoss ) # In dB
            pathLossForUE = pathLossForUE[0, :]

            # Associate the UEs to the GBS from that with lowest 
            # path loss to that with highest
            listsOfIdsOfUEstoAssociate = []
            idsOfUEstoAssociate = np.argsort(pathLossForUE)
            idsOfGBSs = np.arange(0, numberOfGBSs+numberOfUAVs) # including 1 UAV
            # **For multiple GBSs, you will need to make a modification here**
            # **by defining how many UEs can be associated to an GBS**

            # Convert the list to that of GBSs with their associated UE
            listOfAssociatedGBSsIds = np.zeros(idsOfUEstoAssociate.shape[0], dtype=int) 
            listOfAssociatedUEsIds = idsOfUEstoAssociate
            listsOfAssociatedUEsIds = []
            listsOfAssociatedGBSsIds = []
            listsOfAssociatedUEsIds.append(listOfAssociatedUEsIds)
            listsOfAssociatedGBSsIds.append(listOfAssociatedGBSsIds)        

            # Allocations for GBS-UE links
            # Allocate random channels for the GBS-UE pairs (connections). 
            # All channels are available for the single GBS case, while those which
            # have low SINR will be used by the UAV, rather than the GBS
            apListIdx = 0 # Denotes the GBS

            # The GBS and UAV divide from the same set of channels 
            listOfUnoccupiedChannelsIDs = np.arange(0, int(numberOfChannels*percentageOfChannelsForGBS))
            # Perform allocation of the defined number of channels for all connection
            performChannelAllocation = ChannelAllocation(movementIterationIdx)
            performChannelAllocation.DetermineAllocatedChannelIDs(listOfUnoccupiedChannelsIDs,
                                                            listsOfAssociatedUEsIds[apListIdx], 
                                                            numberOfChannels)
            # List which holds all connection IDs and their respective allocated channels
            # in the form [connectionID, [listOfAllocatedChannelsForThisConnection]]
            listOfChannelAllocationsUE_GBS = performChannelAllocation.listOfChannelAllocations      
            listOfChannelAllocations_ = np.array(performChannelAllocation.debugArray)

            listsOfChannelAllocationsAllUEs = []
            listsOfChannelAllocationsAllUEs.append(listOfChannelAllocationsUE_GBS)

            ########################################################################

            seed = len(xPositionsMovingUEs) + movementIterationIdx + episodeIdx + 123
            # Set the UAV location for the measurement in the current iteration
            # by using information from the previous reconstructed REMs,  
            # i.e. a number of locations (f.ex. "numberOfMeasurements") with lowest RSRP;
            # or use random locations for the 1st period of measurements
            
            # DEFAULT simulation (or the 1st numberOfMeasurements iterations) - 
            # the locations of UAV measurements are chosen at random
            # measurementLocationsIndices = np.zeros((numberOfMeasurements), dtype=int)
            if( uavMeasurementsIdx == 0 and ((defaultScenario == False and counterOfUpdates < 1
                ) or defaultScenario == True or uavServingLocationCase != 'L_REM') ):
                measurementLocationsIndices = np.random.choice(np.arange(0, overallNumber),
                                                size=numberOfMeasurements, replace=False)                           
            # In subsequent iterations - Set the UAV location for the measurement in 
            # the current iteration by using information from the previous reconstructed REM     
            elif( uavMeasurementsIdx == 0 and (allReconstructedREMs.shape[0] > lastREMIdx )
                  and uavServingLocationCase == 'L_REM'):
                uavAssociationsDone = False
                # Choose the locations with numberOfMeasurements lowest UL RSSs from all UEs as 
                # they will be of most interest
                # The lastREMIdx determines the last reconstructed REM
                measurementLocationsIndices = np.argsort(allReconstructedREMs[lastREMIdx - 1,
                                                                    :])[0:numberOfMeasurements]

                lastREMIdx += 1 

            # Making the RSS measurements for each UE
            # Measure RSS from all UEs to the ABS
            pathLossForUE_ABS = np.zeros(numberOfUEs)
            for ueIdx in range(0, numberOfUEs):    
                currentUAVMeasurementPositions[uavMeasurementsIdx, :] = allPossibleMeasurementPoints[
                                                measurementLocationsIndices[uavMeasurementsIdx]]
                # Perform UAV measurements at the selected positions
                # Calculates the PathLoss for each UE to the UAV. 
                pathLossObject = PathLossCalculationFast_.PathLossCalculation(seed)
                # The UAV-UE PL model are used here
                pathLossObject.pathLossUAV_GN(  currentUAVMeasurementPositions[uavMeasurementsIdx, 0],
                                                currentUAVMeasurementPositions[uavMeasurementsIdx, 1],
                                                currentUAVMeasurementPositions[uavMeasurementsIdx, 2],
                                                xPositionsMovingUEs[movementIterationIdx, ueIdx],
                                                yPositionsMovingUEs[movementIterationIdx, ueIdx],
                                                zPositionsMovingUEs[movementIterationIdx, ueIdx], 
                                                uavPathlossCase )
                pathLossForUE_ABS[ueIdx] = pathLossObject.pathLoss[0, :][0]
    
            uavMeasurementsIdx += 1 
            
            # Collect the RSRP measurements at the UAV, for the REM reconstruction
            # Allocate the channels for each UE
            currentChannelsIDs = []
            for currentConnectionIDIdx in range(0, len(listsOfChannelAllocationsAllUEs[0])):
                # Calculate the Performance Metrics for the DL connection
                # The channel IDs which are allocated to the current connection
                currentChannelsIDsForGBS = listsOfChannelAllocationsAllUEs[0][currentConnectionIDIdx][1]
                currentChannelsIDs.append(currentChannelsIDsForGBS)
                currentNumberOfAllocatedChannels[listsOfChannelAllocationsAllUEs[0][
                                            currentConnectionIDIdx][0]] = len(currentChannelsIDsForGBS) 
            # Calculate the power measured by the UAV for each GBS 
            # for the channels allocated by the GBS for each GBS-UE connection
            for apListIdx in range(0, len(listsOfAssociatedGBSsIds)):
                for apIdx in range(0, len(listsOfAssociatedGBSsIds[apListIdx])):
                    ueIdx = listsOfAssociatedUEsIds[apListIdx][apIdx]
                    rng = np.random.default_rng((movementIterationIdx*ueIdx) + ueIdx)
                    # # This is the measurement noise
                    receivedUsefulSignalPowerAtABS[ueIdx] = currentNumberOfAllocatedChannels[ueIdx
                    ] * uplinkPowerLevelOfUE * pathLossForUE_ABS[ueIdx] #* np.power(10,
                    #(rng.normal(0, 9, 1)[0])/10) #0, 9
            receivedUsefulSignalPowerAtABS = receivedUsefulSignalPowerAtABS[np.argsort(listsOfAssociatedUEsIds)[0]]

            # The measured power at the current point for the utilized bandwidth 
            # is obtained by the mean power over all UEs
            currentMeasurementIteration = movementIterationIdx % updateNumberOfIterations
            currentUAVMeasurements[:, currentMeasurementIteration] = receivedUsefulSignalPowerAtABS
            ########################################################################       
            
            # Reconstruct REM after the measurements are gathered 
            if( movementIterationIdx % updateNumberOfIterations == updateNumberOfIterations - 1 ):
                simulationUpdateIdx = 1
                currentSeed = movementIterationIdx + episodeIdx + 123
                np.random.seed(currentSeed)
                uavMeasurementsIdx = 0

            
            
                # Reconstruct the 3D REM for each UE
                startingTime_ = time()
                for ueIdx in range(0, numberOfUEs):
                    
                    
                    # For Kriging:
                    knownPoints = allPossibleMeasurementPoints[measurementLocationsIndices, :] 
                    unknownIds = np.array([item for item in np.arange(0, allPossibleMeasurementPoints.shape[0
                                                     ]) if item not in measurementLocationsIndices])
                    unknownPoints = allPossibleMeasurementPoints[unknownIds, :]

                    measurementsInDb = 10*np.log10(currentUAVMeasurements[ueIdx, :])
                    normalizedMeasurements = 1/np.abs(measurementsInDb/measurementsInDb.max()) 
                

                    krigingReconstructedREM = np.zeros(allPossibleMeasurementPoints.shape[0])
                    variogramDistances, variogramPowers, indices, pointEstimates,_,_ = estimateEmpiricalVariogram(
                        KriPoints,
                                                                                knownPoints,
                                                                                normalizedMeasurements, 
                                                                                unknownPoints[:,0],
                                                                                unknownPoints[:,1],
                                                                                unknownPoints[:, 2])
                    nugget_, sill_, range_ = FindVariogramParameters(variogramPowers, variogramDistances)
                    # Estimate the unknown points' power through Kriging
                    estimatedPower, std, _, _ = KrigingInterpolation(
                        normalizedMeasurements,
                                                                        variogramPowers,
                                                                        nugget_, sill_, range_, variogramDistances,
                                                                        knownPoints, 
                                                                        indices, pointEstimates, nu, 123)
                    reconstructedREM = np.zeros(allPossibleMeasurementPoints.shape[0])


                    if(defaultScenario == True or uavServingLocationCase != 'L_REM'):   
                        # In the DEFAULT simulation scenario, 
                        # the REM is represented by the measurements themselves                                               
                        reconstructedREM = currentUAVMeasurements[ueIdx, :]
                    else:

                        # for Kriging method
                        reconstructedREM[unknownIds] = estimatedPower
                        reconstructedREM[measurementLocationsIndices] = normalizedMeasurements#currentUAVMeasurements[ueIdx, :]
                        estimationTime = time() - startingTime_
                        # print("single rem reconstructed in ", estimationTime)
                        allReconstructedREMsEachUE[ueIdx, idxOfRemUpdates, :] = reconstructedREM
                        # The output is in dB so it is converted back to non-logarithmic numbers
                # Sum all REMs
                allReconstructedREMs[idxOfRemUpdates, :] = np.sum(allReconstructedREMsEachUE[:, 
                                idxOfRemUpdates, :], axis=0)
                idxOfRemUpdates += 1                

                print("ABS Location determined " + str(movementIterationIdx) +
                    ' overall time in minutes: ' +  str((time() - startingTime)/60))                
                if(uavServingLocationCase == 'L_REM'): 
                    # # REM-based UAV Location
                    potentialServingLocationIds = np.argsort(allReconstructedREMs[idxOfRemUpdates, :])[:numberPotentialServingLocations]

                    # # Calculate the downlink RSS and throughput for all UEs
                    # # Measure RSS from the ABS to all UEs
                    receivedUsefulSignalPowerAtUEs = np.zeros(numberOfUEs)
                    sumThroughputs = np.zeros(potentialServingLocationIds.shape[0])
                    for currentABSLocationIdx in range(0, potentialServingLocationIds.shape[0]):
                        pathLossForUE_ABS = np.zeros(numberOfUEs)
                        for ueIdx in range(0, numberOfUEs):    
                            currentUAVPosition = allPossibleMeasurementPoints[
                                                            potentialServingLocationIds[currentABSLocationIdx]]
                            # Perform UAV measurements at the selected positions
                            # Calculates the PathLoss for each UE to the UAV. We need the PL from the UEs to
                            # the UAV for each UE, so we can multiply it by the GBS's transmit power
                            # and aggregate all of them to obtain the cumulative RSS of the GBS over all UEs
                            # at each measurement position. BUT this is not what happens in the real world.
                            # So, for it to make sense we need to measure from the GBS as it is the actual transmitter.
                            pathLossObject = PathLossCalculationFast_.PathLossCalculation(seed)
                            # The UAV-UE PL model are used here
                            pathLossObject.pathLossUAV_GN(  currentUAVPosition[0],
                                                            currentUAVPosition[1],
                                                            currentUAVPosition[2],
                                                            xPositionsMovingUEs[movementIterationIdx, ueIdx],
                                                            yPositionsMovingUEs[movementIterationIdx, ueIdx],
                                                            zPositionsMovingUEs[movementIterationIdx, ueIdx], 
                                                            uavPathlossCase )
                            pathLossForUE_ABS[ueIdx] = pathLossObject.pathLoss[0, :][0]
                            
                        # Collect the RSRP measurements at the UEs
                        # Allocate the channels for each UE
                        currentChannelsIDs = []
                        for currentConnectionIDIdx in range(0, len(listsOfChannelAllocationsAllUEs[0])):
                            # Calculate the Performance Metrics for the DL connection
                            # The channel IDs which are allocated to the current connection
                            currentChannelsIDsForGBS = listsOfChannelAllocationsAllUEs[0][currentConnectionIDIdx][1]
                            currentChannelsIDs.append(currentChannelsIDsForGBS)
                            currentNumberOfAllocatedChannels[listsOfChannelAllocationsAllUEs[0][
                                                        currentConnectionIDIdx][0]] = len(currentChannelsIDsForGBS) 
                        # Calculate the power measured by all UEs 
                        # for their corresponding channels ABS-UE link
                        for apListIdx in range(0, len(listsOfAssociatedGBSsIds)):
                            for apIdx in range(0, len(listsOfAssociatedGBSsIds[apListIdx])):
                                ueIdx = listsOfAssociatedUEsIds[apListIdx][apIdx]
                                # rng = np.random.default_rng((movementIterationIdx*ueIdx) + ueIdx)
                                # # This is the measurement noise
                                receivedUsefulSignalPowerAtUEs[ueIdx] = currentNumberOfAllocatedChannels[ueIdx
                                ] * downlinkPowerLevelOfGBSorUAV * pathLossForUE_ABS[ueIdx] #* np.power(10,
                                #(rng.normal(0, 9, 1)[0])/10) #0, 9

                        # Initialize the performanceMetrics object for the metrics' calculation
                        performanceMetrics = PerformanceMetricsCalculation(channelBandwidth)
                        # Calculate the Performance Metrics for DL (i.e. UEs' point of view)
                        performanceMetrics.CalculateSINR(receivedUsefulSignalPowerAtUEs,#[listsOfSortedIDsForUEs], #[listOfAssociatedUEsIds],
                        currentNumberOfAllocatedChannels,#[listsOfSortedIDsForUEs], #[listOfAssociatedUEsIds],
                        receivedPowerSumForInterferingGBSsAtUE,#[listsOfSortedIDsForUEs],#[listOfAssociatedUEsIds],
                        noiseFigure = ueNoiseFigure )
                        SINR_UE = performanceMetrics.SinrInDb     
                        SINR = performanceMetrics.SINR
                                            # currentNumberOfAllocatedChannels[listOfAssociatedUEsIds]
                        capacityUEs = overallNumberOfAllocatedChannels * channelBandwidth * np.log2(1 + SINR) / 1e6 # for Mbps
                        sumThroughputs[currentABSLocationIdx] += np.sum(capacityUEs)
                    uavServingLocationIdx = potentialServingLocationIds[np.argmax(sumThroughputs)]


                elif(uavServingLocationCase == 'L_R'):
                    # Random Location for each association
                    uavServingLocationIdx = np.random.choice(np.arange(0, reconstructedREM.shape[0]))

                # Find the nearest measurementLocationIdx (of the measurementLocationsIndices)
                # to the idx of the lowest DL GBS_RSS, among the allPossibleMeasurementPoints
                uavServingLocation = allPossibleMeasurementPoints[uavServingLocationIdx]
                pathLossObject = PathLossCalculationFast_.PathLossCalculation(currentSeed)
                
                # Record the idx of the current ABS serving position
                idsOfABSPositions.append(uavServingLocationIdx)             
            
                idsOfUEstoAssociateForUAV = np.where(capacityCurrentUE <= 5)[0] # in Mbps
                if(len(idsOfUEstoAssociateForUAV) == 0):
                    # idsOfPotentialUEs = np.array([np.argmin(receivedUsefulSignalPowerUEs_GBS)])
                    simulationUpdateIdx = 0
                    print('len(idsOfPotentialUEs) == 0, agrmin')
                else:
                    activeABS[movementIterationIdx] = 1
                    
                # all UEs with thoughput under the threshold are associated
                # DEFAULT - Random selection of UEs to associate with the UAV for this cycle
                if(defaultScenario == True):
                    # Choose random number of UEs and random UE indices for testing
                    numberOfUEsForUAV = np.random.choice( np.arange(1, numberOfUEs-1) )       
                    # Get the indices of the UEs that are to be associated to the UAV for testing
                    idsOfUEstoAssociateForUAV = np.random.choice(listOfAssociatedUEsIds,
                                                    numberOfUEsForUAV, replace = False )
                # Debug
                if (plotting == True):
                    plt.figure()
                    arr = 10*np.log10(currentUAVMeasurements).reshape(
                        int(np.sqrt(updateNumberOfIterations)), int(np.sqrt(updateNumberOfIterations)))
                    plt.imshow(arr, cmap='hot', interpolation='nearest')
                    plt.colorbar()
                    plt.show()              

        # Perform associations of the selected UEs (i.e. idsOfUEstoAssociateForUAV) to the UAV
        if( (movementIterationIdx % updateNumberOfIterations == 0 and
            1 <= simulationUpdateIdx <= proprietyCoefficient+1 and movementIterationIdx > 0 
            and len(idsOfUEstoAssociateForUAV) > 0) ):

            # Set the UAV at the location with lowest RSRP determined from the REM, i.e. uavServingLocation
            # DEFAULT location is (10, 10, 13)
            if(defaultScenario == True):
                xCurrentPositionUAV = 10
                yCurrentPositionUAV = 10
                zCurrentPositionUAV = 13
            else:
                xCurrentPositionUAV = uavServingLocation[0]
                yCurrentPositionUAV = uavServingLocation[1]
                zCurrentPositionUAV = uavServingLocation[2]
            
            meanTotalCapacity[episodeIdx, counterOfUpdates] = totalCapacityPerUpdate/updateNumberOfIterations
            meanTotalInterference[episodeIdx, counterOfUpdates] = totalInterferencePerUpdate/updateNumberOfIterations

            # Comment if the simulation is run for reference methods (i.e. 
            # the UE associations are already recorded).
            numberOfUEsAssociatedToUAV[episodeIdx, counterOfUpdates] = len(idsOfUEstoAssociateForUAV)
            # idsOfUEsAssociatedToABS.append(idsOfUEstoAssociateForUAV)
            listOfUpdateIds.append(movementIterationIdx)

            # Debug    
            # print(movementIterationIdx, idsOfUEstoAssociateForUAV)#, 10*np.log10(uavMeasurements))

            if(len(listsOfChannelAllocationsAllUEs) == 1):
                idsOfGBSonly = np.append(idsOfGBSonly, counterOfUpdates)                              
            else:
                idsOfGBS_UAV = np.append(idsOfGBS_UAV, counterOfUpdates)

            simulationUpdateIdx += 1 # Denote the current UAV serving session                  
            counterOfUpdates += 1  # Denote the current update of the system  
            totalCapacityPerUpdate = 0   
            totalInterferencePerUpdate = 0    
            
            # Remove association lists from previous updates of the UAV associations
            if( len(listsOfAssociatedGBSsIds) > 1 ):
                del listsOfAssociatedGBSsIds[1]
                del listsOfAssociatedUEsIds[1]
                del listsOfChannelAllocationsAllUEs[1]
                apListIdx = 0 # Signify that the associations to the GBS are to be defined first, 
                              # and after that, the associations to the UAV 
            
            # In case there are left-over indices lists from previous associations
            if(len(listsOfChannelAllocationsAllUEs)>2):
                del listsOfChannelAllocationsAllUEs[1]                     
          
            # Remove the indices of the UEs which are to be associated to the UAV
            correspondingIds = np.where(np.isin(listOfAssociatedUEsIds, idsOfUEstoAssociateForUAV))[0]
            listOfAssociatedGBSsIds = np.delete(listOfAssociatedGBSsIds, correspondingIds)  
            listsOfAssociatedGBSsIds[apListIdx] = listOfAssociatedGBSsIds
            # Add the indices of the UEs which are to be associated to the UAV                  
            listOfAssociatedGBSsIds = np.append(listOfAssociatedGBSsIds, 
                                            np.ones(idsOfUEstoAssociateForUAV.shape[0], dtype=int) )
            listsOfAssociatedGBSsIds.append(np.ones(idsOfUEstoAssociateForUAV.shape[0], dtype=int))
            # Convert the list to that of UEs with their associated UAV
            listOfAssociatedUEsIds = np.delete(listOfAssociatedUEsIds, correspondingIds)
            listsOfAssociatedUEsIds[apListIdx] = listOfAssociatedUEsIds
            listsOfAssociatedUEsIds.append(idsOfUEstoAssociateForUAV)
            listOfAssociatedUEsIds = np.append(listOfAssociatedUEsIds, idsOfUEstoAssociateForUAV)  

            uavAssociationsDone = True

        # For each iteration for which the UAV is serving its UEs, allocate channels
        # and find the path loss
        if( (uavAssociationsDone == True and 1 <= simulationUpdateIdx <= proprietyCoefficient+1 
            and movementIterationIdx > 0 and len(listsOfAssociatedUEsIds) > 1) ):  
            # Allocate random channels for the GBS-UE pairs (connections). 
            # All channels are available for the single GBS case, while those which
            # have low SINR will be used by the UAV, rather than the GBS 
            apListIdx = 1 # Indicates the UAV
            listOfUnoccupiedChannelsIDs = np.setdiff1d( np.arange(0, numberOfChannels), 
                                                    listOfChannelAllocations_ )
            # Perform allocation of the defined number of channels for all connection
            performChannelAllocation = ChannelAllocation(movementIterationIdx)
            performChannelAllocation.DetermineAllocatedChannelIDs(listOfUnoccupiedChannelsIDs,
                                                                listsOfAssociatedUEsIds[apListIdx], 
                                                                numberOfChannels)
            # List which holds all connection IDs and their respective allocated channels
            # in the form [connectionID, [listOfAllocatedChannelsForThisConnection]]
            listOfChannelAllocationsUE_UAV = performChannelAllocation.listOfChannelAllocations      
            listOfChannelAllocations = np.append( listOfChannelAllocations_, 
                                            np.array(performChannelAllocation.debugArray) )
            
            # In case there are left-over indices lists from previous associations
            if(len(listsOfChannelAllocationsAllUEs)>=2):
                del listsOfChannelAllocationsAllUEs[1]                    

            listsOfChannelAllocationsAllUEs.append(listOfChannelAllocationsUE_UAV)

            ##############################################################            

            # Find the path loss from the UAV to the UEs
            pathLossObject.pathLossUAV_GN(  xCurrentPositionUAV,
                                            yCurrentPositionUAV,
                                            zCurrentPositionUAV,
                                            xPositionsMovingUEs[movementIterationIdx], 
                                            yPositionsMovingUEs[movementIterationIdx],
                                            zPositionsMovingUEs[movementIterationIdx], uavPathlossCase)
            # For Debug
            currentPathLoss[:, apListIdx] = pathLossObject.pathLoss.T[:, 0]
            currentPathLoss[listsOfAssociatedUEsIds[~apListIdx], apListIdx] = 0
            # Leave only the path loss to the UEs that are currently associated with the UAV
            pathLossBetweenUEandGBS_or_UAV[listsOfAssociatedUEsIds[apListIdx]] = pathLossObject.pathLoss.T[:,0][
                                                                        listsOfAssociatedUEsIds[apListIdx]]
            # In absolute units because it is used for further calculations        

        # Calculate the path loss of the UEs associated with the GBS for the current iteration
        # Calculate the DL useful signal power
        currentSeed = int((movementIterationIdx + episodeIdx + 2) * 3)
        pathLossObject = PathLossCalculationFast_.PathLossCalculation(currentSeed + 25)
        
        currentPathLoss = np.zeros((numberOfUEs, numberOfGBSs+numberOfUAVs)) # Debug array
        apListIdx = 0 # Indicating the GBS
        pathLossObject.traditionalModel(xPositionsGBS,
                                        yPositionsGBS,
                                        zPositionsGBS,
                                        xPositionsMovingUEs[movementIterationIdx], #listsOfAssociatedUEsIds[apListIdx]
                                        yPositionsMovingUEs[movementIterationIdx],
                                        zPositionsMovingUEs[movementIterationIdx])
        # For Debug
        currentPathLoss[:, apListIdx] = pathLossObject.pathLoss.T[:, 0]
        currentPathLoss[listsOfAssociatedUEsIds[~apListIdx], apListIdx] = 0
        # Leave only the path loss to the UEs that are currently associated with the GBS
        pathLossBetweenUEandGBS_or_UAV[listsOfAssociatedUEsIds[apListIdx]] = pathLossObject.pathLoss.T[:,0][
                                                                    listsOfAssociatedUEsIds[apListIdx]]
        # In absolute units because it is used for further calculations

        # Delete those entries from listsOfChannelAllocationsAllUEs[0], which are  
        # the same as those in listsOfAssociatedUEsIds[1]
        if(len(listsOfAssociatedUEsIds)>1):
            idsToFilter = []
            if(len(listsOfChannelAllocationsAllUEs[0]) == numberOfUEs):            
                for elementIdx in range(0, len(listsOfChannelAllocationsAllUEs[0]) ):
                    if( any( listsOfChannelAllocationsAllUEs[0][elementIdx][0
                                        ] == listsOfAssociatedUEsIds[1] ) ): 
                        idsToFilter.append(elementIdx)

                for index in sorted(idsToFilter, reverse=True):
                    del listsOfChannelAllocationsAllUEs[0][index]
                


        currentChannelsIDs = []
        overallNumberOfAllocatedChannels = np.zeros(numberOfUEs, dtype=int)
        for listOfChannelAllocationsUE in listsOfChannelAllocationsAllUEs:
            for currentConnectionIDIdx in range(0, len(listOfChannelAllocationsUE)):
                # Calculate the Performance Metrics for the DL connection
                # The channel IDs which are allocated to the current connection
                currentChannelsIDsForGBS = listOfChannelAllocationsUE[currentConnectionIDIdx][1]
                currentChannelsIDs.append(currentChannelsIDsForGBS)
                overallNumberOfAllocatedChannels[listOfChannelAllocationsUE[currentConnectionIDIdx][0]] = len(currentChannelsIDsForGBS) 

        # Calculate for each UE starting from those associated with the GBS,
        # and then for those associated with the UAV
        for apListIdx in range(0, len(listsOfAssociatedGBSsIds)):
            sortedUEIDs = np.sort(listsOfAssociatedUEsIds[apListIdx])
            for apIdx in range(0, len(listsOfAssociatedGBSsIds[apListIdx])):
                ueIdx = sortedUEIDs[apIdx]
                rng = np.random.default_rng((movementIterationIdx*ueIdx) + ueIdx)
                receivedUsefulSignalPowerAtUE[ueIdx] = overallNumberOfAllocatedChannels[ #overallNumberOfAllocatedChannels
                    ueIdx] * downlinkPowerLevelOfGBSorUAV * pathLossBetweenUEandGBS_or_UAV[ueIdx] #* np.power(10,
                     #(rng.normal(0, 9, 1)[0])/10) 
                    # This is the measurement noise
                     
        # No interference case
        if( len(listsOfAssociatedUEsIds)>1 ):
            # In this case the UAV is already serving its UEs, and if the 
            # simulationUpdateIdx is higher than the proprietyCoefficient, i.e.
            # the REM is no longer relevant, go back to the UAV measurement stage
            if( movementIterationIdx % updateNumberOfIterations == updateNumberOfIterations - 1
                and simulationUpdateIdx > proprietyCoefficient):
                simulationUpdateIdx = 0 # Go back to the UAV measurement and the stage at which the GBS 
                                        # serves the UEs on its own    

        # Initialize the performanceMetrics object for the metrics' calculation
        performanceMetrics = PerformanceMetricsCalculation(channelBandwidth)
        # Calculate the Performance Metrics for DL (i.e. UEs' point of view)
        performanceMetrics.CalculateSINR(receivedUsefulSignalPowerAtUE,#[listsOfSortedIDsForUEs], #[listOfAssociatedUEsIds],
        overallNumberOfAllocatedChannels,#[listsOfSortedIDsForUEs], #[listOfAssociatedUEsIds],
        receivedPowerSumForInterferingGBSsAtUE,#[listsOfSortedIDsForUEs],#[listOfAssociatedUEsIds],
        noiseFigure = ueNoiseFigure )
        SINR_UE = performanceMetrics.SinrInDb     
        SINR = performanceMetrics.SINR
                            # currentNumberOfAllocatedChannels[listOfAssociatedUEsIds]
        capacityCurrentUE = overallNumberOfAllocatedChannels * channelBandwidth * np.log2(1 + SINR) / 1e6 # for Mbps
        totalCapacityPerUpdate += np.sum(capacityCurrentUE)

        totalInterferencePerUpdate = np.sum(receivedPowerSumForInterferingGBSsAtUE)

        if( movementIterationIdx == numberOfMovementIterations - 1 or (movementIterationIdx > 0 
            and movementIterationIdx % updateNumberOfIterations == 0 and simulationUpdateIdx == 0) ):
            # Save the results for the previous associations
            meanTotalCapacity[episodeIdx, counterOfUpdates] = totalCapacityPerUpdate/updateNumberOfIterations
            meanTotalInterference[episodeIdx, counterOfUpdates] = totalInterferencePerUpdate/updateNumberOfIterations

            if(len(idsOfUEstoAssociateForUAV) < 1):
                idsOfGBSonly = np.append(idsOfGBSonly, counterOfUpdates)                              
            else:
                idsOfGBS_UAV = np.append(idsOfGBS_UAV, counterOfUpdates)  
                absAndGBSSessionCounter += 1          
            totalCapacityPerUpdate = 0   
            totalInterferencePerUpdate = 0  
            # Debug
            # print(movementIterationIdx, meanTotalCapacity[episodeIdx, counterOfUpdates])
            counterOfUpdates += 1          
        

            breakpoint = ''
        # Debug                 
        # print(movementIterationIdx, np.sum(capacityCurrentUE))
        # print(movementIterationIdx, SINR_UE, listOfAssociatedUEsIds, totalCapacityPerUpdate)

        concludingCalculationsEndingTime = time()


    # Separate into two arrays using the indices of the simulation stages, which correspond
    # to the GBS only, and UAV+GBS stages
    apOnlyCapacity = meanTotalCapacity[ episodeIdx, idsOfGBSonly ]
    uavCapacity = meanTotalCapacity[ episodeIdx, idsOfGBS_UAV ]
    apOnlyInterference = meanTotalInterference[episodeIdx, idsOfGBSonly]
    uavInterference = meanTotalInterference[episodeIdx, idsOfGBS_UAV]
    apOnlyCapacityOverall.append(apOnlyCapacity)
    uavCapacityOverall.append(uavCapacity)
    apOnlyInterferenceOverall.append(apOnlyInterference)
    uavInterferenceOverall.append(uavInterference)    

    variablesString = 'norm_'+str(proprietyCoefficient) + '_'+ uavServingLocationCase + '_' + str(resolution) + '_' + str(numberOfUEs) + '_' + str(int(remSamplingRate*100)) + '_' + str(int(percentageOfChannelsForGBS*100)) +  '_' + str(int(numberOfMovementIterations/1000))

    endingTimeOverall = time()
    print('overall time in minutes: ' + str((endingTimeOverall - startingTime)/60))
    np.save(str('Results/apOnlyCapacityOverall_' + variablesString +'k.npy'), apOnlyCapacityOverall)
    np.save(str('Results/uavCapacityOverall_'+ variablesString +'k.npy'), uavCapacityOverall)
    np.save(str('Results/totalCapacity_'+ variablesString +'k.npy'), meanTotalCapacity)
    np.save(str('Results/numberOfUEsAssociatedToUAV_'+ variablesString +'k.npy'), numberOfUEsAssociatedToUAV)
    np.save(str('Results/coordinatesOfUEsAssociatedToABS_'+ variablesString +'k.npy'),
     np.array(idsOfUEsAssociatedToABS, dtype="object"))
    np.save(str('Results/listOfUpdateIds_'+ variablesString +'k.npy'), listOfUpdateIds)
    np.save(str('Results/idsOfABSPositions_'+ variablesString +'k.npy'), np.array(idsOfABSPositions))
    np.save(str('Results/activeABS_'+ variablesString +'k.npy'), activeABS)
    np.save(str('Results/allReconstructedREMs_'+ variablesString +'k.npy'), allReconstructedREMs)
    print('')

    # Get the CDF for analysis    
    count, bins_count = np.histogram(uavCapacityOverall, bins=10) 
    pdf = count / sum(count) # finding the PDF of the histogram using count values 
    cdf = np.cumsum(pdf) # using numpy np.cumsum to calculate the CDF 
    plt.plot(bins_count[1:], cdf, label="CDF") 
    

# %%
print('')


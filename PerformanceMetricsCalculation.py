# %%
import numpy as np
import random
import matplotlib.pyplot as plt
import PathLossCalculationFast_
import ChannelAllocation
# %%
# Class which takes the calculated channel parameters (fading, path-loss, etc.) and
# computes the Performance Metrics of the link
class PerformanceMetricsCalculation:
    staticPowerConsumption = 56 # In [W] -the static power consumption of the 
                                # UE’s components for cooling and signal processing
    feederAndPowerAmplifierLossAdditionalCost = 2.6 # Constant in which accounts for 
                                    # the additional power cost due 
                                    # to feeder loss and power amplifier
    
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        # Calculate the noise power for the relevant bandwidth (BW)
        # The formula is in [dBm] so it's converted to [dBW] by subtracting 30 dB
        self.noisePowerInDbw = - 174 - 30 + ( 10 * np.log10(bandwidth) )
        # Convert to absolute units
        self.noisePower = np.power(10, self.noisePowerInDbw / 10)

    # Calculate the signal-to-noise-plus-interference ratio (SINR) of the link
    def CalculateSINR(self, receivedUsefulSignalPowerSum, numberOfAllocatedChannels = 1, 
                    receivedPowerSumForInterferingUEs = 0, receivedPowerSumForInterferingAPs = 0,
                    noiseFigure = 1):
                    self.SINR = receivedUsefulSignalPowerSum / ( receivedPowerSumForInterferingUEs +
                    receivedPowerSumForInterferingAPs + 
                    (self.noisePower * numberOfAllocatedChannels * noiseFigure) ) 
                    # The line above calculates the noise power over all allocated channels and
                    # it is also multipled by the NoiseFugure which is in times (absolute unit).
                    self.SinrInDb = 10 * np.log10(self.SINR) # Convert SINR to dB
    
    # Calculate the Data Rate of the link
    def CalculateDataRate(self, SINR, numberOfAllocatedChannels = 1):
        self.dataRate = self.bandwidth * numberOfAllocatedChannels * np.log2(1 + SINR)
    
    # Calculate the Energy Efficiency Coefficient of the link
    # It describes the ratio of the rate achievable by an AP being 
    # the communication source, for the power it consumes.
    def CalculateEnergyEfficiency(self, dataRate, transmissionPower, numberOfAllocatedChannels = 1):
        self.energyEfficiency = dataRate / ( self.staticPowerConsumption +
         (self.feederAndPowerAmplifierLossAdditionalCost * transmissionPower * numberOfAllocatedChannels) )


    def calculateReceivedPowerForConnection(self, idxOfUE, idxOfBS, xPositionsBS, yPositionsBS,
                                         xPositionsUE, yPositionsUE, movementIterationIdx,
                                         listOfChannelAllocations, bsTransmissionPower, ueTransmissionPower):
        seed = idxOfUE + movementIterationIdx                                         
        # Calculate the DL useful signal power
        pathLossOfCurrentPair = PathLossCalculationFast_.PathLossCalculationFast_(seed, xPositionsBS[idxOfBS],
                                                    yPositionsBS[idxOfBS],
                                                    xPositionsUE[idxOfUE][movementIterationIdx],
                                                    yPositionsUE[idxOfUE][movementIterationIdx])
        pathLossOfCurrentPair.traditionalModel()
        pathLossBetweenUEandBS = pathLossOfCurrentPair.pathLoss # In absolute units because it is used for further calculations
        self.distanceBetweenUEandBSIn2D = round(pathLossOfCurrentPair.distance2D, 2)
        self.distanceBetweenUEandBSIn3D = round(pathLossOfCurrentPair.distance3D, 2)
        # Calculate the Performance Metrics for the DL and UL connections
        # The channel IDs which are allocated to the current connection
        # Review it later so we can have 1 channel for each PU and bar it from use for SUs
        # but SUs channels can be shared, i.e. their interference is permitted.
        currentChannelsIDs = listOfChannelAllocations[idxOfUE][1] # for now
        currentNumberOfAllocatedChannels = 1 #len(currentChannelsIDs)
        
        # Received power on the allocated channels for DL (at UE) 
        self.receivedUsefulSignalPowerAtUE = currentNumberOfAllocatedChannels * bsTransmissionPower / pathLossBetweenUEandBS
        # Received power on the allocated channels for UL (at AP)
        self.receivedUsefulSignalPowerAtBS = currentNumberOfAllocatedChannels * ueTransmissionPower / pathLossBetweenUEandBS
               

# %% Test Performance Metrics Calculation on vanilla user assciation (UA) schemes
performanceMetricsTest = False
#%%
if(performanceMetricsTest):
    #Input Parameters
    numberOfIterations = 1

    # Parameters for UE positions generation
    LengthOfSimulatedRegion = 200 # Lenght of the simulated region (a square), measured in m
    distribution = 'Matern' # Desired distribution for the UEs in the grid
                            # Options:
                            # 'Poisson'
                            # 'Uniform'
                            # 'Matern' (for Mattern hard core process which considers the minimum distance
                            # between the points generated by a Poisson process)
    numberOfUEs = 20 # For Uniform distribution, we set the exact number of UEs
    meanNumberOfUEs = 20 # For Poisson distribtion, we define the mean around which 
                        # the Poisson distributed number of UEs will vary

    userAssociationScheme = 1 # 1 - For single user association (UA) based on smallest path-loss between AP and UE
                            # 2 - For single UA based on shortest distance between AP and UE

    numberOfAPs = 20 # For Uniform distribution, we set the exact number of APs
    meanNumberOfAPs = 20 # For Poisson distribtion, we define the mean around which 
                        # the Poisson distributed number of APs will vary
    overallBandwidth = 10e6 # 10 MHz - System BW according to TR 36.828
    channelBandwidth = 180e3 # 180 kHz per channel  
    numberOfChannels = int(overallBandwidth / channelBandwidth) # Number of frequency channels (subbands)

    # UE and AP Tx power from Ming Ding's papers
    ueTransmissionPower = np.power(10, (23 - 30) / 10) / numberOfChannels # 23 dBm, first converted
                                                    # into dBW and then into absolute units and
                                                    # divided equally by the number of channels
    apTransmissionPower = np.power(10, (24 - 30) / 10) / numberOfChannels # 24 dBm, first converted
                                                    # into dBW and then into absolute units and
                                                    # divided equally by the number of channels

    apNoiseFigure = np.power(10, 13 / 10) # 13 dB for AP according to TR 36.828, converted
                                        # into absolute units
    ueNoiseFigure = np.power(10, 9 / 10) # 9 dB for UE according to TR 36.828, converted
                                        # into absolute units

    # Generate the number of UEs and APs (for Poisson distributed points, only) 
    # and their locations for this iteration
    for iteration in range(0, numberOfIterations): 
        if(distribution == 'Poisson'):  
            # Generate the UEs
            # The density describes the number of UEs per square area (i.e. the squared length of the simulation area)
            densityOfUEs = meanNumberOfUEs / np.power(LengthOfSimulatedRegion, 2)  
            ueLocations = PathLossCalculationFast_.DefineLocationsOfAPsAndUEs(LengthOfSimulatedRegion, iteration) # The iteration's
                                                                                        # number is used as
                                                                                        # seed for the UE
                                                                                        # locations generators.
            
            # Determine the number of UEs according to the Poisson distribution, as well as their locations
            ueLocations.PoissonDistributedPoints(densityOfUEs)
            numberOfUEs = ueLocations.numberOfPoissonDistributedPoints
            xPositionsUE = ueLocations.xPositions 
            yPositionsUE = ueLocations.yPositions 

            # Generate the APs
            # The density describes the number of APs per square area (i.e. the squared length of the simulation area)
            densityOfAPs = meanNumberOfAPs / np.power(LengthOfSimulatedRegion, 2)  
            apLocations = PathLossCalculationFast_.DefineLocationsOfAPsAndUEs(LengthOfSimulatedRegion, iteration + 10) # The iteration's
                                                                                        # number is used as
                                                                                        # seed for the AP
                                                                                        # locations generators. An arbitrary number
                                                                                        # is added to avoid generating the APs
                                                                                        # on the same locations as the UEs.
            
            # Determine the number of APs according to the Poisson distribution, as well as their locations
            apLocations.PoissonDistributedPoints(densityOfAPs)
            numberOfAPs = apLocations.numberOfPoissonDistributedPoints
            xPositionsAP = apLocations.xPositions 
            yPositionsAP = apLocations.yPositions 
        
        elif(distribution == 'Uniform'):
            ueLocations = PathLossCalculationFast_.DefineLocationsOfAPsAndUEs(LengthOfSimulatedRegion, iteration) # The iteration's
                                                                                        # number is used as
                                                                                        # seed for the UE
                                                                                        # locations generators.
            # Determine the locations of the Uniform distributed UEs
            ueLocations.UniformDistributedPoints(numberOfUEs)
            xPositionsUE = ueLocations.xPositions 
            yPositionsUE = ueLocations.yPositions    

            # Generate the APs
            apLocations = PathLossCalculationFast_.DefineLocationsOfAPsAndUEs(LengthOfSimulatedRegion, iteration + 10) # The iteration's
                                                                                        # number is used as
                                                                                        # seed for the AP
                                                                                        # locations generators. An arbitrary number
                                                                                        # is added to avoid generating the APs
                                                                                        # on the same locations as the UEs.
            
            # Determine the number of APs according to the Poisson distribution, as well as their locations
            apLocations.UniformDistributedPoints(numberOfAPs)
            xPositionsAP = apLocations.xPositions 
            yPositionsAP = apLocations.yPositions 

        elif(distribution == 'Matern'):
            # Generate the APs
            # The density describes the number of APs per square area (i.e. the squared length of the simulation area)
            densityOfAPs = meanNumberOfAPs / np.power(LengthOfSimulatedRegion, 2)  
            apLocations = PathLossCalculationFast_.DefineLocationsOfAPsAndUEs(LengthOfSimulatedRegion, iteration + 10) # The iteration's
                                                                                        # number is used as
                                                                                        # seed for the AP
                                                                                        # locations generators. An arbitrary number
                                                                                        # is added to avoid generating the APs
                                                                                        # on the same locations as the UEs.
            
            # Determine the number of APs according to the Matern distribution, as well as their locations
            # Remove APs which are within 30 m from each other
            apLocations.MaternDistributedPoints(densityOfAPs, 30)
            numberOfAPs = apLocations.NumberOfThinnedPoints
            xPositionsAP = apLocations.xPositions 
            yPositionsAP = apLocations.yPositions 

            # Generate the UEs
            # The density describes the number of UEs per square area (i.e. the squared length of the simulation area)
            densityOfUEs = meanNumberOfUEs / np.power(LengthOfSimulatedRegion, 2)  
            ueLocations = PathLossCalculationFast_.DefineLocationsOfAPsAndUEs(LengthOfSimulatedRegion, iteration) # The iteration's
                                                                                        # number is used as
                                                                                        # seed for the UE
                                                                                        # locations generators.
            
            # Determine the number of UEs according to the Poisson distribution, as well as their locations
            ueLocations.PoissonDistributedPoints(densityOfUEs)
            # Utilize the thinning process to remove UEs which are within 0.3 m from any AP
            ueLocations.MaternDistributedPoints(densityOfUEs, 0.3, xPositionsAP, yPositionsAP)
            numberOfUEs = ueLocations.NumberOfThinnedPoints
            xPositionsUE = ueLocations.xPositions 
            yPositionsUE = ueLocations.yPositions      

        idsOfAssociatedUEs = [] # List containing the indices for the UEs with which
                                # the APs are associated with.
        parameterLabelsForUEs = [] # List containing the distance, pathLoss and 
                                    # Performance Metrics for UEs (i.e. in DL)
        parameterLabelsForAPs = [] # List containing the distance, pathLoss and 
                                    # Performance Metrics for APs (i.e. in UL)

        # AP-UE association based on lowest PathLoss 
        if(userAssociationScheme == 1):              
            # User Association:
            # Calculates the PathLoss for each AP to the current UE. The lowest loss
            # will determine which AP the UE will associate with (association scheme 2).
            # Once associated with a UE, change of this association will not be performed.    
            for apLocationIdx in range(0, numberOfAPs):
                pathLossForUE = [] # List containing the path-losses for all possible UE paths
                                # to the current AP
                for ueLocationIdx in range(0, numberOfUEs):
                    # Calculates the PathLoss for each AP to the current UE.
                    pathLoss = PathLossCalculationFast_.PathLossCalculationFast_(ueLocationIdx, xPositionsAP[apLocationIdx],
                                                                        yPositionsAP[apLocationIdx],
                                                                        xPositionsUE[ueLocationIdx],
                                                                        yPositionsUE[ueLocationIdx])
                    # Determine the P(LOS) for each point and calculate the respective PathLoss
                    pathLoss.determineProbabilityOfLOS()
                    pathLoss.calculatePathLoss()
                    pathLossBetweenUEandAP = round(10 * np.log10(pathLoss.pathLoss), 2) # PathLoss in dB
                                                                                        # rounded to the
                                                                                        # 2nd digit after 
                                                                                        # the decimal point.
                    pathLossForUE.append(pathLossBetweenUEandAP)      

                # In the Poisson-distibuted APs and UEs case, the number of APs may be higher than 
                # that of the UEs and once all UEs are associated the association process
                # (the loop) must cease. This is not necessary for the Uniform case because
                # the number of UEs and APs will always be the same.
                if(len(idsOfAssociatedUEs) == numberOfUEs):
                    #print(len(idsOfAssociatedUEs), numberOfUEs) # Debug
                    break

                # Generate the ids of the sorted (in ascending order) path-losses of all UEs to the current AP  
                sortedIdsOfPathLossForUE = list( np.argsort(pathLossForUE) )
                idxOfAssociatedUE = sortedIdsOfPathLossForUE[-1] # Get the index of the smallest path-loss for 
                                                                # this AP. It will be the last element because
                                                                # the path-losses are in dB and negative numbers.
                
                # If the idx of the UE with lowest path-loss is already taken (i.e. the UE is associated
                # with another AP), the idx of the UE with the next lowest path-loss is chosen
                # until the idx of an unassociated UE is reached
                while(idxOfAssociatedUE in idsOfAssociatedUEs):  
                    sortedIdsOfPathLossForUE.remove(idxOfAssociatedUE)
                    idxOfAssociatedUE = sortedIdsOfPathLossForUE[-1] 
                
                # Add the idx of the UE associated to the current AP, to the list
                idsOfAssociatedUEs.append(idxOfAssociatedUE)

            # Channel Allocation:
            # Allocate channels for current connection. Uses random allocation as a baseline.
            numberOfConnections = np.minimum(numberOfAPs, numberOfUEs) # In single UA, it is the smaller
                                                                    # number between the APs and UEs
                                                                    # because each UE associates with
                                                                    # only one AP.
            idsOfConnection = np.arange(0, numberOfConnections) # Creates a list of the connection IDs
            listOfUnoccupiedChannelsIDs = np.arange(0, numberOfChannels + 1) # Creates a list of the
                                                                            # unoccupied channel IDs
            # Perform allocation of the defined number of channels for all connection
            performChannelAllocation = ChannelAllocation.ChannelAllocation(numberOfChannels, iteration)
            performChannelAllocation.DetermineAllocatedChannelIDs(idsOfConnection, listOfUnoccupiedChannelsIDs)
            # List which holds all connection IDs and their respective allocated channels
            # in the form [connectionID, [listOfAllocatedChannelsForThisConnection]]
            listOfChannelAllocations = performChannelAllocation.listOfChannelAllocations      

            # Loop thorugh all connections IDs to find the interference
            for currentConnectionIDIdx in range(0, len(listOfChannelAllocations)): 
                # print(currentConnectionIDIdx, idsOfAssociatedUEs[currentConnectionIDIdx]) # Debug

                # Calculate the DL useful signal power
                pathLossOfCurrentPair = PathLossCalculationFast_.PathLossCalculationFast_(currentConnectionIDIdx, 
                                                            xPositionsAP[currentConnectionIDIdx],
                                                            yPositionsAP[currentConnectionIDIdx],
                                                            xPositionsUE[idsOfAssociatedUEs[currentConnectionIDIdx]],
                                                            yPositionsUE[idsOfAssociatedUEs[currentConnectionIDIdx]])
                pathLossOfCurrentPair.determineProbabilityOfLOS()
                pathLossOfCurrentPair.calculatePathLoss()
                pathLossBetweenUEandAP = pathLossOfCurrentPair.pathLoss # In absolute units because it is used for further calculations
                distanceBetweenUEandAPIn3D = round(pathLossOfCurrentPair.distance3D, 2)
                # Calculate the Performance Metrics for the DL and UL connections
                # The channel IDs which are allocated to the current connection
                currentChannelsIDs = listOfChannelAllocations[currentConnectionIDIdx][1]
                currentNumberOfAllocatedChannels = len(currentChannelsIDs)
                # Received power on the allocated channels for DL (at UE) 
                receivedUsefulSignalPowerAtUE = currentNumberOfAllocatedChannels * apTransmissionPower * pathLossBetweenUEandAP
                # Received power on the allocated channels for UL (at AP)
                receivedUsefulSignalPowerAtAP = currentNumberOfAllocatedChannels * ueTransmissionPower * pathLossBetweenUEandAP

                # Check if and which channels allocated to this connection, are used in others as
                # they create interference.

                # Create a list of the possible interfering connections, i.e. all connections
                # except the current one
                listOfPossibleInterferers = [] # List that stores the possible interfering connections
                # Loop through all connections (their channels are laready allocated)
                for idx in range(0, len(listOfChannelAllocations)):
                    if(idx != currentConnectionIDIdx):
                        listOfPossibleInterferers.append(listOfChannelAllocations[idx])

                interferenceForCurrentConnectionAtAP = 0 # Variable which stores the accumulated 
                                                        # interference at AP of all interfering channels
                                                        # at the current step (interfering connection)
                interferenceForCurrentConnectionAtUE = 0 # Variable which stores the accumulated 
                                                        # interference at UE of all interfering channels
                                                        # at the current step (interfering connection)
                # Loop through all possible interfering connections to accumulate the interference
                # of their interfering channels (where other connections transmit on the same channels)
                for currentPotentialInterferingConnectionID in listOfPossibleInterferers: 
                    # Find the number of interfering channels for the current potential-interfering connection
                    # by comparing the number of matching channel IDs between the channel IDs of 
                    # the current potential-interfering connection and the channels of the current connection
                    numberOfInterferingChannelsForCurrentInterferingID = np.count_nonzero( np.in1d(currentPotentialInterferingConnectionID[1],
                                                                    currentChannelsIDs) )
                        
                    # Find interference for current interfering ID (currentPotentialInterferingConnectionIDIdx) and 
                    # the current connection ID (currentConnectionIDIdx) here and add it to
                    # find per-subchannel interference for this potential-interfering connection
                                        
                    # Calculate the interference per channel here for the current interfering ID
                    # (if the number of interfering channels are > 0, then there are interfering channels)
                    if(numberOfInterferingChannelsForCurrentInterferingID > 0):
                        # Calculate the DL interference from the APs which the other UEs are associated with, to the AP
                        # which the current UE is associated with (does not include the interference from APs
                        # which are not associated to any UEs, if there are any, they are inactive).
                        
                        receivedPowerSumForInterferingAPsAtAP = 0  # Stores the accumulated received interference
                                                                # from other APs.
                        # Determine the P(LOS) for each interfering AP to the current AP and calculate
                        # the respective PathLoss and Performance Metrics             
                        # All APs but the AP which the UE is associated to, are interfering.
                        pathLoss = PathLossCalculationFast_.PathLossCalculationFast_(currentConnectionIDIdx,
                                                            xPositionsAP[currentConnectionIDIdx],
                                                            yPositionsAP[currentConnectionIDIdx],
                                                            xPositionsAP[currentPotentialInterferingConnectionID[0]],
                                                            yPositionsAP[currentPotentialInterferingConnectionID[0]])
                        pathLoss.determineProbabilityOfLOS()
                        pathLoss.calculatePathLoss()
                        pathLossBetweenAPandInterferingAP = pathLoss.pathLoss # In absolute units because 
                                                                            # it is used for further calculations
                        # The cumulative interference of the current interfering connection is equal
                        # to the numberOfInterferingChannels * powerPerChannel * PathLoss                                                                        
                        receivedPowerSumForInterferingAPsAtAP += numberOfInterferingChannelsForCurrentInterferingID * apTransmissionPower * pathLossBetweenAPandInterferingAP

                        # Calculate the DL interference from the APs which the other UEs are associated with, to the current UE
                        # (does not include the interference from APs which are not associated to any UEs, 
                        # if there are any, they are inactive).
                        receivedPowerSumForInterferingAPsAtUE = 0 
                        # Determine the P(LOS) for each interfering AP to the current UE and calculate
                        # the respective PathLoss and Performance Metrics             
                        # All APs but the AP which the UE is associated to, are interfering
                        pathLoss = PathLossCalculationFast_.PathLossCalculationFast_(currentConnectionIDIdx,
                                                                    xPositionsUE[idsOfAssociatedUEs[currentConnectionIDIdx]],
                                                                    yPositionsUE[idsOfAssociatedUEs[currentConnectionIDIdx]],
                                                                    xPositionsAP[currentPotentialInterferingConnectionID[0]],
                                                                    yPositionsAP[currentPotentialInterferingConnectionID[0]])
                        pathLoss.determineProbabilityOfLOS()
                        pathLoss.calculatePathLoss()
                        pathLossBetweenUEandInterferingAP = pathLoss.pathLoss # In absolute units because 
                                                                            # it is used for further calculations
                        # The cumulative interference of the current interfering connection is equal
                        # to the numberOfInterferingChannels * powerPerChannel * PathLoss                     
                        receivedPowerSumForInterferingAPsAtUE += numberOfInterferingChannelsForCurrentInterferingID * apTransmissionPower * pathLossBetweenUEandInterferingAP
                    else:
                        # If the number of interfering channels is 0, then there are no interfering channels
                        receivedPowerSumForInterferingAPsAtAP = 0
                        receivedPowerSumForInterferingAPsAtUE = 0

                    # Accumulate the interference of the current interfering connection to the AP
                    interferenceForCurrentConnectionAtAP += receivedPowerSumForInterferingAPsAtAP
                    # Accumulate the interference of the current interfering connection to the UE
                    interferenceForCurrentConnectionAtUE += receivedPowerSumForInterferingAPsAtUE
                
                # Initialize the performanceMetrics object for the metrics' calculation
                performanceMetrics = PerformanceMetricsCalculation(channelBandwidth)
                # Calculate the Performance Metrics for DL (i.e. UEs' point of view)
                performanceMetrics.CalculateSINR(receivedUsefulSignalPowerAtUE, currentNumberOfAllocatedChannels, 0,
                        interferenceForCurrentConnectionAtUE, ueNoiseFigure)
                downlinkSinrInDb = performanceMetrics.SinrInDb
                performanceMetrics.CalculateDataRate(performanceMetrics.SINR, currentNumberOfAllocatedChannels)
                downlinkDataRate = performanceMetrics.dataRate
                # Calculate the Performance Metrics for UL  (i.e. APs' point of view)
                performanceMetrics.CalculateSINR(receivedUsefulSignalPowerAtAP, currentNumberOfAllocatedChannels, 0,
                        interferenceForCurrentConnectionAtAP, apNoiseFigure)
                uplinkSinrInDb = performanceMetrics.SinrInDb
                performanceMetrics.CalculateDataRate(performanceMetrics.SINR, currentNumberOfAllocatedChannels)
                uplinkDataRate = performanceMetrics.dataRate            
                # Enery Efficiency for the AP requires the DL Data Rate 
                performanceMetrics.CalculateEnergyEfficiency(downlinkDataRate, apTransmissionPower, currentNumberOfAllocatedChannels)
                energyEfficiencyAP = performanceMetrics.energyEfficiency

                # Store the DL Parameter Metrics for each UE as a list of strings
                parameterLabelsForUEs.append([ " UE: " + str(idsOfAssociatedUEs[currentConnectionIDIdx]),
                                                            "ch: " + str(currentNumberOfAllocatedChannels),
                                                            " AP: " + str(currentConnectionIDIdx), 
                                                            str(distanceBetweenUEandAPIn3D) + " m ", 
                                                            str(round(downlinkSinrInDb, 2)) + " dB ",
                                                            str(round(downlinkDataRate/1e3, 2)) + " kbps"
                                                                    ])

                # Store the UL Parameter Metrics for each AP as a list of strings
                parameterLabelsForAPs.append([ " AP: " + str(currentConnectionIDIdx), 
                                                            " UE: " + str(idsOfAssociatedUEs[currentConnectionIDIdx]), 
                                                            str(round(uplinkSinrInDb, 2)) + " dB ",
                                                            str(round(uplinkDataRate/1e3, 2)) + " kbps",
                                                            " EE: " + str(round(energyEfficiencyAP/1e3, 2)) + " kbpJ"
                                                                    ])                                                                    
        
        # AP-UE association based on shortest distance
        elif(userAssociationScheme == 2):
            # User Association:
            # Calculates the distance (in 3D) for each AP to the current UE. The AP will 
            # associate with the UE which is nearest to it (association scheme 2).
            # Once associated with a UE, change of this association will not be performed.    
            for apLocationIdx in range(0, numberOfAPs):#for ueLocationIdx in range(0, numberOfUEs):#
                distanceForUE = [] # List containing the path-losses for all possible UE paths
                                # to the current AP
                for ueLocationIdx in range(0, numberOfUEs):#for apLocationIdx in range(0, numberOfAPs):#
                    # Calculates the distance for each AP to the current UE.
                    pathLoss = PathLossCalculationFast_.PathLossCalculationFast_(ueLocationIdx, xPositionsAP[apLocationIdx],
                                                                        yPositionsAP[apLocationIdx],
                                                                        xPositionsUE[ueLocationIdx],
                                                                        yPositionsUE[ueLocationIdx])
                    distanceBetweenUEandAPIn3D = round(10 * np.log10(pathLoss.distance3D), 2) # Distance in m
                                                                                        # rounded to the
                                                                                        # 2nd digit after 
                                                                                        # the decimal point.
                    distanceForUE.append(distanceBetweenUEandAPIn3D)      

                # In the Poisson-distibuted APs and UEs case, the number of APs may be higher than 
                # that of the UEs and once all UEs are associated the association process
                # (the loop) must cease. This is not necessary for the Uniform case because
                # the number of UEs and APs will always be the same.
                if(len(idsOfAssociatedUEs) == numberOfUEs):#if(len(idsOfAssociatedUEs) == numberOfAPs):#
                    print(len(idsOfAssociatedUEs), numberOfUEs)#print(len(idsOfAssociatedUEs), numberOfAPs)#
                    break

                # Generate the ids of the sorted (in ascending order) path-losses of all UEs to the current AP  
                sortedIdsOfDistanceForUE = list( np.argsort(distanceForUE) )
                idxOfAssociatedUE = sortedIdsOfDistanceForUE[0] # Get the index of the shortest distance for 
                                                                # this AP. 
                
                # If the idx of the UE with shortest distance is already taken (i.e. the UE is associated
                # with another AP), the idx of the UE with the next shortest distance is chosen
                # until the idx of an unassociated UE is reached
                while(idxOfAssociatedUE in idsOfAssociatedUEs):  
                    sortedIdsOfDistanceForUE.remove(idxOfAssociatedUE)
                    idxOfAssociatedUE = sortedIdsOfDistanceForUE[-1] 
                
                # Add the idx of the UE associated to the current AP, to the list
                idsOfAssociatedUEs.append(idxOfAssociatedUE)

            # Channel Allocation:
            # Allocate channels for current connection. Uses random allocation as a baseline.
            numberOfConnections = np.minimum(numberOfAPs, numberOfUEs) # In single UA, it is the smaller
                                                                    # number between the APs and UEs
                                                                    # because each UE associates with
                                                                    # only one AP.
            idsOfConnection = np.arange(0, numberOfConnections) # Creates a list of the connection IDs
            listOfUnoccupiedChannelsIDs = np.arange(0, numberOfChannels + 1) # Creates a list of the
                                                                            # unoccupied channel IDs
            # Perform allocation of the defined number of channels for all connection
            performChannelAllocation = ChannelAllocation.ChannelAllocation(numberOfChannels, iteration)
            performChannelAllocation.DetermineAllocatedChannelIDs(idsOfConnection, listOfUnoccupiedChannelsIDs)
            # List which holds all connection IDs and their respective allocated channels
            # in the form [connectionID, [listOfAllocatedChannelsForThisConnection]]
            listOfChannelAllocations = performChannelAllocation.listOfChannelAllocations      

            # Loop thorugh all connections IDs to find the interference
            for currentConnectionIDIdx in range(0, len(listOfChannelAllocations)): 
                # print(currentConnectionIDIdx, idsOfAssociatedUEs[currentConnectionIDIdx]) # Debug

                # Calculate the DL useful signal power
                pathLossOfCurrentPair = PathLossCalculationFast_.PathLossCalculationFast_(currentConnectionIDIdx, 
                                                            xPositionsAP[currentConnectionIDIdx],
                                                            yPositionsAP[currentConnectionIDIdx],
                                                            xPositionsUE[idsOfAssociatedUEs[currentConnectionIDIdx]],
                                                            yPositionsUE[idsOfAssociatedUEs[currentConnectionIDIdx]])
                pathLossOfCurrentPair.determineProbabilityOfLOS()
                pathLossOfCurrentPair.calculatePathLoss()
                pathLossBetweenUEandAP = pathLossOfCurrentPair.pathLoss # In absolute units because it is used for further calculations
                distanceBetweenUEandAPIn3D = round(pathLossOfCurrentPair.distance3D, 2)
                # Calculate the Performance Metrics for the DL and UL connections
                # The channel IDs which are allocated to the current connection
                currentChannelsIDs = listOfChannelAllocations[currentConnectionIDIdx][1]
                currentNumberOfAllocatedChannels = len(currentChannelsIDs)
                # Received power on the allocated channels for DL (at UE) 
                receivedUsefulSignalPowerAtUE = currentNumberOfAllocatedChannels * apTransmissionPower * pathLossBetweenUEandAP
                # Received power on the allocated channels for UL (at AP)
                receivedUsefulSignalPowerAtAP = currentNumberOfAllocatedChannels * ueTransmissionPower * pathLossBetweenUEandAP

                # Check if and which channels allocated to this connection, are used in others as
                # they create interference.

                # Create a list of the possible interfering connections, i.e. all connections
                # except the current one
                listOfPossibleInterferers = [] # List that stores the possible interfering connections
                # Loop through all connections (their channels are laready allocated)
                for idx in range(0, len(listOfChannelAllocations)):
                    if(idx != currentConnectionIDIdx):
                        listOfPossibleInterferers.append(listOfChannelAllocations[idx])

                interferenceForCurrentConnectionAtAP = 0 # Variable which stores the accumulated 
                                                        # interference at AP of all interfering channels
                                                        # at the current step (interfering connection)
                interferenceForCurrentConnectionAtUE = 0 # Variable which stores the accumulated 
                                                        # interference at UE of all interfering channels
                                                        # at the current step (interfering connection)
                # Loop through all possible interfering connections to accumulate the interference
                # of their interfering channels (where other connections transmit on the same channels)
                for currentPotentialInterferingConnectionID in listOfPossibleInterferers: 
                    # Find the number of interfering channels for the current potential-interfering connection
                    # by comparing the number of matching channel IDs between the channel IDs of 
                    # the current potential-interfering connection and the channels of the current connection
                    numberOfInterferingChannelsForCurrentInterferingID = np.count_nonzero( np.in1d(currentPotentialInterferingConnectionID[1],
                                                                    currentChannelsIDs) )
                        
                    # Find interference for current interfering ID (currentPotentialInterferingConnectionIDIdx) and 
                    # the current connection ID (currentConnectionIDIdx) here and add it to
                    # find per-subchannel interference for this potential-interfering connection
                                        
                    # Calculate the interference per channel here for the current interfering ID
                    # (if the number of interfering channels are > 0, then there are interfering channels)
                    if(numberOfInterferingChannelsForCurrentInterferingID > 0):
                        # Calculate the DL interference from the APs which the other UEs are associated with, to the AP
                        # which the current UE is associated with (does not include the interference from APs
                        # which are not associated to any UEs, if there are any, they are inactive).
                        
                        receivedPowerSumForInterferingAPsAtAP = 0  # Stores the accumulated received interference
                                                                # from other APs.
                        # Determine the P(LOS) for each interfering AP to the current AP and calculate
                        # the respective PathLoss and Performance Metrics             
                        # All APs but the AP which the UE is associated to, are interfering.
                        pathLoss = PathLossCalculationFast_.PathLossCalculationFast_(currentConnectionIDIdx,
                                                            xPositionsAP[currentConnectionIDIdx],
                                                            yPositionsAP[currentConnectionIDIdx],
                                                            xPositionsAP[currentPotentialInterferingConnectionID[0]],
                                                            yPositionsAP[currentPotentialInterferingConnectionID[0]])
                        pathLoss.determineProbabilityOfLOS()
                        pathLoss.calculatePathLoss()
                        pathLossBetweenAPandInterferingAP = pathLoss.pathLoss # In absolute units because 
                                                                            # it is used for further calculations
                        # The cumulative interference of the current interfering connection is equal
                        # to the numberOfInterferingChannels * powerPerChannel * PathLoss                                                                        
                        receivedPowerSumForInterferingAPsAtAP += numberOfInterferingChannelsForCurrentInterferingID * apTransmissionPower * pathLossBetweenAPandInterferingAP

                        # Calculate the DL interference from the APs which the other UEs are associated with, to the current UE
                        # (does not include the interference from APs which are not associated to any UEs, 
                        # if there are any, they are inactive).
                        receivedPowerSumForInterferingAPsAtUE = 0 
                        # Determine the P(LOS) for each interfering AP to the current UE and calculate
                        # the respective PathLoss and Performance Metrics             
                        # All APs but the AP which the UE is associated to, are interfering
                        pathLoss = PathLossCalculationFast_.PathLossCalculationFast_(currentConnectionIDIdx,
                                                                    xPositionsUE[idsOfAssociatedUEs[currentConnectionIDIdx]],
                                                                    yPositionsUE[idsOfAssociatedUEs[currentConnectionIDIdx]],
                                                                    xPositionsAP[currentPotentialInterferingConnectionID[0]],
                                                                    yPositionsAP[currentPotentialInterferingConnectionID[0]])
                        pathLoss.determineProbabilityOfLOS()
                        pathLoss.calculatePathLoss()
                        pathLossBetweenUEandInterferingAP = pathLoss.pathLoss # In absolute units because 
                                                                            # it is used for further calculations
                        # The cumulative interference of the current interfering connection is equal
                        # to the numberOfInterferingChannels * powerPerChannel * PathLoss                     
                        receivedPowerSumForInterferingAPsAtUE += numberOfInterferingChannelsForCurrentInterferingID * apTransmissionPower * pathLossBetweenUEandInterferingAP
                    else:
                        # If the number of interfering channels is 0, then there are no interfering channels
                        receivedPowerSumForInterferingAPsAtAP = 0
                        receivedPowerSumForInterferingAPsAtUE = 0

                    # Accumulate the interference of the current interfering connection to the AP
                    interferenceForCurrentConnectionAtAP += receivedPowerSumForInterferingAPsAtAP
                    # Accumulate the interference of the current interfering connection to the UE
                    interferenceForCurrentConnectionAtUE += receivedPowerSumForInterferingAPsAtUE
            
                # Initialize the performanceMetrics object for the metrics' calculation
                performanceMetrics = PerformanceMetricsCalculation(channelBandwidth)
                # Calculate the Performance Metrics for DL (i.e. UEs' point of view)
                performanceMetrics.CalculateSINR(receivedUsefulSignalPowerAtUE, currentNumberOfAllocatedChannels, 0,
                        interferenceForCurrentConnectionAtUE, ueNoiseFigure)
                downlinkSinrInDb = performanceMetrics.SinrInDb
                performanceMetrics.CalculateDataRate(performanceMetrics.SINR, currentNumberOfAllocatedChannels)
                downlinkDataRate = performanceMetrics.dataRate
                # Calculate the Performance Metrics for UL  (i.e. APs' point of view)
                performanceMetrics.CalculateSINR(receivedUsefulSignalPowerAtAP, currentNumberOfAllocatedChannels, 0,
                        interferenceForCurrentConnectionAtAP, apNoiseFigure)
                uplinkSinrInDb = performanceMetrics.SinrInDb
                performanceMetrics.CalculateDataRate(performanceMetrics.SINR, currentNumberOfAllocatedChannels)
                uplinkDataRate = performanceMetrics.dataRate            
                # Enery Efficiency for the AP requires the DL Data Rate 
                performanceMetrics.CalculateEnergyEfficiency(downlinkDataRate, apTransmissionPower, currentNumberOfAllocatedChannels)
                energyEfficiencyAP = performanceMetrics.energyEfficiency

                # Store the DL Parameter Metrics for each UE as a list of strings
                parameterLabelsForUEs.append([ " UE: " + str(idsOfAssociatedUEs[currentConnectionIDIdx]),
                                                            "ch: " + str(currentNumberOfAllocatedChannels),
                                                            " AP: " + str(currentConnectionIDIdx), 
                                                            str(distanceBetweenUEandAPIn3D) + " m ", 
                                                            str(round(downlinkSinrInDb, 2)) + " dB ",
                                                            str(round(downlinkDataRate/1e3, 2)) + " kbps"
                                                                    ])

                # Store the UL Parameter Metrics for each AP as a list of strings
                parameterLabelsForAPs.append([ " AP: " + str(currentConnectionIDIdx), 
                                                            " UE: " + str(idsOfAssociatedUEs[currentConnectionIDIdx]), 
                                                            str(round(uplinkSinrInDb, 2)) + " dB ",
                                                            str(round(uplinkDataRate/1e3, 2)) + " kbps",
                                                            " EE: " + str(round(energyEfficiencyAP/1e3, 2)) + " kbpJ"
                                                                    ])                   

        print("APs: ", numberOfAPs, "UEs: ", numberOfUEs)
        # Plotting - APs are in red, UEs in green, next to each AP, its associated UE idx
        # and the Performance Metrics of the AP-UE pair
        fig, ax = plt.subplots(figsize=(15, 15)) # Large figure size so the parameterLabels can be clearly seen
        # Plot the APs and UEs as scatter points
        ax.scatter(xPositionsAP, yPositionsAP, edgecolor='r', facecolor='none')
        ax.scatter(xPositionsUE, yPositionsUE, edgecolor='g', facecolor='none')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.grid()
        # Label each UE point with the relevant parameters
        for pointCounter, parameterLabel in enumerate(parameterLabelsForUEs):
            ax.annotate(parameterLabel, (xPositionsUE[idsOfAssociatedUEs[pointCounter]], 
                                        yPositionsUE[idsOfAssociatedUEs[pointCounter]]))

        # Label each AP point with the relevant parameters
        for pointCounter, parameterLabel in enumerate(parameterLabelsForAPs):
            ax.annotate(parameterLabel, (xPositionsAP[pointCounter], yPositionsAP[pointCounter]))

        # Label any unassociated APs/UEs with their idx number for clarity
        if(numberOfAPs > numberOfUEs):
            for pointCounter, parameterLabel in enumerate(np.arange(numberOfAPs-numberOfUEs, numberOfAPs)):
                ax.annotate(parameterLabel, (xPositionsAP[pointCounter+(numberOfAPs-numberOfUEs)],
                                            yPositionsAP[pointCounter+(numberOfAPs-numberOfUEs)]))        

        if(numberOfUEs > numberOfAPs):
            for pointCounter, parameterLabel in enumerate(np.arange(0, numberOfUEs)):
                # Labels a UE by its number idx only if it does not exist in the list of associated UE's indices
                if(pointCounter not in idsOfAssociatedUEs):
                    ax.annotate(parameterLabel, (xPositionsUE[pointCounter],
                                                yPositionsUE[pointCounter]))        
        plt.show()

# %%

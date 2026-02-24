#%%
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

import PathLossCalculationFast

#%% Class definition for generating user locations
class LocationsGenerator:

    def __init__(self, LengthOfSimulatedRegion, seed):
        # Input variables for the inner-class methods
        self.seed = seed
        self.LengthOfSimulatedRegion = LengthOfSimulatedRegion
        self.simulatedArea = np.power(self.LengthOfSimulatedRegion, 2) # Area of the simulated region (square)
                                                                
    # Method for generation of Poisson-distributed points
    def PoissonDistributedPoints(self, seedseed, numberOfPoints):
        ''' 
        Generates Poisson point process
        '''
        
        # Define the Poisson distributed points' coordinates
        np.random.seed(self.seed + seedseed) # Seed for the Poisson distribution
        
        # The density describes the number of UEs per square area (i.e. the squared length of the simulation area)
        densityForPoissonDistribution = numberOfPoints / self.simulatedArea  

        # Determine the number of points according to the Poisson distribution.        
        self.numberOfPoissonDistributedPoints = np.random.poisson(densityForPoissonDistribution * self.simulatedArea) 
        # The points are generated to span between -0.5*LengthOfSimulatedRegion and 0.5*LengthOfSimulatedRegion
        self.xPositions = self.LengthOfSimulatedRegion * np.random.uniform(-0.5, 0.5, self.numberOfPoissonDistributedPoints) 
        self.yPositions = self.LengthOfSimulatedRegion * np.random.uniform(-0.5, 0.5, self.numberOfPoissonDistributedPoints) 
    
    # Method for generation of uniformly-distributed points
    def UniformDistributedPoints(self, seedseed, numberOfUEsForUniformDistribution):
        # Define the uniformly distributed points' coordinates
        np.random.seed(self.seed + seedseed) # Seed for the Uniform distribution
        # The points are generated to span between -0.5*LengthOfSimulatedRegion and 0.5*LengthOfSimulatedRegion
        self.xPositions = np.random.uniform(- (self.LengthOfSimulatedRegion/2), 
                                            (self.LengthOfSimulatedRegion/2), numberOfUEsForUniformDistribution) 
        self.yPositions = np.random.uniform(- (self.LengthOfSimulatedRegion/2), 
                                            (self.LengthOfSimulatedRegion/2), numberOfUEsForUniformDistribution)          
    
    # Method for generation of Matern Type II distributed points (a type of Poisson hard core process)
    def MaternDistributedPoints(self, 
                                seedseed, 
                                numberOfPoints, 
                                minimumDistanceBetweenPoints,
                                xPositionsForReference = [], 
                                yPositionsForReference = []):
        
        # The density describes the number of APs per square area (i.e. the squared length of the simulation area)
        densityForPoissonDistribution = numberOfPoints / self.simulatedArea  

        # Define Poisson-distributed points' coordinates
        np.random.seed(self.seed + seedseed) # Seed for the Poisson distribution
        halfsideOfSimulatedRegion = self.LengthOfSimulatedRegion / 2
        # Determine the number of points according to the Poisson distribution.        
        self.numberOfPoissonDistributedPoints = np.random.poisson(densityForPoissonDistribution * self.simulatedArea) 
        # Generating Poisson-distributed points
        # The points are generated to span between -0.5*LengthOfSimulatedRegion and 0.5*LengthOfSimulatedRegion
        self.xPositionsPoisson = self.LengthOfSimulatedRegion * np.random.uniform(-0.5, 0.5, self.numberOfPoissonDistributedPoints) 
        self.yPositionsPoisson = self.LengthOfSimulatedRegion * np.random.uniform(-0.5, 0.5, self.numberOfPoissonDistributedPoints) 
        
        # If there aren't already produced reference points to which those within the minimum distance 
        # are to be removed, Matern type II distributed points are generated
        if(len(xPositionsForReference) == 0):
            # Retain only the points which are inside the simulated region
            ifPointsInsideTheRegion = ((self.xPositionsPoisson >=  - halfsideOfSimulatedRegion)&
                                    (self.xPositionsPoisson <= halfsideOfSimulatedRegion)&
                                    (self.yPositionsPoisson >= - halfsideOfSimulatedRegion)&
                                    (self.yPositionsPoisson <= halfsideOfSimulatedRegion)) 
            # Ids of the points inside the region                                
            pointsInsideTheRegionIds = np.arange(self.numberOfPoissonDistributedPoints)[ifPointsInsideTheRegion]
            self.xPositionsInsideTheRegion = self.xPositionsPoisson[pointsInsideTheRegionIds]
            self.yPositionsInsideTheRegion = self.yPositionsPoisson[pointsInsideTheRegionIds]
            # Set the new number of Poisson-distributed points after those outside the region are removed
            self.numberOfPointsInsideTheRegion = len(self.xPositionsInsideTheRegion)            
            # Define random marks in [0, 1] for the initial Poisson-distributed points
            marksOfinitalPoints = np.random.rand(self.numberOfPoissonDistributedPoints)
            # Boolean array which stores the ids of the retained points after thinning
            idsOfRetainedPoints = np.zeros(self.numberOfPointsInsideTheRegion, dtype=bool)
        
            # Thinning - remove all points which are within the minimumDistanceBetweenPoints
            # and have smaller marks than their neighbors in the initial points 
            for idxOfPoints in range(self.numberOfPointsInsideTheRegion):
                distanceBetweenPoints = np.hypot(self.xPositionsInsideTheRegion[idxOfPoints] - self.xPositionsPoisson, 
                                            self.yPositionsInsideTheRegion[idxOfPoints] - self.yPositionsPoisson);  
                # Equivalent to np.sqrt( (xPositionsInsideTheRegion[idxOfPoints] - xPositionsPoisson)**2 
                # + (yPositionsInsideTheRegion[idxOfPoints] - yPositionsPoisson)**2 )
                ifInsideTheMinimumDistance = ( (distanceBetweenPoints < minimumDistanceBetweenPoints) & 
                                            (distanceBetweenPoints > 0) )

                # If the current point has no marks, it is retained.
                # All initial points which have higher marks than their neighbors, are retained
                # (their indices are set to True)
                if len(marksOfinitalPoints[ifInsideTheMinimumDistance]) == 0:
                    idsOfRetainedPoints[idxOfPoints] = True
                else:
                    idsOfRetainedPoints[idxOfPoints]=all( marksOfinitalPoints[ifInsideTheMinimumDistance] > 
                                    marksOfinitalPoints[pointsInsideTheRegionIds[idxOfPoints]] )

            # Define the Matern-distributed points
            self.xPositions = self.xPositionsInsideTheRegion[idsOfRetainedPoints]
            self.yPositions = self.yPositionsInsideTheRegion[idsOfRetainedPoints] 
            self.NumberOfThinnedPoints = len(self.xPositions)   

        # If there are already produced reference points to which those within the minimum distance 
        # are to be removed, this is performed
        else:

            # Generate numberOfPoints according the Poisson distribution
            self.PoissonDistributedPoints(seedseed,numberOfPoints)

            # Thinning according to the points of reference - remove all points which are within 
            # the minimumDistanceBetweenPoints to the points of reference

            # Boolean array which stores the ids of the retained points after thinning
            idsOfRetainedPoints = np.ones(len(self.xPositionsPoisson), dtype=bool)            
            for idxOfPoints in range(len(self.xPositionsPoisson)):
                distanceBetweenPoints = np.hypot(self.xPositionsPoisson[idxOfPoints] - xPositionsForReference, 
                                            self.yPositionsPoisson[idxOfPoints] - yPositionsForReference)         
                # Equivalent to np.sqrt( (xPositionsInsideTheRegion[idxOfPoints] - xPositionsPoisson)**2 
                # + (yPositionsInsideTheRegion[idxOfPoints] - yPositionsPoisson)**2 )
                ifInsideTheMinimumDistance = ( (distanceBetweenPoints < minimumDistanceBetweenPoints) & 
                                            (distanceBetweenPoints > 0) )
                # If the point is within the minimum distance range with at least one of the 
                # reference points, it is to be removed, i.e. its index is False
                if(any(ifInsideTheMinimumDistance)):
                    idsOfRetainedPoints[idxOfPoints] = False

            # Define the thinned Poisson-distributed points
            self.xPositions = self.xPositionsPoisson[idsOfRetainedPoints]
            self.yPositions = self.yPositionsPoisson[idsOfRetainedPoints]               
            # The number of points that remain after thinning
            self.NumberOfThinnedPoints = len(self.xPositions)

#%%
class HetNet:
    
    def __init__(self,
                 numberOfSUs = 100,
                 numberOfPUs = 50,
                 numberOfAPs = 10, 
                 numberOfPBS = 1,
                 overallBandwidth = 20e6, # 20 MHz - System BW 
                 channelBandwidth = 180e3, # 180 kHz per channel  
                 apNoiseFigureInAbs = np.power(10, 13 / 10), # 13 dB for AP according to TR 36.828, converted, into absolute units
                 ueNoiseFigureInAbs = np.power(10, 9 / 10), # 9 dB for UE according to TR 36.828, converted
                 apNoisePowerPerChannel = 7.161434102129027e-16,
                 LengthOfSimulatedRegionInMeters = 1000,
                 seed = 2904):

        # Copy the configuration paramters to the object instance
        self.numberOfSUs = numberOfSUs
        self.numberOfPUs = numberOfPUs
        self.numberOfAPs = numberOfAPs 
        self.numberOfPBS = numberOfPBS
        self.overallBandwidth = overallBandwidth
        self.channelBandwidth = channelBandwidth
        self.apNoiseFigureInAbs = apNoiseFigureInAbs
        self.ueNoiseFigureInAbs = ueNoiseFigureInAbs
        self.apNoisePowerPerChannel = apNoisePowerPerChannel
        self.LengthOfSimulatedRegionInMeters = LengthOfSimulatedRegionInMeters
        self.seed = seed

        # Initalize some relevant matrix
        self.userAssociationMatrix = np.zeros((self.numberOfAPs,self.numberOfSUs))
        self.pathLossMatrix = np.zeros((self.numberOfAPs,self.numberOfSUs))

        # Init locations
        self.locationsSUs = np.zeros((self.numberOfSUs,2))
        self.locationsPUs = np.zeros((self.numberOfPUs,2))
        self.locationsAPs = np.zeros((self.numberOfAPs,2))
        self.locationPBS  = np.zeros((1,2))

        # RA stuff initializations
        self.numberOfChannels = int(self.overallBandwidth / self.channelBandwidth) # Number of frequency channels (subbands)
        self.channelAllocationMatrix = np.zeros((self.numberOfAPs,self.numberOfChannels))
        
        # Power stuff initializations
        self.apTransmissionPower = np.power(10, (24 - 30) / 10) / self.numberOfChannels # 24 dBm, first converted
                                                        # into dBW and then into absolute units and
                                                        # divided equally by the number of channels
        self.pbsTransmissionPower = np.power(10, (43 - 30) / 10) / self.numberOfChannels # 43 dBm, first converted
        self.powerLevelsForAP = self.apTransmissionPower * np.ones(self.numberOfAPs)

        #Helper objects initializations
        self.pathLossObject = PathLossCalculationFast.PathLossCalculation(self.seed)
        
    def GeneratInitialLocations(self,seedseed,distribution):
        ''' 
            This method generates randomly distributed points according one of the following distributions:
            Poisson, Uniform, Mattern. 
            Note that during the Poisson and Mattern generation process the values of numberOfPUs, numberOfSUs 
            and numberOfAPs are used as the mean values for the generation process. After the points are generated then these
            values are overwritten with the ones generated by the generator    
        '''

        # ------------------------------------------
        # Generate the PBSs
        self.xPositionPBS = np.array([0])
        self.yPositionPBS = np.array([0])

        #Create the random points generator object
        locgen = LocationsGenerator(self.LengthOfSimulatedRegionInMeters, self.seed + seedseed) 
        
        if(distribution == 'Poisson'):  

            # -------------------------------------------------            
            # Generate the PUEs - rewrite numberOfPUs
            locgen.PoissonDistributedPoints(23355, self.numberOfPUs)
            self.numberOfPUs = locgen.numberOfPoissonDistributedPoints
            self.xStartingPositionsPUE = locgen.xPositions 
            self.yStartingPositionsPUE = locgen.yPositions 

            # ------------------------------------------
            # Generate the SUEs - rewrite numberOfSUs
            locgen.PoissonDistributedPoints(45634, self.numberOfSUs)
            self.numberOfSUs = locgen.numberOfPoissonDistributedPoints
            self.xStartingPositionsSUE = locgen.xPositions 
            self.yStartingPositionsSUE = locgen.yPositions 

            # ------------------------------------------
            # Generate the APs - rewrite numberOfAPs
            locgen.PoissonDistributedPoints(19274384,self.numberOfAPs)
            self.numberOfAPs = locgen.numberOfPoissonDistributedPoints
            self.xPositionsAP = locgen.xPositions
            self.yPositionsAP = locgen.yPositions

        elif(distribution == 'Uniform'):

            # Generate the locations of the Uniform distributed UEs
            locgen.UniformDistributedPoints(3456239, self.numberOfPUs)
            self.xStartingPositionsPUE = locgen.xPositions 
            self.yStartingPositionsPUE = locgen.yPositions    

            # Generate the locations of the Uniform distributed UEs
            locgen.UniformDistributedPoints(2432, self.numberOfSUs)
            self.xStartingPositionsSUE = locgen.xPositions 
            self.yStartingPositionsSUE = locgen.yPositions    

            # Generate uniformly distributed locations of APs
            locgen.UniformDistributedPoints(1121212, self.numberOfAPs)
            self.xPositionsAP = locgen.xPositions
            self.yPositionsAP = locgen.yPositions

        elif(distribution == 'Matern'):

            # ---------------------------------------------------------            
            # Generate APs according to the Matern distribution
            # Remove APs which are within 30 m from each other
            locgen.MaternDistributedPoints(4344099,self.numberOfAPs, 30)
            self.numberOfAPs = locgen.NumberOfThinnedPoints
            self.xPositionsAP = locgen.xPositions
            self.yPositionsAP = locgen.yPositions

            # ---------------------------------------------------------            
            # Generate PUs according to the Matern distribution
            # Utilize the thinning process to remove UEs which are within 0.3 m from any AP
            locgen.MaternDistributedPoints(2343242, self.numberOfPUs, 0.3, self.xPositionsAP, self.yPositionsAP)
            self.numberOfPUs = locgen.NumberOfThinnedPoints
            self.xStartingPositionsPUE = locgen.xPositions 
            self.yStartingPositionsPUE = locgen.yPositions      

            
            # ---------------------------------------------------------            
            # Generate the SUEs according Mattern
            # Utilize the thinning process to remove UEs which are within 0.3 m from any AP
            locgen.MaternDistributedPoints(65732,self.numberOfSUs, 0.3, self.xPositionsAP, self.yPositionsAP)
            self.numberOfSUs = locgen.NumberOfThinnedPoints
            self.xStartingPositionsSUE = locgen.xPositions 
            self.yStartingPositionsSUE = locgen.yPositions   

    def SimulateUserMovements(self,seedseed, numberOfMovementIterations = 100, averageMovementSpeed = 1):
        ''' 
            Simulates movements of all PUs and SUs using Random Walk model.
            maxMovementSpeed is in m/s and 1 means 3 km/h
        '''
        
        # Set seed
        np.random.seed( self.seed + seedseed )

        # --------------------------------------------------------------------
        # Generate movement for PUs
        # speed = np.random.uniform(low = 0, high = 2*averageMovementSpeed, size=(self.numberOfPUs,numberOfMovementIterations))
        speed = np.random.normal (loc = averageMovementSpeed, scale = averageMovementSpeed/2, size=(self.numberOfPUs,numberOfMovementIterations))

        direction = np.random.uniform(low = 0, high = 90, size=(self.numberOfPUs,numberOfMovementIterations))
        self.locationsPUs = np.zeros((self.numberOfPUs,2,numberOfMovementIterations))
        self.locationsPUs[:,0,:] = speed * np.cos(direction) 
        self.locationsPUs[:,1,:] = speed * np.sin(direction) 
        self.locationsPUs[:,0,0] = np.array(self.xStartingPositionsPUE) 
        self.locationsPUs[:,1,0] = np.array(self.yStartingPositionsPUE) 
        self.locationsPUs = np.cumsum(self.locationsPUs,axis=2)    

        # --------------------------------------------------------------------
        # Generate movement for SUs
        # speed = np.random.uniform(low = 0, high = 2*averageMovementSpeed, size=(self.numberOfSUs,numberOfMovementIterations))
        speed = np.random.normal (loc = averageMovementSpeed, scale = averageMovementSpeed/2, size=(self.numberOfSUs,numberOfMovementIterations))
        
        direction = np.random.uniform(low = 0, high = 90, size=(self.numberOfSUs,numberOfMovementIterations))
        self.locationsSUs = np.zeros((self.numberOfSUs,2,numberOfMovementIterations))
        self.locationsSUs[:,0,:] = speed * np.cos(direction) 
        self.locationsSUs[:,1,:] = speed * np.sin(direction) 
        self.locationsSUs[:,0,0] = np.array(self.xStartingPositionsSUE) 
        self.locationsSUs[:,1,0] = np.array(self.yStartingPositionsSUE) 
        self.locationsSUs = np.cumsum(self.locationsSUs,axis=2)    

    def RandomUserAssocationSUs(self):
        '''
        This method does user assosiaction by random assingmet SU to AP. It is used when the baseline capacity of SUNet is to be calculated. 
        '''
        
        # listOfAssociatedUEsIds = np.array([2, 2, 3, 3, 1, 3, 1, 3]) # This is for debug
        self.listOfAssociatedUEsIds = np.argmax(np.random.rand(self.numberOfAPs,self.numberOfSUs),axis=1) # Holds the argmax row-wise 
        self.userAssociationMatrix = np.zeros((self.numberOfAPs, self.numberOfSUs)) 
        self.userAssociationMatrix[np.arange(0,self.numberOfAPs), self.listOfAssociatedUEsIds] = 1

    def SmallestPathLossUserAssocationSUs(self):
        #TODO
        return 

    def RandomChannelDistribution(self):
        
        '''
        Split the number of available channels among the PUs and SUs and then allocate the corresponding indecies. 
        The amount of channels for PUs is at least the number of PUs. Then generate the channel indecies for PUs and SUs 
        '''
        
        self.numberOfChannelsForPUs = self.numberOfPUs + np.random.choice(self.numberOfChannels - self.numberOfPUs - int(self.numberOfSUs/5))
        self.channelIdsForPUs = np.random.choice(self.numberOfChannels, self.numberOfChannelsForPUs, replace = False)

        self.channelIdsForSUs = np.delete(np.arange(0, self.numberOfChannels), self.channelIdsForPUs)
        self.numberOfChannelsForSUs = self.channelIdsForSUs.size

    def RandomChannelAllocationSUsUnique(self):
        '''
        Allocates channels for APs associated to SUs, from the list of channels allocated to SUs. 
        Care is taken that no two channels serve the same station. The cannels are allocated without substitution, 
        i.e. no two same channels are used. The list of available channels is distributed among the current connections  
        ''' 

        # Create channel allocation matrix such as no two same chanels are allocated to all APs serving SU
        self.channelAllocationMatrix = np.zeros((self.numberOfAPs,self.numberOfChannels))
        associatedAPIdx = np.where(np.sum(self.userAssociationMatrix,axis=1))[0]
        TotalAPSUConnections = associatedAPIdx.size
        a = int(np.floor(self.numberOfChannelsForSUs/TotalAPSUConnections))
        r = int(self.numberOfChannelsForSUs*(1-a))
        numberChannelsToAllocateIter = [self.numberOfChannelsForSUs for i in range(0, a)]
        numberChannelsToAllocateIter.append(r)

        for i in range(0,len(numberChannelsToAllocateIter)):
            # suIdxS holds the indexes of the luckiest PUs which will get a channel
            suIdxS = np.random.choice(associatedAPIdx, numberChannelsToAllocateIter[i], replace = False)
            selChannels = self.channelIdsForSUs[i*numberChannelsToAllocateIter[i]:(i*numberChannelsToAllocateIter[i] + numberChannelsToAllocateIter[i])] 
            self.channelAllocationMatrix[suIdxS,selChannels] = 1

    def RandomChannelAllocationSUs(self):
        '''
        Allocates channels for APs associated to SUs, from the list of channels allocated to SUs. 
        Care is taken that no two channels serve the same station. The channels are allocated with substitution, 
        i.e. channel might be used by two APs not serving the same user  
        ''' 

        # Create channel allocation matrix such as no two same chanels are allocated to all APs serving SU
        self.channelAllocationMatrix = np.zeros((self.numberOfAPs,self.numberOfChannels))
        self.allChannelsIds = []
        for i in range(0,self.numberOfSUs):
            # if(i == 8):
            #     print("sadasdsa: ",i)
            APsServingSUi = np.where(self.userAssociationMatrix[:,i])[0]
            if( APsServingSUi.size != 0):
                
                # Allocate two time more channels than the available connections
                numChannelForSu = 2*APsServingSUi.shape[0]
                if(numChannelForSu > self.numberOfChannelsForSUs): 
                    numChannelForSu = int(APsServingSUi.shape[0]) #int(2 * np.floor(self.numberOfChannelsForSUs/2))

                # Subsample from the available list    
                channelIdsForSu = np.random.choice(self.channelIdsForSUs, numChannelForSu, replace = False)
                channelIdsForSu2D = np.reshape(channelIdsForSu,(int(APsServingSUi.shape[0]),-1))
                self.allChannelsIds.append(channelIdsForSu) #Debug list
                for j in range(0,APsServingSUi.shape[0]):
                    self.channelAllocationMatrix[APsServingSUi[j],channelIdsForSu2D[j,:]] = 1

    def ChannelAllocationForPUs(self):
        '''
        Allocates channels for PUs based on the already distributed indecies. Method RandomChannelDistribution should be called first
        '''
        
        # Fixed and verified 
        #Build list of channels for allocation
        a = int(np.floor(self.numberOfChannelsForPUs/self.numberOfPUs))
        r = self.numberOfChannelsForPUs - self.numberOfPUs
        numberChannelsToAllocateIter = [self.numberOfPUs for i in range(0, a)]
        numberChannelsToAllocateIter.append(r)
        self.channelAllocationsForPUs = np.zeros((self.numberOfPUs,self.numberOfChannels))
        
        for i in range(0,len(numberChannelsToAllocateIter)):
            # puIdxS holds the indexes of the luckiest PUs which will get a channel
            puIdxS = np.random.choice(range(0, self.numberOfPUs), numberChannelsToAllocateIter[i], replace = False)
            selChannels = self.channelIdsForPUs[i*numberChannelsToAllocateIter[i]:(i*numberChannelsToAllocateIter[i] + numberChannelsToAllocateIter[i])] 
            channelIdx = 0
            for pudix in puIdxS:
                self.channelAllocationsForPUs[pudix, selChannels[channelIdx]] = 1
                channelIdx += 1
            # print(channelIdx)

    def SetUserAssociationMatrix(self,UAM):
        ''' Sets the User Association Matrix'''
        #TODO: check for valid dimensions
        self.userAssociationMatrix = UAM
    
    def SetChannelAllocationMatrix(self,CAM):
        ''' Sets the Channel Allocation Matrix'''
        #TODO: check for valid dimensions
        self.channelAllocationMatrix = CAM
    
    def SetPathLossMatrix(self,PLM):
        ''' Sets the path loss matrix'''
        #TODO: check for valid dimensions
        self.channelAllocationMatrix = PLM

    def SetPowerLevelsForAP(self,powLev):
        ''' Sets the power levels for each AP. If only scalar is provided, then this scaller is assumed for all APS'''

        if(powLev.size == 1):
            self.powerLevelsForAP = powLev * np.ones(self.numberOfAPs)
        else:
            self.powerLevelsForAP = np.random.choice(powLev, self.numberOfAPs, replace=True) 

    def SUNetOnlyCapacityCalculation(self,seed,movementIterationIdx):
        ''' 
            Calculates the capacity of the SUNet given the current 
            allocation, association and power schemes. Make sure that 
            the channelAllocation, userAssocition and powerLevels are set before call to this function
        '''
        
        #Path loss calculation - assuming no interference form PUNet
        self.pathLossObject.ComputeUDNPathLoss(self.xPositionsAP,
                                               self.yPositionsAP,
                                               self.locationsSUs[:,:,movementIterationIdx])
        
        # Calculate the pathloss
        self.PathLoss = self.pathLossObject.pathLoss.cpu().detach().numpy()  # In absolute units because 

        #  Calcualte received power, channel agnostic
        PR = np.tile(np.reshape(self.powerLevelsForAP,(self.numberOfAPs,-1)), (1,self.numberOfSUs)) * self.PathLoss 
        self.C_i = np.zeros(self.numberOfSUs)
        C = 0

        # Loop over each SU, calculate capacity for each SU and accumulate
        for i in range(0,self.numberOfSUs): 
            
            #Calcualte received power of SU_i, on each channel serving it (allocated to the APs serving the SU)
            PR_i_w = np.tile(np.reshape(PR[:,i],(self.numberOfAPs,-1)), (1,self.numberOfChannels)) * self.channelAllocationMatrix
            
            # Integarte over the APs, thus calculating the total power received by the SU, channel wise
            PR_i_w_T = PR_i_w[self.userAssociationMatrix[:,i]==1,:] # get the received power by the SU only using the associated APs
            if( PR_i_w_T.size == 0): PR_i_w_T = np.zeros(self.numberOfChannels) # no Ap associated to the SU
            else:                    PR_i_w_T = np.sum(PR_i_w_T,axis=0) # Integrate out the APs

            #Calculate the interference, by integrating the power over all APs not associated to the SU, channel wise 
            I_i_w = PR_i_w[self.userAssociationMatrix[:,i]==0,:]
            if(I_i_w.size == 0): I_i_w = np.zeros(self.numberOfChannels)
            else:                I_i_w = np.sum(I_i_w,axis=0) 

            # Calculate the SINR
            SINR_i_w = PR_i_w_T / (I_i_w + self.ueNoiseFigureInAbs * self.apNoisePowerPerChannel)   
            
            # Calculate the datarate on each channel serving SU_i and the total datarate
            C_i_w = self.channelBandwidth * np.log2(1+SINR_i_w)  
            self.C_i[i] = np.sum(C_i_w)  # calculate the datarate over all channels
            C += self.C_i[i]
        
        return C


#%% Unit testing cell

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Main function used when unit test is performed 
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == "__main__":

#pragma region Input parameters

    # Parameters for UE positions generation
    LengthOfSimulatedRegion = 1000 # Lenght of the simulated region (a square), measured in m
    locationsDistribution  = 'Uniform' # Desired distribution for the UEs in the grid
                            # Options:
                            # 'Poisson'
                            # 'Uniform'
                            # 'Matern' (for Mattern hard core process which considers the minimum distance
                            # between the points generated by a Poisson process)
    numberOfSUs = 50  # For Uniform distribution, we set the exact number of UEs
    numberOfPUs = 30  # For Uniform distribution, we set the exact number of UEs
    numberOfAPs = 100 # For Uniform distribution, we set the exact number of APs
    numberOfMovementIterations = 1000
    averageMovementSpeed = 1 # 1 m/s ~ 3 kmh/h 

    # Channel related parameters
    overallBandwidth = 20e6 # 10 MHz - System BW according to TR 36.828
    channelBandwidth = 180e3 # 180 kHz per channel  
    numberOfChannels = int(overallBandwidth / channelBandwidth) # Number of frequency channels (subbands)

    # UE and AP Tx power from Ming Ding's papers
    ueTransmissionPower = np.power(10, (23 - 30) / 10) / numberOfChannels # 23 dBm, first converted
                                                    # into dBW and then into absolute units and
                                                    # divided equally by the number of channels
    apTransmissionPower = np.power(10, (24 - 30) / 10) / numberOfChannels # 24 dBm, first converted
                                                    # into dBW and then into absolute units and
                                                    # divided equally by the number of channels
    # apTransmissionPower = np.array([np.power(10, (24 - 30) / 10) / numberOfChannels,
    #                         np.power(10, (20 - 30) / 10) / numberOfChannels,
    #                         np.power(10, (15 - 30) / 10) / numberOfChannels])

    apNoiseFigureInAbs = np.power(10, 13 / 10) # 13 dB for AP according to TR 36.828, converted
                                        # into absolute units
    ueNoiseFigureInAbs = np.power(10, 9 / 10) # 9 dB for UE according to TR 36.828, converted
                                        # into absolute units
#pragma endregion

#pragma region HetNet Init

    #Create the HetNet object
    HetNetObject = HetNet(
                        numberOfSUs = numberOfSUs,
                        numberOfPUs = numberOfPUs,
                        numberOfAPs = numberOfAPs, 
                        numberOfPBS = 1,
                        overallBandwidth = overallBandwidth, 
                        channelBandwidth = channelBandwidth,   
                        apNoiseFigureInAbs = apNoiseFigureInAbs,
                        ueNoiseFigureInAbs = ueNoiseFigureInAbs,
                        apNoisePowerPerChannel = 7.161434102129027e-16,
                        LengthOfSimulatedRegionInMeters = LengthOfSimulatedRegion,
                        seed = 78778)

    # Generate initial locations of PBS, PUs, APs and SUs and display on a figure
    HetNetObject.GeneratInitialLocations(seedseed = 124632742, distribution = locationsDistribution)
    
    print("Requested number of PUs:", numberOfPUs, ", generated number of PUs: ", HetNetObject.numberOfPUs)
    print("Requested number of SUs:", numberOfSUs, ", generated number of SUs: ", HetNetObject.numberOfSUs)
    print("Requested number of APs:", numberOfAPs, ", generated number of APs: ", HetNetObject.numberOfAPs)

    plt.figure()
    plt.scatter(HetNetObject.xStartingPositionsPUE,HetNetObject.yStartingPositionsPUE,c='g')
    plt.scatter(HetNetObject.xStartingPositionsSUE,HetNetObject.yStartingPositionsSUE,c='b')
    plt.scatter(HetNetObject.xPositionsAP,HetNetObject.yPositionsAP, s=200, c='r', marker="*")
    plt.grid()
    plt.title("AP,PUs and SUs Locations")
    plt.show()

#%%
    # Generate PUs and SUs user movement
    HetNetObject.SimulateUserMovements(seedseed=65, 
                                       numberOfMovementIterations = numberOfMovementIterations, 
                                       averageMovementSpeed = averageMovementSpeed)
    plt.figure()
    for i in range(0,numberOfMovementIterations,int(numberOfMovementIterations/10)):
        plt.scatter(HetNetObject.locationsSUs[:,0,i],HetNetObject.locationsSUs[:,1,i],c=str(i/numberOfMovementIterations))
    plt.grid()
    plt.title("All movements of secondary users in 10 steps")
    plt.show()

#%%
    Capacity = np.array([])

    #Loop over all user movements
    for i in range(0,numberOfMovementIterations):
        
        # Associate users to APs
        HetNetObject.RandomUserAssocationSUs()

        # Distribute available channels among PUs and SUs
        HetNetObject.RandomChannelDistribution()

        # Allocate channels - select one of the schemes below 
        HetNetObject.RandomChannelAllocationSUs()
        # HetNetObject.RandomChannelAllocationSUsUnique()
        
        # Allocate power
        HetNetObject.SetPowerLevelsForAP(apTransmissionPower)

        # Calculate capacity
        C = HetNetObject.SUNetOnlyCapacityCalculation(2344234,i)
        Capacity = np.append(Capacity,C)

        # HetNetObject.ChannelAllocationForPUs()

# Display capacity results
print("Min,Max and Mean capacity over all iterrations: ", np.min(Capacity),np.max(Capacity),np.mean(Capacity))
plt.figure()
plt.plot(Capacity)
plt.show()

#pragma endregion
# %%

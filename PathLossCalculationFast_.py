# %%
import numpy as np
import random
import uav_radio
from scipy import spatial


#%%
class PathLossCalculation:
    # Probability PathLoss model from TR 36.828, commonly employed in Ding's papers
    heightDifference = 2#8.5 # Absolute antenna height difference between an GBS and a UE.
                           # It is 8.5 m in TR 36.814. According to Ding et al, it should be > 0.
                           # The 3GPP value is widely used in a variety of papers.
                           # Intermediary values have not been considered so far.
                           # Depending on the scenario, however, it may not be appropriate
                           # to assume such a high value because the UEs and GBSs may be
                           # much closer to each other within the same vicinity.
    # PathLoss parameters from 3GPP TR 36.828
    R1 = 156.0  # in m according to TR 36.828
    R2 = 30.0  # in m according to TR 36.828
    d1 = R1 / np.log(10.0)  # distance threshold which determines P(LOS), it is measured in m and taken from TR 36.828
    referencePathLossForLOS = 10.0 ** (-10.38)  # Reference LOS pathLoss at 1 m
    pathLossExponentForLOS = 2.09
    referencePathLossForNLOS = 10.0 ** (-10.54)  # Reference NLOS pathLoss at 1 m
    pathLossExponentForNLOS = 3.75
                                 
    def __init__(self, seed):
        # Input variables for the inner-class methods
        self.isLOS = True
        self.seed = seed

    def traditionalModel(self, xPositionGBS, yPositionGBS, zPositionGBS, xPositionUE, yPositionUE, zPositionUE):
        PositionGBS = np.vstack((xPositionGBS, yPositionGBS, zPositionGBS)).T
        PositionUE = np.vstack((xPositionUE, yPositionUE, zPositionUE)).T

        self.distance3D = np.linalg.norm(PositionGBS[:, np.newaxis] - PositionUE[np.newaxis, :], axis=2)

        self.pathLoss = 137.3 + (35.2 * np.log10(self.distance3D / 1000.0))
        self.pathLoss = np.power(10.0, self.pathLoss / 10.0)
        self.pathLoss = 1.0 / self.pathLoss
        
    def ComputeUDNPathLoss(self, xPositionGBS, yPositionGBS, zPositionGBS, xPositionUE, yPositionUE, zPositionUE):
        PositionGBS = np.vstack((xPositionGBS, yPositionGBS, zPositionGBS)).T
        PositionUE = np.vstack((xPositionUE, yPositionUE, zPositionUE)).T

        self.distance3D = np.linalg.norm(PositionGBS[:, np.newaxis] - PositionUE[np.newaxis, :], axis=2)

        self.probabilityOfLOS = (1.0 - (5.0 * np.exp(-self.R1 / self.distance3D))) * (self.distance3D <= self.d1) + (
            5.0 * np.exp(-self.distance3D / self.R2) * (self.distance3D > self.d1))

        random.seed(self.seed) # Seed for the uniformRandomVariable
        self.uniformRandomVariable = random.random() # Between 0 and 1, drawn from a uniform distribution

        self.pathLoss = self.referencePathLossForLOS * (np.power(self.distance3D, 
                            -self.pathLossExponentForLOS) * (self.probabilityOfLOS >= self.uniformRandomVariable).astype(np.float64)
                            ) + self.referencePathLossForNLOS * (np.power(self.distance3D, 
                                -self.pathLossExponentForNLOS) * (self.probabilityOfLOS < self.uniformRandomVariable).astype(np.float64))        ##self.pathLoss = self.pathLoss.T
        '...'
    
    # Modified Two Ray Path Loss Model - best for this scenario
    # "UAVRadio: Radio Link Path Loss Estimation for UAVs", http://51.91.59.240/uav-radio/
    # Calculate PL between the UAV and a ground node (GN), i.e. an UE or an GBS
    def pathLossUAV_GN(self, xPositionUAV, yPositionUAV, zPositionUAV,
                        xPositionGN, yPositionGN, zPositionGN, pathlossCase):
        carrierFrequency = 2e9
        PositionUAV = np.vstack((xPositionUAV, yPositionUAV, zPositionUAV)).T
        PositionGN = np.vstack((xPositionGN, yPositionGN, zPositionGN)).T
        self.distance3D = spatial.distance.cdist(PositionGN, PositionUAV)

        if(np.array(zPositionGN).shape == ()):
            zPositionGN = np.array([zPositionGN])
            
        UAV_PL_Calculator = uav_radio.PathLossCalculator(reference_distance=1.0, n=2.0)
        if(pathlossCase == 'U_2RM'):
            # According to the paper, G_l = 15, G_r = 5, γ = 3.5, R = -1 for zPositionGN < 30 m
            self.pathLoss = np.array([UAV_PL_Calculator.modified_two_ray_pl(self.distance3D[idx], 
                                                carrierFrequency, zPositionGN[idx], zPositionUAV, G_l_custom=15,
                                                G_r_custom=5, R_custom=-1, gamma_h_custom=3.5)[
                                                0] for idx in np.arange(0, self.distance3D.shape[0
                                                ])]).reshape(1, self.distance3D.shape[0])
        elif(pathlossCase == 'U_AB'):
            self.pathLoss = np.array([UAV_PL_Calculator.log_distance_alpha_beta_pl(distance=self.distance3D[idx], 
                                    frequency=carrierFrequency, scenario="lightly_hilly_rural", rx_height=zPositionUAV,
                                    alpha_custom = 2.9, beta_custom=7.4, sigma_custom=6.2)[
                        0]for idx in np.arange(0, self.distance3D.shape[0])]).reshape(1, self.distance3D.shape[0])
        elif(pathlossCase == 'U_2R'):
            self.pathLoss = np.array([UAV_PL_Calculator.two_ray_pl(distance=self.distance3D[idx], 
                                    frequency=carrierFrequency, h_t=zPositionGN[idx], h_r=zPositionUAV)[
                        0]for idx in np.arange(0, self.distance3D.shape[0])]).reshape(1, self.distance3D.shape[0])
        elif(pathlossCase == 'U_DS'):
            self.pathLoss = np.array([UAV_PL_Calculator.dual_slope_pl(distance=self.distance3D[idx], 
                                    frequency=carrierFrequency, gamma_1=0.74, gamma_2=2.29, default=20, d_b=9.0)[
                        0]for idx in np.arange(0, self.distance3D.shape[0])]).reshape(1, self.distance3D.shape[0])
        
        self.pathLoss = np.power(10.0, self.pathLoss / 10.0)
        self.pathLoss = 1.0 / self.pathLoss
        #print('')


# test = PathLossCalculation(2)
# test.pathLossUAV_GN(10, 20, 14, 5, 5, 1.5, 2e9)


#%% 
# Class which defines the device (GBS or UE) locations according to a desired distribution
class DefineLocationsOfGBSsAndUEsIn3D:
    def __init__(self, LengthOfSimulatedRegion, HeightOfSimulatedRegion, seed):
        # Input variables for the inner-class methods
        self.seed = seed
        self.LengthOfSimulatedRegion = LengthOfSimulatedRegion
        self.HeightOfSimulatedRegion = HeightOfSimulatedRegion
                                                                
    # Method for generation of Poisson-distributed points
    def PoissonDistributedPoints(self, densityForPoissonDistribution):
        self.simulatedArea = np.power(self.LengthOfSimulatedRegion, 2) # Area of the simulated region (square)
        
        # Define the Poisson distributed points' coordinates
        np.random.seed(self.seed) # Seed for the Poisson distribution
        
        # Determine the number of points according to the Poisson distribution.        
        self.numberOfPoissonDistributedPoints = np.random.poisson(densityForPoissonDistribution * self.simulatedArea) 
        # The points are generated to span between -0.5*LengthOfSimulatedRegion and 0.5*LengthOfSimulatedRegion
        self.xPositions = self.LengthOfSimulatedRegion * np.random.uniform(-0.5, 0.5, self.numberOfPoissonDistributedPoints) 
        self.yPositions = self.LengthOfSimulatedRegion * np.random.uniform(-0.5, 0.5, self.numberOfPoissonDistributedPoints) 
        self.zPositions = self.HeightOfSimulatedRegion * np.random.uniform(0.01, 0.99, self.numberOfPoissonDistributedPoints)
    
    # Method for generation of uniformly-distributed points
    def UniformDistributedPoints(self, numberOfUEsForUniformDistribution):
        # Define the uniformly distributed points' coordinates
        np.random.seed(self.seed) # Seed for the Uniform distribution
        # The points are generated to span between -0.5*LengthOfSimulatedRegion and 0.5*LengthOfSimulatedRegion
        self.xPositions = np.random.uniform(- (self.LengthOfSimulatedRegion/2), 
                                            (self.LengthOfSimulatedRegion/2), numberOfUEsForUniformDistribution) 
        self.yPositions = np.random.uniform(- (self.LengthOfSimulatedRegion/2), 
                                            (self.LengthOfSimulatedRegion/2), numberOfUEsForUniformDistribution)          
    
    # Method for generation of Matern Type II distributed points (a type of Poisson hard core process)
    def MaternDistributedPoints(self, densityForPoissonDistribution, minimumDistanceBetweenPoints,
                                xPositionsForReference = [], yPositionsForReference = [],
                                 zPositionsForReference = []):
        self.simulatedArea = np.power(self.LengthOfSimulatedRegion, 2) # Area of the simulated region (square)
        # Define Poisson-distributed points' coordinates
        np.random.seed(self.seed) # Seed for the Poisson distribution
        halfsideOfSimulatedRegion = self.LengthOfSimulatedRegion / 2
        # Determine the number of points according to the Poisson distribution.        
        self.numberOfPoissonDistributedPoints = np.random.poisson(densityForPoissonDistribution * self.simulatedArea) 
        # Generating Poisson-diostributed points
        # The points are generated to span between -0.5*LengthOfSimulatedRegion and 0.5*LengthOfSimulatedRegion
        self.xPositionsPoisson = self.LengthOfSimulatedRegion * np.random.uniform(-0.5, 0.5, self.numberOfPoissonDistributedPoints) 
        self.yPositionsPoisson = self.LengthOfSimulatedRegion * np.random.uniform(-0.5, 0.5, self.numberOfPoissonDistributedPoints) 
        self.zPositionsPoisson = self.HeightOfSimulatedRegion * np.random.uniform(0.01, 0.99, self.numberOfPoissonDistributedPoints)

        # If there aren't already produced reference points to which those within the minimum distance 
        # are to be removed, Matern type II distributed points are generated
        if(len(xPositionsForReference) == 0):
            # Retain only the points which are inside the simulated region
            ifPointsInsideTheRegion = ((self.xPositionsPoisson >=  - halfsideOfSimulatedRegion)&
                                    (self.xPositionsPoisson <= halfsideOfSimulatedRegion)&
                                    (self.yPositionsPoisson >= - halfsideOfSimulatedRegion)&
                                    (self.yPositionsPoisson <= halfsideOfSimulatedRegion)) 
            # Ids of the points inside the region                                
            pointsInsideTheRegionIds = np.arange(self.numberOfPoissonDistributedPoints)[
                                                                ifPointsInsideTheRegion]
            self.xPositionsInsideTheRegion = self.xPositionsPoisson[pointsInsideTheRegionIds]
            self.yPositionsInsideTheRegion = self.yPositionsPoisson[pointsInsideTheRegionIds]
            self.zPositionsInsideTheRegion = self.zPositionsPoisson[pointsInsideTheRegionIds]
            # Set the new number of Poisson-distributed points after those outside the region are removed
            self.numberOfPointsInsideTheRegion = len(self.xPositionsInsideTheRegion)            
            # Define random marks in [0, 1] for the initial Poisson-distributed points
            marksOfinitalPoints = np.random.rand(self.numberOfPoissonDistributedPoints)
            # Boolean array which stores the ids of the retained points after thinning
            idsOfRetainedPoints = np.zeros(self.numberOfPointsInsideTheRegion, dtype=bool)
        
            # Thinning - remove all points which are within the minimumDistanceBetweenPoints
            # and have smaller marks than their neighbors in the initial points 
            for idxOfPoints in range(self.numberOfPointsInsideTheRegion):
                distanceBetweenPoints = np.sqrt( (self.xPositionsInsideTheRegion[idxOfPoints] - self.xPositionsPoisson)**2 
                + (self.yPositionsInsideTheRegion[idxOfPoints] - self.yPositionsPoisson)**2 
                + (self.zPositionsInsideTheRegion[idxOfPoints] - self.zPositionsPoisson)**2 )
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
            self.zPositions = self.zPositionsInsideTheRegion[idsOfRetainedPoints]
            self.NumberOfThinnedPoints = len(self.xPositions)   

        # If there are already produced reference points to which those within the minimum distance 
        # are to be removed, this is performed
        else:
            # Thinning according to the points of reference - remove all points which are within 
            # the minimumDistanceBetweenPoints to the points of reference
            
            # Boolean array which stores the ids of the retained points after thinning
            idsOfRetainedPoints = np.ones(len(self.xPositionsPoisson), dtype=bool)            
            for idxOfPoints in range(len(self.xPositionsPoisson)):
                distanceBetweenPoints = np.sqrt( (self.xPositionsPoisson[idxOfPoints] - xPositionsForReference)**2 
                + (self.yPositionsPoisson[idxOfPoints] - yPositionsForReference)**2 
                + (self.zPositionsPoisson[idxOfPoints] - zPositionsForReference)**2 )
                ifInsideTheMinimumDistance = ( (distanceBetweenPoints < minimumDistanceBetweenPoints) & 
                                            (distanceBetweenPoints > 0) )
                # If the point is within the minimum distance range with at least one of the 
                # reference points, it is to be removed, i.e. its index is False
                if(any(ifInsideTheMinimumDistance)):
                    idsOfRetainedPoints[idxOfPoints] = False

            # Define the thinned Poisson-distributed points
            self.xPositions = self.xPositionsPoisson[idsOfRetainedPoints]
            self.yPositions = self.yPositionsPoisson[idsOfRetainedPoints]
            self.zPositions = self.zPositionsPoisson[idsOfRetainedPoints]    
            # The number of points that remain after thinning
            self.NumberOfThinnedPoints = len(self.xPositions)

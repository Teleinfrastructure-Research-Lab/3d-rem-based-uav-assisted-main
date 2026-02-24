# %%
import numpy as np
import random
import matplotlib.pyplot as plt
#%%
class MovementSimulation:

    def __init__(self):
        super().__init__()

    def RandomWalkGeneratedPoints(self, currentUEIdx, xStartingPoint, yStartingPoint, zStartingPoint, 
                                    heightMin = 0, heightMax = 2, LengthOfSimulatedRegion = 10, numberOfMovementIterations = 100):
        self.xPositionsMovingUEs = [xStartingPoint]
        self.yPositionsMovingUEs = [yStartingPoint]  
        self.zPositionsMovingUEs = [zStartingPoint]
        def movementDirection(idx, movementIterationIdx):
            rng = np.random.default_rng((idx + 5) * movementIterationIdx)
            maxMovementSpeed = 1  # in m/s for 3 km/h
            currentMovementSpeed = rng.uniform(low = 0.5, high = maxMovementSpeed)
            currentDirection = rng.uniform(low = 0, high = 360) # random number between 0 and 360 degrees
            currentDirectionHeight = rng.uniform(low = 0, high = 180) # random number between 0 and 180 degrees                       
            return currentMovementSpeed, currentDirection, currentDirectionHeight

        for movementIteration in range(1, numberOfMovementIterations):
            currentMovementSpeed, currentDirection, currentDirectionHeight = movementDirection(currentUEIdx, movementIteration)                      
            xPositionCurrent = self.xPositionsMovingUEs[movementIteration - 1] + (currentMovementSpeed * np.cos(currentDirection)) 
            yPositionCurrent = self.yPositionsMovingUEs[movementIteration - 1] + (currentMovementSpeed * np.sin(currentDirection)) 
            zPositionCurrent = self.zPositionsMovingUEs[movementIteration - 1] + (currentMovementSpeed * np.sin(currentDirectionHeight))

            # Corrections for all 3 dimensions, to make sure that the users do not cross out of the region
            idx_ = movementIteration
            while(zPositionCurrent < heightMin):
                # print('zPositionCurrent < heightMin', zPositionCurrent)
                currentMovementSpeed, currentDirection, currentDirectionHeight = movementDirection(currentUEIdx+idx_, movementIteration+5)
                zPositionCurrent = self.zPositionsMovingUEs[movementIteration - 1] - (currentMovementSpeed * np.sin(currentDirection)) 
                idx_ += 1
                if(zPositionCurrent > heightMin):
                    break
            while(zPositionCurrent >= heightMax):
                currentMovementSpeed, currentDirection, currentDirectionHeight = movementDirection(currentUEIdx+idx_, movementIteration+7)
                zPositionCurrent = self.zPositionsMovingUEs[movementIteration - 1] + (currentMovementSpeed * np.sin(180 - currentDirection))             
                idx_ += 1
                if(zPositionCurrent < heightMax):
                    break

            while(xPositionCurrent < -LengthOfSimulatedRegion/2):
                currentMovementSpeed, currentDirection, currentDirectionHeight = movementDirection(currentUEIdx+idx_, movementIteration+9)
                xPositionCurrent = self.xPositionsMovingUEs[movementIteration - 1] - (currentMovementSpeed * np.cos(currentDirection)) 
                idx_ += 1
                if(xPositionCurrent > -LengthOfSimulatedRegion/2):
                    break
            while(xPositionCurrent >= LengthOfSimulatedRegion/2):
                currentMovementSpeed, currentDirection, currentDirectionHeight = movementDirection(currentUEIdx+idx_, movementIteration+11)
                xPositionCurrent = self.xPositionsMovingUEs[movementIteration - 1] + (currentMovementSpeed * np.cos(180-currentDirection))           
                idx_ += 1
                if(xPositionCurrent < LengthOfSimulatedRegion/2):
                    break

            while(yPositionCurrent < -LengthOfSimulatedRegion/2):
                currentMovementSpeed, currentDirection, currentDirectionHeight = movementDirection(currentUEIdx+idx_, movementIteration+13)
                yPositionCurrent = self.yPositionsMovingUEs[movementIteration - 1] - (currentMovementSpeed * np.sin(currentDirection)) 
                idx_ += 1
                if(yPositionCurrent > -LengthOfSimulatedRegion/2):
                    break
            while(yPositionCurrent >= LengthOfSimulatedRegion/2):
                currentMovementSpeed, currentDirection, currentDirectionHeight = movementDirection(currentUEIdx+idx_, movementIteration+15)
                yPositionCurrent = self.yPositionsMovingUEs[movementIteration - 1] + (currentMovementSpeed * np.sin(180 - currentDirection))     
                idx_ += 1
                if(yPositionCurrent < LengthOfSimulatedRegion/2):
                    break
            self.zPositionsMovingUEs.append( zPositionCurrent )   
            self.yPositionsMovingUEs.append( yPositionCurrent ) 
            self.xPositionsMovingUEs.append( xPositionCurrent )

    
# %%

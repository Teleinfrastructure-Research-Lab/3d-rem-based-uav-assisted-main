import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from scipy.spatial.distance import cdist

def uav_positioning_with_gp(
    user_positions_3d,
    candidate_locations,
    coverage_radius,
    n_initial_samples=10,
    kappa=2.5
):
    """
    Selects the best UAV position using a Gaussian Process-based optimization approach,
    considering users in 3D space.

    This function trains a GP model on a small sample of the candidate locations to
    predict the coverage quality. It uses an Upper Confidence Bound (UCB) acquisition
    function to select the best location, balancing exploration and exploitation.

    Args:
        user_positions_3d (np.ndarray): Array of user positions, shape (N, 3) for (x, y, z).
        candidate_locations (np.ndarray): Array of all possible UAV positions, shape (M, 3).
        coverage_radius (float): The 3D spherical coverage radius of the UAV.
        n_initial_samples (int): The number of random locations to sample for training the GP.
                                 Must be less than the total number of candidate locations.
        kappa (float): The exploration-exploitation trade-off parameter for the UCB.

    Returns:
        tuple: A tuple containing the best UAV position (np.ndarray of shape (3,)),
               and the index of that position in the original candidate_locations array.
    """
    n_total_candidates = len(candidate_locations)
    if n_initial_samples >= n_total_candidates:
        raise ValueError("The number of initial samples must be smaller than the total number of candidate locations.")

    # --- 1. Sample and Evaluate (Create Training Data) ---
    # Select a random subset of candidate locations to evaluate
    sample_indices = np.random.choice(n_total_candidates, n_initial_samples, replace=False)
    X_train_3d = candidate_locations[sample_indices]
    X_train_2d = X_train_3d[:, :2]  # GP model still learns the 2D ground plane vs. score relationship

    # Calculate the "ground truth" coverage score for these samples
    y_train = []
    for uav_pos in X_train_3d:
        # MODIFICATION: Calculate direct 3D Euclidean distance
        distances = np.linalg.norm(user_positions_3d - uav_pos, axis=1)
        covered_count = np.sum(distances <= coverage_radius)
        y_train.append(covered_count)
    y_train = np.array(y_train)

    # --- 2. Train the Gaussian Process Model ---
    kernel = C(1.0, (1e-3, 1e3)) * RBF(100, (1e-2, 1e2))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=42
    )
    gp.fit(X_train_2d, y_train)

    # --- 3. Predict and Optimize using UCB ---
    all_locations_2d = candidate_locations[:, :2]
    predicted_mean, predicted_std = gp.predict(all_locations_2d, return_std=True)

    ucb_scores = predicted_mean + kappa * predicted_std
    best_location_idx = np.argmax(ucb_scores)
    best_location = candidate_locations[best_location_idx]

    return best_location_idx


def calculate_data_rate(channel_gain, tx_power_watts, bandwidth_hz, noise_power_watts):
    """Calculates data rate using the Shannon-Hartley theorem."""
    snr = (tx_power_watts * channel_gain) / noise_power_watts
    rate = bandwidth_hz * np.log2(1 + snr)
    return rate / 1e6  # Return rate in Mbps

def radiomap_placement_maxmin_rate(
    user_positions_3d,
    candidate_locations,
    radio_map,
    map_resolution,
    tx_power_dbm=24,
    bandwidth_mhz=20,
    noise_figure_db=9
):
    """
    Finds the best UAV position to maximize the minimum data rate among all users.

    Args:
        user_positions_3d (np.ndarray): Array of 3D user positions, shape (N, 3).
        candidate_locations (np.ndarray): Array of possible 3D UAV positions, shape (M, 3).
        radio_map (np.ndarray): The 3D Radio Map grid representing channel gain.
        map_resolution (int): The resolution (cell size) of the radio map in meters.
        tx_power_dbm (float): UAV's transmit power in dBm.
        bandwidth_mhz (float): System bandwidth in MHz.
        noise_figure_db (float): Receiver noise figure in dB.

    Returns:
        tuple: The best 3D UAV position and the guaranteed minimum data rate (Mbps).
    """
    best_location = None
    max_of_min_rates = -1.0

    # Convert system parameters from dB/MHz to linear scale/Hz
    tx_power_watts = 10**((tx_power_dbm - 30) / 10)
    bandwidth_hz = bandwidth_mhz * 1e6
    thermal_noise_dbm_per_hz = -174
    noise_power_dbm = thermal_noise_dbm_per_hz + 10 * np.log10(bandwidth_hz) + noise_figure_db
    noise_power_watts = 10**((noise_power_dbm - 30) / 10)

    map_shape = np.array(radio_map.shape)

    for uav_pos in candidate_locations:
        user_rates = []
        for user_pos in user_positions_3d:
            # Radio Map Lookup: Use UAV's coordinates to find the gain
            # This simplified model assumes gain is mainly dependent on UAV position
            uav_idx = np.floor(uav_pos / map_resolution).astype(int)
            
            # Boundary check
            if np.all(uav_idx >= 0) and np.all(uav_idx < map_shape):
                channel_gain = radio_map[tuple(uav_idx)]
                
                # Calculate the data rate for this user
                rate = calculate_data_rate(channel_gain, tx_power_watts, bandwidth_hz, noise_power_watts)
                user_rates.append(rate)
            else:
                user_rates.append(0) # Append 0 rate if UAV is outside map bounds
        
        # Find the minimum rate for the current UAV position
        if not user_rates: continue
        min_rate_for_this_pos = min(user_rates)
        
        # If this minimum rate is the best we've seen so far, update the best location
        if min_rate_for_this_pos > max_of_min_rates:
            max_of_min_rates = min_rate_for_this_pos
            best_location = uav_pos

    return best_location, max_of_min_rates

def exhaustiveSearch(user_positions, candidate_locations, coverage_radius):
    """
    Finds the best UAV position from a set of candidate locations to maximize user coverage.

    This function iterates through each possible UAV location and calculates the number of
    users it can cover based on a fixed coverage radius. The position that serves the
    most users is selected.

    Args:
        user_positions (np.ndarray): A NumPy array of shape (N, 2) where N is the
                                     number of users. Each row represents the (x, y)
                                     coordinates of a user.
        candidate_locations (np.ndarray): A NumPy array of shape (M, 3) where M is
                                          the number of candidate locations. Each row
                                          represents the (x, y, z) coordinates of a
                                          potential UAV position.
        coverage_radius (float): The radius of the circular coverage area on the ground
                                 provided by the UAV.

    Returns:
        tuple: A tuple containing the best UAV position (np.ndarray of shape (3,))
               and the number of users covered at that position (int). Returns
               (None, 0) if no users can be covered or inputs are empty.
    """
    if user_positions is None or candidate_locations is None or user_positions.size == 0 or candidate_locations.size == 0:
        return None, 0

    best_location = None
    max_covered_users = -1

    # Iterate through each candidate location for the UAV
    for uav_pos in candidate_locations:
        covered_users_count = 0
        uav_ground_pos = uav_pos[:3]  # Project UAV position to the ground (x, y, z)

        # Count how many users are within the coverage radius
        for user_pos in user_positions:
            # Calculate the 3D Euclidean distance between the user and the UAV's ground projection
            distance = np.linalg.norm(user_pos - uav_ground_pos)
            if distance <= coverage_radius:
                covered_users_count += 1

        # If the current location covers more users than the previous best, update it
        if covered_users_count > max_covered_users:
            max_covered_users = covered_users_count
            best_location = uav_pos
    
    best_location_idx = np.where(np.all(candidate_locations == best_location, axis=1))[0][0]
    return best_location_idx, best_location, max_covered_users

def weightedAverage(user_positions, candidate_locations, coverage_radius):
    """
    Finds the best UAV position based on a weighted score of covered users.

    This algorithm prioritizes users who are harder to cover. It first calculates a
    'rarity' weight for each user based on how many candidate locations can serve them.
    It then selects the UAV location that covers a set of users with the highest
    total combined weight.

    Args:
        user_positions (np.ndarray): A NumPy array of shape (N, 2) where N is the
                                     number of users. Each row represents the (x, y)
                                     coordinates of a user.
        candidate_locations (np.ndarray): A NumPy array of shape (M, 3) where M is
                                          the number of candidate locations. Each row
                                          represents the (x, y, z) coordinates of a
                                          potential UAV position.
        coverage_radius (float): The radius of the circular coverage area on the ground.

    Returns:
        tuple: A tuple containing the best UAV position (np.ndarray of shape (3,))
               and its corresponding total weighted score (float). Returns
               (None, 0) if inputs are invalid or no coverage is possible.
    """
    if user_positions is None or candidate_locations is None or user_positions.size == 0 or candidate_locations.size == 0:
        return None, 0

    num_users = user_positions.shape[0]
    user_weights = np.zeros(num_users)

    # --- Phase 1: Calculate user weights based on rarity ---
    for i, user_pos in enumerate(user_positions):
        # Count how many candidate locations can cover this user
        covering_locations_count = 0
        for uav_pos in candidate_locations:
            uav_ground_pos = uav_pos[:3]
            distance = np.linalg.norm(user_pos - uav_ground_pos)
            if distance <= coverage_radius:
                covering_locations_count += 1
        
        # The weight is the inverse of the count. If no location can cover the user, weight is 0.
        if covering_locations_count > 0:
            user_weights[i] = 1.0 / covering_locations_count
        else:
            user_weights[i] = 0

    # --- Phase 2: Find the location with the maximum weighted score ---
    best_location = None
    max_weighted_score = 0.00001

    for uav_pos in candidate_locations:
        current_weighted_score = 0.0
        uav_ground_pos = uav_pos[:3]
        
        # Sum the weights of all users covered by this location
        for i, user_pos in enumerate(user_positions):
            distance = np.linalg.norm(user_pos - uav_ground_pos)
            if distance <= coverage_radius:
                current_weighted_score += user_weights[i]
        
        # If this location has a higher score, it's the new best
        if current_weighted_score > max_weighted_score:
            max_weighted_score = current_weighted_score
            best_location = uav_pos
   
    best_location_idx = np.where(np.all(candidate_locations == best_location, axis=1))[0][0]
    return best_location_idx, best_location, max_weighted_score


def generate_2d_grid_encoding(grid_size:int, max_value:float):
    """ Generates a uniform 2D grid. """
    x = np.linspace(-max_value, max_value, grid_size)
    y = np.linspace(-max_value, max_value, grid_size)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    return np.stack([mesh_x.flatten(), mesh_y.flatten()])

def generate_3d_grid_encoding(LengthOfSimulatedRegion, HeightOfSimulatedRegion, numberOfPlanes, gridSizeOfPlane, startingHeight):
    xCoordinatesSpan = np.round(np.linspace(0, LengthOfSimulatedRegion, gridSizeOfPlane), 5)
    yCoordinatesSpan = np.round(np.linspace(0, LengthOfSimulatedRegion, gridSizeOfPlane), 5)
    zCoordinatesSpan = np.round(np.linspace(startingHeight, HeightOfSimulatedRegion, numberOfPlanes), 5)
    allEstimatedPoints = []
    for z in zCoordinatesSpan:
        for x in xCoordinatesSpan:
            for y in yCoordinatesSpan:
                allEstimatedPoints.append((x, y, z))
    allEstimatedPoints = np.array(allEstimatedPoints)
    return allEstimatedPoints.reshape(numberOfPlanes, int(allEstimatedPoints.shape[0]/numberOfPlanes), 3)


def generateCoordinates(xCoordinatesSpan, yCoordinatesSpan, zCoordinatesSpan,
                         pointsPerAxis, pointsForZAxis = 0):
    if(pointsForZAxis == 0):
        pointsForZAxis = pointsPerAxis
    startingCoordinate_x = np.amin(xCoordinatesSpan)
    endingCoordinate_x = np.amax(xCoordinatesSpan)
    startingCoordinate_y = np.amin(yCoordinatesSpan)
    endingCoordinate_y = np.amax(yCoordinatesSpan)
    # Coordinates of the bounding points of the grid
    point1 = (startingCoordinate_x, startingCoordinate_y)
    point2 = (startingCoordinate_x, endingCoordinate_y)
    point3 = (endingCoordinate_x, endingCoordinate_y)
    point4 = (endingCoordinate_x, startingCoordinate_y)

    # Generate equidistant points in the grid
    x_coords = np.linspace(point1[0], point3[0], pointsPerAxis)
    y_coords = np.linspace(point1[1], point3[1], pointsPerAxis)
    z_coords = np.linspace(np.amin(zCoordinatesSpan),
    np.amax(zCoordinatesSpan), pointsForZAxis)

    allEstimatedPoints = []
    for z in z_coords:
        for x in x_coords:
            for y in y_coords:
                allEstimatedPoints.append((x, y, z))
    allEstimatedPoints = np.array(allEstimatedPoints)
    return allEstimatedPoints

#Estimate the variogram
def estimateEmpiricalVariogram(KriPoints, measurementPoints, meanPowersAllPlanes, xEstimate, yEstimate, zEstimate):
    # Input data type consolidation
    if(type(yEstimate) != np.ndarray and type(xEstimate) == np.ndarray):
        yEstimate = np.array([yEstimate])
    if(type(xEstimate) != np.ndarray and type(yEstimate) == np.ndarray):
        xEstimate = np.array([xEstimate])
    if(type(xEstimate) != np.ndarray and type(yEstimate) != np.ndarray):
            xEstimate = np.array([xEstimate])
            yEstimate = np.array([yEstimate])
    
    # Calculate the distances between each of the estimated points and the measurement ones, 
    # and estimate the variogram for these points
    pointEstimates = np.zeros((0, 3)) 
    measurementPointsPower = meanPowersAllPlanes
    # Contain all variogram points distances and powers in separate lists for each variogram
    variogramDistances = []
    variogramPowers = []
    
    indices = []
    
    # Iterate over the axes of the estimated points
    for x, y, z in zip(xEstimate, yEstimate, zEstimate):
        # Form the estimated point in shape (1, 2)
        pointEstimate = np.column_stack((x, y, z))
        # Obtain the coordinates of all estimated points
        pointEstimates = np.append(pointEstimates, pointEstimate, axis=0)        
        
    # Contain all variogram points distances and powers in a numpy array     
    variogramPointsDistances = np.zeros(0)
    variogramPointsPowers = np.zeros(0)
    
    distances = np.zeros((pointEstimates.shape[0], measurementPoints.shape[0]))
    # Iterate over the coordinates of all estimated points
    for pointsIndex in range(0, pointEstimates.shape[0]):    
        # Find the distance between the estimated points and the measurement ones
        distances = np.linalg.norm(pointEstimates[pointsIndex] - measurementPoints, axis=1)
        #([pointEstimates[pointsIndex]], measurementPoints)[0]
        
        # Find the indices that represent KriPoints measurement points that are closest to 
        # the estimated points      
        desiredIndices = np.argsort(distances)[0:KriPoints]  
        indices.append(desiredIndices)
        # Get the distances from the origin that correspond to the points closest
        # from the estimated point 
        currentDistances = distances[desiredIndices]            
        currentPowers = measurementPointsPower[desiredIndices]
      
        distancesBetweenKrigingPoints = np.ceil(np.linalg.norm(pointEstimates[pointsIndex] - measurementPoints, axis=1))

        currentVariogramDistances = np.unique(distancesBetweenKrigingPoints)
        currentVariogramPowers = []
        for distanceIndex in range(0, currentVariogramDistances.shape[0]):
            uniqueIndices = np.where(distancesBetweenKrigingPoints == currentVariogramDistances[distanceIndex])[0]
            if(uniqueIndices.shape[0] > 1):
                currentVariogramPointValue = 0.5 * np.mean(np.power(np.diff(measurementPointsPower[uniqueIndices]), 2)) 
            else:
                currentVariogramPointValue = np.power(measurementPointsPower[uniqueIndices], 2)
        #print("currentVariogramPointValue: ", currentVariogramPointValue)
        currentVariogramPowers.append(currentVariogramPointValue)
        variogramDistances.append(currentVariogramDistances)
        variogramPowers.append(currentVariogramPowers)       
    return variogramDistances, variogramPowers, indices, pointEstimates, variogramPointsDistances, variogramPointsPowers

def weightedAverage(user_positions, candidate_locations, coverage_radius):
    """
    Finds the best UAV position based on a weighted score of covered users.

    This algorithm prioritizes users who are harder to cover. It first calculates a
    'rarity' weight for each user based on how many candidate locations can serve them.
    It then selects the UAV location that covers a set of users with the highest
    total combined weight.

    Args:
        user_positions (np.ndarray): A NumPy array of shape (N, 2) where N is the
                                     number of users. Each row represents the (x, y)
                                     coordinates of a user.
        candidate_locations (np.ndarray): A NumPy array of shape (M, 3) where M is
                                          the number of candidate locations. Each row
                                          represents the (x, y, z) coordinates of a
                                          potential UAV position.
        coverage_radius (float): The radius of the circular coverage area on the ground.

    Returns:
        tuple: A tuple containing the best UAV position (np.ndarray of shape (3,))
               and its corresponding total weighted score (float). Returns
               (None, 0) if inputs are invalid or no coverage is possible.
    """
    if user_positions is None or candidate_locations is None or user_positions.size == 0 or candidate_locations.size == 0:
        return None, 0

    num_users = user_positions.shape[0]
    user_weights = np.zeros(num_users)

    # --- Phase 1: Calculate user weights based on rarity ---
    for i, user_pos in enumerate(user_positions):
        # Count how many candidate locations can cover this user
        covering_locations_count = 0
        for uav_pos in candidate_locations:
            uav_ground_pos = uav_pos[:3]
            distance = np.linalg.norm(user_pos - uav_ground_pos)
            if distance <= coverage_radius:
                covering_locations_count += 1
        
        # The weight is the inverse of the count. If no location can cover the user, weight is 0.
        if covering_locations_count > 0:
            user_weights[i] = 1.0 / covering_locations_count
        else:
            user_weights[i] = 0

    # --- Phase 2: Find the location with the maximum weighted score ---
    best_location = None
    max_weighted_score = 0.00001

    for uav_pos in candidate_locations:
        current_weighted_score = 0.0
        uav_ground_pos = uav_pos[:3]
        
        # Sum the weights of all users covered by this location
        for i, user_pos in enumerate(user_positions):
            distance = np.linalg.norm(user_pos - uav_ground_pos)
            if distance <= coverage_radius:
                current_weighted_score += user_weights[i]
        
        # If this location has a higher score, it's the new best
        if current_weighted_score > max_weighted_score:
            max_weighted_score = current_weighted_score
            best_location = uav_pos
   
    best_location_idx = np.where(np.all(candidate_locations == best_location, axis=1))[0][0]
    return best_location_idx, best_location, max_weighted_score


# Find the sill, nugget and range of the empirical variogram
def FindVariogramParameters(variogramPowers,variogramDistances):
    nugget_ = np.zeros(len(variogramPowers))
    sill_ = np.zeros(len(variogramPowers))
    range_ = np.zeros(len(variogramPowers))
    index = 0
    for currentVariogramPowers in variogramPowers:
        #print(currentVariogramPowers)
        nugget_[index] = currentVariogramPowers[0] 
        sill_[index] = np.max(currentVariogramPowers)
        
        # The range is computed from the element after the maximum value
        # unless that value is the last element, in which case the range is equal to the sill
        if(np.argmax(currentVariogramPowers) == len(currentVariogramPowers)-1):
            range_[index] = variogramDistances[index][np.argmax(currentVariogramPowers)]
        else:
            range_[index] = variogramDistances[index][np.argmax(currentVariogramPowers)+1]
        index+=1
      
    return nugget_, sill_, range_

# Function for covariance computation under a specific distribution with smoothness parameter nu
def covarianceVariogramEstimator(nugget_, sill_, range_, distances, nu=0.5):
    # Exponential
    covariance = nugget_ + ( sill_ * np.exp( - (distances/range_) ) ) 

    return covariance

# Solve the Kriging interpolation for the estimated points
def KrigingInterpolation(meanPowersAllPlanes, variogramPowers, nugget_, sill_, range_,
                         variogramDistances, measurementPoints, indices, pointEstimates, nu, seed):
    
    estimatedPower_p = np.zeros(len(pointEstimates))   
    std_K = np.zeros(len(pointEstimates))
    weights = []
    distances = []
    validIndices = []
    rng = np.random.default_rng(seed=int(123+seed+np.ceil(nu*(np.mean(sill_)+1)))) #sill_[0]
    for estimatedPointIndex in range(0, len(pointEstimates)):
        covarianceBetweenMeasuredPoints = np.zeros((len(measurementPoints[indices[estimatedPointIndex]])+1,
                                                    len(measurementPoints[indices[estimatedPointIndex]])+1))
        
        # For each measured point, estimate the covariance between it and the others,
        # to obtain the measured points matrix (C1)
        currentDistances = []
        for measurementPointIndex in range(0, len(indices[estimatedPointIndex])):
            distancesBetweenMeasuredPoints = np.linalg.norm(measurementPoints[indices[estimatedPointIndex][
                measurementPointIndex]]-measurementPoints[indices[estimatedPointIndex]], axis=1)
            covarianceBetweenMeasuredPoints[measurementPointIndex, 0:-1] = covarianceVariogramEstimator(
                                            nugget_[estimatedPointIndex],
                                            sill_[estimatedPointIndex],
                                            range_[estimatedPointIndex],
                                            distancesBetweenMeasuredPoints, nu)            

        # Fill the last row and last column with ones and the very last element as 0 (from definition)
        covarianceBetweenMeasuredPoints[len(measurementPoints[indices[estimatedPointIndex]]), 0:-1] = 1
        covarianceBetweenMeasuredPoints[0:-1, len(measurementPoints[indices[estimatedPointIndex]])] = 1
        covarianceBetweenMeasuredPoints[len(measurementPoints[indices[estimatedPointIndex]]),
                                        len(measurementPoints[indices[estimatedPointIndex]])] = 0
     
        distancesBetweenMeasuredAndEstimatedPoints = np.linalg.norm(measurementPoints[indices[estimatedPointIndex]
                                                            ]-pointEstimates[estimatedPointIndex], axis=1 )
        covarianceBetweenEstimatedAndMeasuredPoints = np.zeros((len(measurementPoints[indices[estimatedPointIndex]])+1,
                                                               1))
        # Compute the covariance between the estimated point and the measured ones to obtain the vector c0
        covarianceBetweenEstimatedAndMeasuredPoints[0:-1, 0] = covarianceVariogramEstimator(
                                                nugget_[estimatedPointIndex],
                                                sill_[estimatedPointIndex],
                                                range_[estimatedPointIndex],
                                                distancesBetweenMeasuredAndEstimatedPoints,
                                                nu)

        # The last element is 1 (from definition)
        covarianceBetweenEstimatedAndMeasuredPoints[len(measurementPoints[indices[estimatedPointIndex]]), 0] = 1

        # if the covarianceBetweenMeasuredPoints matrix is not invertible, add some random noise
        if(np.linalg.det(covarianceBetweenMeasuredPoints) == 0):
#             print("covarianceBetweenMeasuredPoints matrix is not invertible")
            covarianceBetweenMeasuredPoints[0:-1, 0:-1] = np.multiply(covarianceBetweenMeasuredPoints[0:-1, 0:-1],#np.mean(covarianceBetweenMeasuredPoints[0:-1, 0:-1]),
                                                              rng.random((len(measurementPoints[indices[estimatedPointIndex]]),
                                                                          len(measurementPoints[indices[estimatedPointIndex]])))*0.1)
            
        # Compute the weights vector
        currentWeights = np.matmul( np.linalg.inv(covarianceBetweenMeasuredPoints),
                                   covarianceBetweenEstimatedAndMeasuredPoints )
        currentWeights = np.reshape(currentWeights, (currentWeights.shape[0], 1))
        measurementPointsPowers = np.append(meanPowersAllPlanes[indices[estimatedPointIndex]], [0]) 
                
        # Compute the value of the current estimated point
        estimatedPower_p[estimatedPointIndex] = np.abs(np.matmul( measurementPointsPowers.T, currentWeights ))
        weights.append(currentWeights)
        
        # Compute the Kriging variance
        # The covariance coeff is the covariance between the 1st element and itself so the distance is 0
        covarianceCoefficient = covarianceVariogramEstimator(nugget_[estimatedPointIndex],
                                                             sill_[estimatedPointIndex],
                                                             range_[estimatedPointIndex], 0, nu)
        firstComponentOfVariance = np.matmul( covarianceBetweenEstimatedAndMeasuredPoints.T,
                                                  np.linalg.inv(covarianceBetweenMeasuredPoints) )
        std_K[estimatedPointIndex] = np.sqrt( np.abs( covarianceCoefficient - np.matmul( firstComponentOfVariance,
                                                  covarianceBetweenEstimatedAndMeasuredPoints ) ) )
        
        # Estimated value validity check and consolidation
        # if its value is not valid, it will be estimated as the mean of the values
        # of the "KriPoints" points closest to it
        if(estimatedPower_p[estimatedPointIndex] > np.amax(meanPowersAllPlanes)*2 or estimatedPower_p[estimatedPointIndex] <  1e-15):
            estimatedPower_p[estimatedPointIndex] = np.mean(meanPowersAllPlanes[np.argsort(
                np.linalg.norm( pointEstimates[estimatedPointIndex] - measurementPoints[indices[estimatedPointIndex]], axis=1 ))[0:10] ] )
        if((np.sum(currentWeights[0:-1]) > 1.1 or np.sum(currentWeights[0:-1]) < 0.9) or estimatedPower_p[estimatedPointIndex] < 1e-15):
            estimatedPower_p[estimatedPointIndex] = np.mean(meanPowersAllPlanes[np.argsort(
            np.linalg.norm( pointEstimates[estimatedPointIndex] - measurementPoints[indices[estimatedPointIndex]] ))[0:10] ] )
            if(estimatedPower_p[estimatedPointIndex] > np.amax(meanPowersAllPlanes)):    
                estimatedPower_p[estimatedPointIndex] = np.mean(meanPowersAllPlanes[np.argsort(
                np.linalg.norm( pointEstimates[estimatedPointIndex] - measurementPoints[indices[estimatedPointIndex]] ))[0:10] ] )

    return estimatedPower_p, std_K, weights, distances

# Inverse Distance Weighting interpolation function
def inverseDistanceWeighting(meanPowersAllPlanes, xEstimate, yEstimate, zEstimate,
                             weightDeclineCoeffient, measurementPoints): 
    # Input data type consolidation
    if(type(yEstimate) != np.ndarray and type(xEstimate) == np.ndarray):
        yEstimate = np.array([yEstimate])
    if(type(xEstimate) != np.ndarray and type(yEstimate) == np.ndarray):
        xEstimate = np.array([xEstimate])
    if(type(xEstimate) != np.ndarray and type(yEstimate) != np.ndarray):
            xEstimate = np.array([xEstimate])
            yEstimate = np.array([yEstimate])
            
    # Calculate the distances between each of the estimated points and the measurement ones, 
    # and estimate the power at these points
    pointEstimates = np.zeros((0, 3)) 
    measurementPointsPower = meanPowersAllPlanes
    # Iterate over the axes of the estimated points
    for x, y, z in zip(xEstimate, yEstimate, zEstimate):
        # Form the estimated point in shape (1, 2)
        pointEstimate = np.column_stack((x, y, z))
        # Obtain the coordinates of all estimated points
        pointEstimates = np.append(pointEstimates, pointEstimate, axis=0)
    
    distances = np.zeros((pointEstimates.shape[0], measurementPoints.shape[0]))
    estimatedPower_p = np.zeros((pointEstimates.shape[0]))   
    # Iterate over the coordinates of all estimated points
    for pointsIndex in range(0, pointEstimates.shape[0]):
        # Find the distance between the estimated points and the measurement ones
        distances[pointsIndex, :] = cdist([pointEstimates[pointsIndex]], measurementPoints)
        # Calculate the the power at the estimated points                        
        estimatedPower_p[pointsIndex] = np.sum( np.power(distances[pointsIndex,
                                np.where(distances[pointsIndex, :] != 0)[0] ],
                                weightDeclineCoeffient) *  measurementPointsPower[np.where(distances[pointsIndex, :] != 0)[0]] ) / np.sum( np.power(distances[pointsIndex, np.where(distances[pointsIndex, :] != 0)[0] ], weightDeclineCoeffient) )
    return estimatedPower_p, pointEstimates

test = False
if (test == True):
    # Generate the locations of the in-between points
    # Number of points along each axis
    pointsPerAxis = 16
    xCoordinatesSpan = np.linspace(0, 150, pointsPerAxis)
    yCoordinatesSpan = np.linspace(0, 150, pointsPerAxis)
    zCoordinatesSpan = np.linspace(10, 15, pointsPerAxis)
    allMeasurementPoints = generateCoordinates(xCoordinatesSpan, 
                            yCoordinatesSpan, zCoordinatesSpan, pointsPerAxis)
    allMeasurementPoints.shape
    ''

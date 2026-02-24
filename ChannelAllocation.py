# %%
import numpy as np

# %%
# Class which performs allocations of a defined number of channels and on the basis of the proposed
# method's results in the previous iteration. A random allocation is used as a baseline.
class ChannelAllocation:

    def __init__(self, seed = 1, resultFromPreviousIteration = 0):
        self.resultFromPreviousIteration = resultFromPreviousIteration # The proposed method's results
                                                                       # from the previous iteration.
        self.seed = seed # Seed for the random channel allocation

    # Determines which channels are allocated for each connection ID
    def _DetermineAllocatedChannelIDs(self, connectionIDs, listOfUnoccupiedChannelsIDs, maxNumberOfChannels):
        self.listOfChannelAllocations = [] # Stores the connection ID and its allocated channels IDs
        maxNumberOfInterferingChannelsToAssign = int(len(listOfUnoccupiedChannelsIDs) / len(
                                                        connectionIDs) ) + 2
        self.maxNumberOfChannels = maxNumberOfChannels  #The max number from whichto choose 
                                                        #in case the channels run out
        np.random.seed(self.seed) 
        self.debugArray = []
        np.random.shuffle(connectionIDs)
        # Loop through each connection and assign to it channels IDs
        for currentConnectionID in connectionIDs:
            # If all channels are already allocated, assign random channel IDs without regard
            # that they already in use by other connections. The allocated channels will
            # thus be sources of interference.      
            if( len(listOfUnoccupiedChannelsIDs) > maxNumberOfInterferingChannelsToAssign ):
                    # Choose a random number of channels for this connection out of all.
                    self.numberOfChannelsForCurrentConnection = np.random.choice( range(1, 
                                                    maxNumberOfInterferingChannelsToAssign) )
                                                #2 #np.random.choice( range( 1, len(listOfUnoccupiedChannelsIDs) ) )
                    # Choose a random channels for this connection out of all channels. Generates only 
                    # unique channel IDs.                                         
                    self.channelIDs = np.random.choice(listOfUnoccupiedChannelsIDs,
                                     self.numberOfChannelsForCurrentConnection, replace = False ) 
            
            # Store the current connection ID and a list of its allocated channel IDs     
            elif( len(listOfUnoccupiedChannelsIDs) == 1 ): 
                # Choose a random number of channels for this connection out of all.
                self.numberOfChannelsForCurrentConnection = 1
                # Choose that many random channels for this connection out of all channels. Generates only 
                # unique channel IDs. 
                self.channelIDs = np.array([listOfUnoccupiedChannelsIDs[0]])                            
            elif(len(listOfUnoccupiedChannelsIDs) <= 0): #or 
                # Choose a random number of channels for this connection out of all.
                self.numberOfChannelsForCurrentConnection = np.random.choice( range(1,
                                                             maxNumberOfInterferingChannelsToAssign)  )
                                                            #np.random.choice( range(1, self.numberOfChannels) )#2 #
                
                # Choose that many random channels for this connection out of all channels. Generates only 
                # unique channel IDs. 
                self.channelIDs = np.random.choice( np.arange(0, self.maxNumberOfChannels),
                                        self.numberOfChannelsForCurrentConnection, replace = False )
            # If there are more than 1 unused channels, they are allocated here.                                        
            elif( len(listOfUnoccupiedChannelsIDs) <= maxNumberOfInterferingChannelsToAssign ):
                # Choose a random number of channels for this connection out of all.
                self.numberOfChannelsForCurrentConnection = np.random.choice( range(1,
                                                             len(listOfUnoccupiedChannelsIDs)) )
                # Choose a random channels for this connection out of all channels. Generates only 
                # unique channel IDs.                                         
                self.channelIDs = np.random.choice(listOfUnoccupiedChannelsIDs,
                                    self.numberOfChannelsForCurrentConnection, replace = False )                 

            self.listOfChannelAllocations.append([currentConnectionID, self.channelIDs])

            # If there are yet unoccupied channels, at the end of each step (coonection ID) in the loop,
            # remove the channel IDs allocated during this step from the list of unoccupied channel IDs. 
            if(len(listOfUnoccupiedChannelsIDs) > 0):
                for allocatedChannelIDs in self.channelIDs:
                    listOfUnoccupiedChannelsIDs = np.delete(listOfUnoccupiedChannelsIDs,
                            np.where(listOfUnoccupiedChannelsIDs == allocatedChannelIDs)[0])
                    self.debugArray.append(allocatedChannelIDs)


    def DetermineAllocatedChannelIDs(self, listOfUnoccupiedChannelsIDs, connectionIDs, _):
        """
        Match all elements of two numpy arrays at random, 
        considering that one array has more elements than the other.
        """
        self.debugArray = []
        np.random.shuffle(connectionIDs)
        # Get the lengths of the arrays
        numberOfChannels = len(listOfUnoccupiedChannelsIDs)

        # Initialize empty lists to store the matched elements
        matchedChannels = []
        matchedConnectionIDs = []
        numberOfConnections = connectionIDs.shape[0]
        # Iterate over the shorter array
        for currentConnectionID in connectionIDs:
            # Randomly select a number of elements to match from the longer array
            numberOfMatches = np.random.randint(1, int((numberOfChannels - len(matchedChannels))/numberOfConnections) + 2)
            if(numberOfMatches > numberOfChannels):
                numberOfMatches = numberOfChannels
            remainingChannelIds = np.setdiff1d(listOfUnoccupiedChannelsIDs, matchedChannels)
            if(remainingChannelIds.shape[0] == 0):
                break
            # Select the elements to match
            matches = np.random.choice(remainingChannelIds, numberOfMatches, replace=False)
            # Add the matches to the lists
            matchedChannels.extend(matches)
            matchedConnectionIDs.extend([currentConnectionID] * numberOfMatches)

        # Match any remaining elements of the longer array with a random element of the shorter array
        remainingChannelIds = np.setdiff1d(listOfUnoccupiedChannelsIDs, matchedChannels)
        for channelIds in remainingChannelIds:
            matchedChannels.append(channelIds)
            matchedConnectionIDs.append(np.random.choice(connectionIDs, replace=False))

        # Convert the lists to numpy arrays
        matchedChannels = np.array(matchedChannels)
        matchedConnectionIDs = np.array(matchedConnectionIDs)
 
        self.listOfChannelAllocations = []
        for currentConnectionID in connectionIDs:
            currentConnectionIDIds = np.where(matchedConnectionIDs == currentConnectionID)[0]
            self.listOfChannelAllocations.append([np.unique(matchedConnectionIDs[currentConnectionIDIds])[0],
                                                   np.array([matchedChannels[currentConnectionIDIds]])[0]])
        self.debugArray = np.sort(matchedChannels)

# arr1 = np.array([12, 15, 21, 10, 13, 30, 25, 18, 55, 26, 37])
# arr2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# _ = ChannelAllocation(5)
# one, two = _.DetermineAllocatedChannelIDs(arr1, arr2)
# print(one, two)

# Test script for this class
channelAllocationTestMode = False
if(channelAllocationTestMode):
    connectionIDs = np.arange(0, 10)
    listOfUnoccupiedChannelsIDs = np.arange(0, 101) 
    seed = 1
    performChannelAllocation = ChannelAllocation(len(listOfUnoccupiedChannelsIDs), seed)
    performChannelAllocation.DetermineAllocatedChannelIDs(connectionIDs, listOfUnoccupiedChannelsIDs)
    listOfChannelAllocations = performChannelAllocation.listOfChannelAllocations

    # %%
    interferenceList = []
    for currentConnectionIDIdx in range(0, len(listOfChannelAllocations)): 
        # findTheConnectionsWhichOperateOnThisChannelAndTheirSINR 

        currentChannelsIDs = listOfChannelAllocations[currentConnectionIDIdx][1]
        # Check if and which channels allocated to this connection, are used in others.
        # They are interference.
        
        listOfInterferers = []
        for idx in range(0, len(listOfChannelAllocations)):
            if(idx != currentConnectionIDIdx):
                listOfInterferers.append(listOfChannelAllocations[idx])

        interferenceForCurrentConnection = 0
        for currentInterferingConnectionID in listOfInterferers: 
            currentInterferingChannels = []
            #if(currentInterferingConnectionIDIdx != currentConnectionIDIdx):
            numberOfInterferingChannelsForCurrentInterferingID = np.count_nonzero( np.in1d(currentInterferingConnectionID[1],
                                                            currentChannelsIDs) )

            interferenceList.append([currentConnectionIDIdx, numberOfInterferingChannelsForCurrentInterferingID]) # temp
        ''
    
    ''

# %%

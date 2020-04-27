import torch
import itertools

# A function to calculate the cycle decomposition of a permutation sigma[]
# Input sigma[] is a proper permutation of any length
def cycleDecomp(sigma):
    # First, test input for correctness:
    a = [sigma.index(i) for i in range(len(sigma))]
    # Any improprieties in seq[] will cause Python to throw an error 
    #   when it can't find an index

    # Now form the fundamental set:
    fundSet = list(range(len(sigma)))

    # initialize some things:
    parity = 1
    cycleList = []
    # Start each cycle with an element from the fundamental set
    for start in fundSet:
        # a loop to establish the cycle; 1 loop per cycle
        thisCycle = [start]
        for step in thisCycle:
            # We start by mapping fundSet[start] -> sigma[start], which 
            #   generalizes to fundSet[step] -> sigma[step] after the first loop.

            # If at any point, including the first loop, 
            #   seq[step] = start for this loop, then the 
            #   cycle is closed and added to cycleList
            if sigma[step] == start:
                # Append the cycle to cycleList
                cycleList.append(thisCycle)
            else:
                # Add seq[step] to the cycle
                thisCycle.append(sigma[step])
                # Now, the current loop should proceed to step = seq[step]

                # Now remove seq[step] from fundSet[], since we won't need 
                #   to consider it again later as a "start"
                fundSet.remove(sigma[step])
            
        # the length of identified cycle is related to the parity of the permutations.
        # Each elementary 2-cycle is achieved by a single swap. The parity is defined
        # as the 
        #              parity(sigma) := (-1)^(#swaps mod 2),
        #
        # hence, for odd number of swaps the parity is -1 while for even number of swaps
        # its +1. Each cycle can be realized by len(cycle)-1 swaps.
        parity *= (-1)**(len(thisCycle)-1)
      
    return cycleList, parity

# define canonical completely anti-symmetric tensor
def levi_civita_3D(dtype=torch.float64,device='cpu'):
    lc3D=torch.zeros((3,3,3),dtype=dtype,device=device)
    for p in itertools.permutations(tuple(range(3))):
        lc3D[p]= cycleDecomp(p)[1]

    return lc3D
import torch
import json
from args import PEPSARGS, GLOBALARGS

class IPEPS(torch.nn.Module):
    def __init__(self, sites, vertexToSite, peps_args=PEPSARGS(), global_args=GLOBALARGS()):
        
        super(IPEPS, self).__init__()
        self.dtype = global_args.dtype
        self.device = global_args.device

        # Dict of non-equivalent on-site tensors 
        #
        # A B                     A B C
        # B A results in {A, B};  C A B results in {A,B,C}; etc
        #
        #                                             u s 
        #                                             |/ 
        # each site has indices in following order l--A--r  <=> A[s,u,l,d,r]
        #                                             |
        #                                             d
        # (anti-clockwise direction)
        self.sites = sites
        
        # A mapping function from coord on a square lattice
        # to one of the on-site tensor <key>
        #
        # y\x -2 -1 0 1 2 3  =>    1 2 0 1 2 0
        #  -2  A  B C A B C  =>  1 A B C A B C
        #  -1  B  C A B C A  =>  2 B C A B C A
        #   0  C  A B C A B  =>  0 C A B C A B
        #   1  A  B C A B C  =>  1 A B C A B C
        #   2  B  C A B C A  =>  2 B C A B C A
        #
        # given (x,y) pair on square lattice, apply appropriate
        # transformation and returns <key> of the appropriate 
        # tensor
        self.vertexToSite = vertexToSite

    # return site at coord=(x,y)
    def site(self, coord):
        return self.sites[self.vertexToSite(coord)]

# WARNING a simple PBC vertexToSite function is used by default
def read_ipeps(jsonfile, vertexToSite=None, peps_args=PEPSARGS(), global_args=GLOBALARGS()):

    sites = dict()
    
    with open(jsonfile) as j:
        raw_state = json.load(j)

        # Loop over non-equivalent tensor,site pairs in the unit cell
        for ts in raw_state["map"]:
            coord = (ts["x"],ts["y"])

            # find the corresponding tensor (and its elements) 
            # identified by "siteId" in the "sites" list
            t = None
            for s in raw_state["sites"]:
                if s["siteId"] == ts["siteId"]:
                    t = s
            if t == None:
                raise Exception("Tensor with siteId: "+ts["sideId"]+" NOT FOUND in \"sites\"") 

            # 0) find the dimensions of auxiliary indices
            # branch 1: key "auxInds" exists

            # branch 2: key "auxInds" does not exist, all auxiliary 
            # indices have the same dimension
            X = torch.zeros((t["physDim"], t["auxDim"], t["auxDim"], \
                t["auxDim"], t["auxDim"]), dtype=global_args.dtype, device=global_args.device)

            # 1) fill the tensor with elements from the list "entries"
            # which list the non-zero tensor elements in the following
            # notation. Dimensions are indexed starting from 0
            # 
            # index (integer) of physDim, left, up, right, down, (float) Re, Im  
            for entry in t["entries"]:
                l = entry.split()
                X[int(l[0]),int(l[2]),int(l[1]),int(l[4]),int(l[3])]=float(l[5])

            sites[coord]=X

        # Unless given, construct a function mapping from
        # any site of square-lattice back to unit-cell
        if vertexToSite == None:
            # check for legacy keys
            lX = 0
            lY = 0
            if "sizeM" in raw_state:
                lX = raw_state["sizeM"]
            else:
                lX = raw_state["lX"]
            if "sizeN" in raw_state:
                lY = raw_state["sizeN"]
            else:
                lY = raw_state["lY"]

            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*lX)%lX, (y + abs(y)*lY)%lY )

    return IPEPS(sites, vertexToSite, peps_args, global_args)
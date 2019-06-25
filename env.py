import torch
from args import args
from ipeps import IPEPS

class ENV(torch.nn.Module):
    def __init__(self, args, ipeps, dtype=torch.float64, device='cpu', use_checkpoint=False):
        super(ENV, self).__init__()
        self.use_checkpoint = use_checkpoint
        
        # 
        self.chi = args.chi

        # initialize environment tensors
        self.C = dict()
        self.T = dict()

        # for each pair (coord, site) create corresponding T's and C's
        # y\x -1 0 1
        #  -1  C T C
        #   0  T A T
        #   1  C T C 
        # where the directional vectors are given as coord(env-tensor) - coord(A)
        # C(-1,-1)   T        (1,-1)C 
        #            |(0,-1)
        # T--(-1,0)--A(0,0)--(1,0)--T 
        #            |(0,1)
        # C(-1,1)    T         (1,1)C
        # and analogously for corners C
        #
        # The dimension-position convention is as follows: 
        # Start from index in direction "up" <=> (0,-1) and
        # continue anti-clockwise
        # 
        # C--1 0--T--2 0--C
        # |       |       |
        # 0       1       1
        # 0               0
        # |               |
        # T--2         1--T
        # |               |
        # 1               2
        # 0       0       0
        # |       |       |
        # C--1 1--T--2 1--C
        for coord, site in ipeps.sites.items():
            #for vec in [(0,-1), (-1,0), (0,1), (1,0)]:
            #    self.T[(coord,vec)]="T"+str(ipeps.site(coord))
            self.T[(coord,(0,-1))]=torch.empty(self.chi,site.size()[1]*site.size()[1],self.chi)
            self.T[(coord,(-1,0))]=torch.empty(self.chi,self.chi,site.size()[2]*site.size()[2])
            self.T[(coord,(0,1))]=torch.empty(site.size()[3]*site.size()[3],self.chi,self.chi)
            self.T[(coord,(1,0))]=torch.empty(self.chi,site.size()[4]*site.size()[4],self.chi)

            #for vec in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            #     self.C[(coord,vec)]="C"+str(ipeps.site(coord))
            for vec in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                self.C[(coord,vec)]=torch.empty(self.chi,self.chi)

def init_random(env):
    for key,t in env.C.items():
        env.C[key] = torch.rand(t.size())
    for key,t in env.T.items():
        env.T[key] = torch.rand(t.size())
    return env

def print_env(env):
    for key,t in env.C.items():
        print(str(key)+" "+str(t.size()))
        #print(t)
    for key,t in env.T.items():
        print(str(key)+" "+str(t.size()))
        #print(t)

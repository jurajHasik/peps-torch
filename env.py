import torch
from args import CTMARGS, GLOBALARGS
from ipeps import IPEPS

class ENV(torch.nn.Module):
    def __init__(self, chi, ipeps, ctm_args=CTMARGS(), global_args=GLOBALARGS()):
        super(ENV, self).__init__()
        self.dtype = global_args.dtype
        self.device = global_args.device
        
        # 
        self.chi = chi

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
            self.T[(coord,(0,-1))]=torch.empty((self.chi,site.size()[1]*site.size()[1],self.chi), dtype=self.dtype, device=self.device)
            self.T[(coord,(-1,0))]=torch.empty((self.chi,self.chi,site.size()[2]*site.size()[2]), dtype=self.dtype, device=self.device)
            self.T[(coord,(0,1))]=torch.empty((site.size()[3]*site.size()[3],self.chi,self.chi), dtype=self.dtype, device=self.device)
            self.T[(coord,(1,0))]=torch.empty((self.chi,site.size()[4]*site.size()[4],self.chi), dtype=self.dtype, device=self.device)

            #for vec in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            #     self.C[(coord,vec)]="C"+str(ipeps.site(coord))
            for vec in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                self.C[(coord,vec)]=torch.empty((self.chi,self.chi), dtype=self.dtype, device=self.device)

def init_env(ipeps, env, ctm_args=CTMARGS()):
    if ctm_args.ctm_env_init_type=='CONST':
        init_const(env)
    elif ctm_args.ctm_env_init_type=='RANDOM':
        init_random(env)
    elif ctm_args.ctm_env_init_type=='CTMRG':
        init_from_ipeps(ipeps, env)
    else:
        raise ValueError("Invalid environment initialization: "+str(ctm_args.ctm_env_init_type))

def init_const(env, verbosity=0):
    for key,t in env.C.items():
        env.C[key] = torch.ones(t.size(), dtype=env.dtype, device=env.device)
    for key,t in env.T.items():
        env.T[key] = torch.ones(t.size(), dtype=env.dtype, device=env.device)

# TODO restrict random corners to have pos-semidef spectrum
def init_random(env, verbosity=0):
    for key,t in env.C.items():
        env.C[key] = torch.rand(t.size(), dtype=env.dtype, device=env.device)
    for key,t in env.T.items():
        env.T[key] = torch.rand(t.size(), dtype=env.dtype, device=env.device)

# TODO finish by initializing Ts as well
def init_from_ipeps(ipeps, env, verbosity=0):
    for coord, site in ipeps.sites.items():
        # Left-upper corner
        #
        #     i      = C--1     
        # j--A--3      0
        #   /\
        #  2  m
        #      \ i
        #    j--A--3
        #      /
        #     2
        vec = (-1,-1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('mijef,mijab->eafb',(A,A)).contiguous().view(dimsA[3]**2, dimsA[4]**2)
        env.C[(coord,vec)] = a

        # right-upper corner
        #
        #     i      = 0--C     
        # 1--A--j         1
        #   /\
        #  2  m
        #      \ i
        #    1--A--j
        #      /
        #     2
        vec = (1,-1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('miefj,miabj->eafb',(A,A)).contiguous().view(dimsA[2]**2, dimsA[3]**2)
        env.C[(coord,vec)] = a

        # right-lower corner
        #
        #     0      =    0     
        # 1--A--j      1--C
        #   /\
        #  i  m
        #      \ 0
        #    1--A--j
        #      /
        #     i
        vec = (1,1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('mefij,mabij->eafb',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2)
        env.C[(coord,vec)] = a

        # left-lower corner
        #
        #     0      = 0     
        # i--A--3      C--1
        #   /\
        #  j  m
        #      \ 0
        #    i--A--3
        #      /
        #     j
        vec = (-1,1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('meijf,maijb->eafb',(A,A)).contiguous().view(dimsA[1]**2, dimsA[4]**2)
        env.C[(coord,vec)] = a

def print_env(env, verbosity=0):
    print("dtype "+str(env.dtype))
    print("device "+str(env.device))

    for key,t in env.C.items():
        print(str(key)+" "+str(t.size()))
        if verbosity>0: 
            print(t)
    for key,t in env.T.items():
        print(str(key)+" "+str(t.size()))
        if verbosity>0:
            print(t)

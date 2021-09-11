import torch
import numpy as np
import json
from ipeps.tensor_io import *


def write_json_to_file(tensor, outputfile):
    serialized_tensor = serialize_bare_tensor_legacy(tensor)
    with open(outputfile, 'w') as f:
        json.dump(serialized_tensor, f, indent=4, separators=(',', ': '))


def write_SU3_D9_tensors():

    # Virtual space D=9, V = \bar{3} + 6

    # ______ Trivalent tensors ________
    M0 = torch.zeros((9, 9, 9), dtype=torch.complex128)
    M1 = torch.zeros((9, 9, 9), dtype=torch.complex128)
    M2 = torch.zeros((9, 9, 9), dtype=torch.complex128)

    # M0: {3,0}, A2
    M0[0, 1, 2] = M0[1, 2, 0] = M0[2, 0, 1] = -1 / np.sqrt(6)
    M0[0, 2, 1] = M0[1, 0, 2] = M0[2, 1, 0] = 1 / np.sqrt(6)

    # M1: {2,1}, A1
    M1[0, 0, 8] = 1 / (3 * np.sqrt(2))
    M1[0, 1, 7] = -(1 / 6)
    M1[0, 2, 5] = 1 / 6
    M1[0, 5, 2] = 1 / 6
    M1[0, 7, 1] = -(1 / 6)
    M1[0, 8, 0] = 1 / (3 * np.sqrt(2))
    M1[1, 0, 7] = -(1 / 6)
    M1[1, 1, 6] = 1 / (3 * np.sqrt(2))
    M1[1, 2, 4] = -(1 / 6)
    M1[1, 4, 2] = -(1 / 6)
    M1[1, 6, 1] = 1 / (3 * np.sqrt(2))
    M1[1, 7, 0] = -(1 / 6)
    M1[2, 0, 5] = 1 / 6
    M1[2, 1, 4] = -(1 / 6)
    M1[2, 2, 3] = 1 / (3 * np.sqrt(2))
    M1[2, 3, 2] = 1 / (3 * np.sqrt(2))
    M1[2, 4, 1] = -(1 / 6)
    M1[2, 5, 0] = 1 / 6
    M1[3, 2, 2] = 1 / (3 * np.sqrt(2))
    M1[4, 1, 2] = -(1 / 6)
    M1[4, 2, 1] = -(1 / 6)
    M1[5, 0, 2] = 1 / 6
    M1[5, 2, 0] = 1 / 6
    M1[6, 1, 1] = 1 / (3 * np.sqrt(2))
    M1[7, 0, 1] = -(1 / 6)
    M1[7, 1, 0] = -(1 / 6)
    M1[8, 0, 0] = 1 / (3 * np.sqrt(2))

    # M2: {0,3}, A1
    M2[3, 6, 8] = 1 / (3 * np.sqrt(2))
    M2[3, 7, 7] = -(1 / (3 * np.sqrt(2)))
    M2[3, 8, 6] = 1 / (3 * np.sqrt(2))
    M2[4, 4, 8] = -(1 / (3 * np.sqrt(2)))
    M2[4, 5, 7] = 1 / 6
    M2[4, 7, 5] = 1 / 6
    M2[4, 8, 4] = -(1 / (3 * np.sqrt(2)))
    M2[5, 4, 7] = 1 / 6
    M2[5, 5, 6] = -(1 / (3 * np.sqrt(2)))
    M2[5, 6, 5] = -(1 / (3 * np.sqrt(2)))
    M2[5, 7, 4] = 1 / 6
    M2[6, 3, 8] = 1 / (3 * np.sqrt(2))
    M2[6, 5, 5] = -(1 / (3 * np.sqrt(2)))
    M2[6, 8, 3] = 1 / (3 * np.sqrt(2))
    M2[7, 3, 7] = -(1 / (3 * np.sqrt(2)))
    M2[7, 4, 5] = 1 / 6
    M2[7, 5, 4] = 1 / 6
    M2[7, 7, 3] = -(1 / (3 * np.sqrt(2)))
    M2[8, 3, 6] = 1 / (3 * np.sqrt(2))
    M2[8, 4, 4] = -(1 / (3 * np.sqrt(2)))
    M2[8, 6, 3] = 1 / (3 * np.sqrt(2))

    # ______ Bivalent tensors ________
    L0 = torch.zeros((3, 9, 9), dtype=torch.complex128)
    L1 = torch.zeros((3, 9, 9), dtype=torch.complex128)
    L2 = torch.zeros((3, 9, 9), dtype=torch.complex128)

    # L0: {2, 0}, B
    L0[0, 0, 1] = L0[1, 0, 2] = L0[2, 1, 2] = -1. / np.sqrt(2.)
    L0[0, 1, 0] = L0[1, 2, 0] = L0[2, 2, 1] = 1. / np.sqrt(2.)

    # L1: {1,1}, A
    L1[0, 0, 5] = 1 / (2 * np.sqrt(2))
    L1[0, 1, 4] = -(1 / (2 * np.sqrt(2)))
    L1[0, 2, 3] = 1 / 2
    L1[0, 3, 2] = 1 / 2
    L1[0, 4, 1] = -(1 / (2 * np.sqrt(2)))
    L1[0, 5, 0] = 1 / (2 * np.sqrt(2))

    L1[1, 0, 7] = 1 / (2 * np.sqrt(2))
    L1[1, 1, 6] = -(1 / 2)
    L1[1, 2, 4] = 1 / (2 * np.sqrt(2))
    L1[1, 4, 2] = 1 / (2 * np.sqrt(2))
    L1[1, 6, 1] = -(1 / 2)
    L1[1, 7, 0] = 1 / (2 * np.sqrt(2))

    L1[2, 0, 8] = 1 / 2
    L1[2, 1, 7] = -(1 / (2 * np.sqrt(2)))
    L1[2, 2, 5] = 1 / (2 * np.sqrt(2))
    L1[2, 5, 2] = 1 / (2 * np.sqrt(2))
    L1[2, 7, 1] = -(1 / (2 * np.sqrt(2)))
    L1[2, 8, 0] = 1 / 2

    # L2: {1,1}, B
    L2[0, 0, 5] = 1 / (2 * np.sqrt(2))
    L2[0, 1, 4] = -(1 / (2 * np.sqrt(2)))
    L2[0, 2, 3] = 1 / 2
    L2[0, 3, 2] = -(1 / 2)
    L2[0, 4, 1] = 1 / (2 * np.sqrt(2))
    L2[0, 5, 0] = -(1 / (2 * np.sqrt(2)))

    L2[1, 0, 7] = 1 / (2 * np.sqrt(2))
    L2[1, 1, 6] = -(1 / 2)
    L2[1, 2, 4] = 1 / (2 * np.sqrt(2))
    L2[1, 4, 2] = -(1 / (2 * np.sqrt(2)))
    L2[1, 6, 1] = 1 / 2
    L2[1, 7, 0] = -(1 / (2 * np.sqrt(2)))

    L2[2, 0, 8] = 1 / 2
    L2[2, 1, 7] = -(1 / (2 * np.sqrt(2)))
    L2[2, 2, 5] = 1 / (2 * np.sqrt(2))
    L2[2, 5, 2] = -(1 / (2 * np.sqrt(2)))
    L2[2, 7, 1] = 1 / (2 * np.sqrt(2))
    L2[2, 8, 0] = -(1 / 2)

    # ___________________________________
    for tensor, name in zip([M0, M1, M2, L0, L1, L2], ['M0', 'M1', 'M2', 'L0', 'L1', 'L2']):
        path = "SU3_CSL_ipeps/SU3_D9_bar3p6_tensors/"
        filename = path + name + '.json'
        write_json_to_file(tensor, filename)


def load_SU3_tensor(name):
    with open(name + '.json') as j:
        # load tensor as a json file
        tensor = json.load(j)
        # convert to torch.tensor object
        tensor = torch.tensor(read_bare_json_tensor_np_legacy(tensor), dtype=torch.complex128)
        return (tensor)

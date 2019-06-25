import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-omp_cores", type=int, default=1,help="number of OpenMP cores")
parser.add_argument("-chi", type=int, default=20, help="chi")
parser.add_argument("-cuda", type=int, default=-1, help="GPU #")

args = parser.parse_args()


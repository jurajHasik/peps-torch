
import torch
from args import args
import env
from env import ENV
import ipeps
import ctmrg

if __name__=='__main__':

	state = ipeps.read_ipeps(None, args.instate)

	# test tiling of square lattice with 3x2 unit cell
	#   0 1 2
	# 0 A B C
	# 1 D E F
	lx, ly = 6, 6
	for y in range(-ly//2,ly//2+1):
		if y == -ly//2:
			for x in range(-lx//2,lx//2):
				print(str(x)+" ", end="")
			print("")
		print(str(y)+" ", end="")
		for x in range(-lx//2,lx//2):
			print(str(state.vertexToSite((x,y)))+" ", end="")
		print("")

	ctm_env = ENV(args,state)
	ctm_env = ctmrg.run(state,ctm_env)


	state = ipeps.read_ipeps(None, args.instate)

	# test tiling of square lattice with 3x2 unit cell
	#   0 1 2
	# 0 A B C
	# 1 D E F
	lx, ly = 6, 6
	for y in range(-ly//2,ly//2+1):
		if y == -ly//2:
			for x in range(-lx//2,lx//2):
				print(str(x)+" ", end="")
			print("")
		print(str(y)+" ", end="")
		for x in range(-lx//2,lx//2):
			print(str(state.vertexToSite((x,y)))+" ", end="")
		print("")

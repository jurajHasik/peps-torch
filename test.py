import torch
from args import args
from ipeps import IPEPS
from env import ENV

if __name__=='__main__':

	sites = dict([((0,0),"A"), ((1,0), "B"), ((2,0), "C"), \
		((0,1), "D"), ((1,1), "E"), ((2,1), "F")])
	def vertexToSite3x2(coord):
		Lx = 3
		Ly = 2
		x = coord[0]
		y = coord[1]

		return ( (x + abs(x)*Lx)%Lx, (y + abs(y)*Ly)%Ly )

	ipeps = IPEPS(args, sites, vertexToSite3x2)

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
			print(str(ipeps.site((x,y)))+" ", end="")
		print("")

	env = ENV(args, ipeps)

	# test map to environment of 3x2 unit cell
	#   0 1 2
	# 0 A B C
	# 1 D E F
	inner_lx, inner_ly = 6, 6
	# top edge
	y0 = -inner_ly//2
	print(env.C[(ipeps.vertexToSite((-inner_lx//2,y0)),(-1,-1))], end=" ")
	for x in range(-inner_lx//2,-inner_lx//2+inner_lx):
		print(env.T[(ipeps.vertexToSite((x,y0)),(0,-1))], end=" ")
	print(env.C[(ipeps.vertexToSite((-inner_lx//2+inner_lx-1,y0)),(1,-1))])
		
	for y in range(-inner_ly//2,-inner_ly//2+inner_ly):
		print(env.T[(ipeps.vertexToSite((-inner_lx//2,y)),(-1,0))], end=" ")
		for x in range(-inner_lx//2,-inner_lx//2+inner_lx):
			print(str(ipeps.site((x,y))), end=" ")
		print(env.T[(ipeps.vertexToSite((-inner_lx//2+inner_lx-1,y)),(1,0))])

	# bottom edge
	y0 = -inner_ly//2+inner_ly-1
	print(env.C[(ipeps.vertexToSite((-inner_lx//2,y0)),(-1,1))], end=" ")
	for x in range(-inner_lx//2,-inner_lx//2+inner_lx):
		print(env.T[(ipeps.vertexToSite((x,y0)),(0,1))], end=" ")
	print(env.C[(ipeps.vertexToSite((-inner_lx//2+inner_lx-1,y0)),(1,1))])

	sites = dict([((0,0),"A"), ((1,0), "B")])
	def vertexToSite2x1(coord):
		Lx = 2
		Ly = 1
		x = coord[0]
		y = coord[1]

		return ( ((x + abs(x)*Lx)%Lx + abs(y))%Lx, 0 )

	
	ipeps = IPEPS(args, sites, vertexToSite2x1)

	# test tiling of square lattice with 2x2 unit cell
	#   0 1 0 1
	# 0 A B A B
	# 1 B A B A
	# 0 A B A B
	lx, ly = 6, 6
	for y in range(-ly//2,ly//2+1):
		if y == -ly//2:
			for x in range(-lx//2,lx//2):
				print(str(x)+" ", end="")
			print("")
		print(str(y)+" ", end="")
		for x in range(-lx//2,lx//2):
			print(str(ipeps.site((x,y)))+" ", end="")
		print("")

	# test map to environment of 2x2 unit cell
	#   0 1 0 1
	# 0 A B A B
	# 1 B A B A
	# 0 A B A B
	inner_lx, inner_ly = 6, 6
	# top edge
	y0 = -inner_ly//2
	print(env.C[(ipeps.vertexToSite((-inner_lx//2,y0)),(-1,-1))], end=" ")
	for x in range(-inner_lx//2,-inner_lx//2+inner_lx):
		print(env.T[(ipeps.vertexToSite((x,y0)),(0,-1))], end=" ")
	print(env.C[(ipeps.vertexToSite((-inner_lx//2+inner_lx-1,y0)),(1,-1))])
		
	for y in range(-inner_ly//2,-inner_ly//2+inner_ly):
		print(env.T[(ipeps.vertexToSite((-inner_lx//2,y)),(-1,0))], end=" ")
		for x in range(-inner_lx//2,-inner_lx//2+inner_lx):
			print(str(ipeps.site((x,y))), end=" ")
		print(env.T[(ipeps.vertexToSite((-inner_lx//2+inner_lx-1,y)),(1,0))])

	# bottom edge
	y0 = -inner_ly//2+inner_ly-1
	print(env.C[(ipeps.vertexToSite((-inner_lx//2,y0)),(-1,1))], end=" ")
	for x in range(-inner_lx//2,-inner_lx//2+inner_lx):
		print(env.T[(ipeps.vertexToSite((x,y0)),(0,1))], end=" ")
	print(env.C[(ipeps.vertexToSite((-inner_lx//2+inner_lx-1,y0)),(1,1))])
import matplotlib.pyplot as plt
import numpy as np

#ensenyarem cada 10 timesteps
plot_every = 10
#calcular distancia entre particules
def distance(x1, y1, x2, y2):
	return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():
	# Simulation parameters
	Nx                     = 300   # resolution x-dir
	Ny                     = 100    # resolution y-dir
	rho0                   = 1.225    # average density
	tau                    = 0.6    # collision timescale
	Nt                     = 10000   # number of iterations
	plotRealTime = True # switch on for plotting as the simulation goes along

	#numero de particules del sistema
	NL = 9
	#direccions X de les particules
	cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
	#direccions Y de les particules
	cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
	#proabilitat que te la particula central de moure's en cada direcció o quedar-se quieta
	weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1

	# Initial Conditions
	#Interferencies per a que la simulació sigui més realista (crear un efecte caos)
	F = np.ones((Ny,Nx,NL)) + 0.015*np.random.randn(Ny,Nx,NL)
	#Fer que el fluit vagi cap a la dreta afegint un valor a les particules de la dreta
	F[:,:,3] = 2.4
	#si el punt es "false" es un espai buit, si es "true" es un obstacle
	cylinder = np.full ((Ny, Nx), False)

	#fiquem el cilindre al mig de l'eix Y i a 3/4 de l'eix X
	for y in range(0, Ny):
		for x in range(0, Nx):
			if(distance(Nx//6, Ny//2, x, y)<15):
				cylinder[y][x] = True

	#################COLORBAR##########################################################
	vmin = 0.001
	rho = np.sum(F,2)
	ux  = np.sum(F*cxs,2) / rho
	uy  = np.sum(F*cys,2) / rho
	#identifiquem els punts que creen vòrtex
	dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
	dfydy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
	curl = dfydx - dfydy
	magnitude = np.abs(curl)
	plt.imshow(magnitude, cmap="twilight", vmin=vmin)
	plt.colorbar(orientation="horizontal")
	###################################################################################

	#main Loop
	#cada timestep (Nt), la particula es moura en una de les 9 direccions cap al seu veí
	for it in range(Nt):
		print(it)

		#contrarrestem els valors que van cap a l'esquerra (les ones reboten a les parets)
		F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
		F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

		#movem les partícules
		for i, cx, cy, in zip(range(NL), cxs, cys):
			F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
			F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)

		#calculem les colisions (punts que estan dintre del cilindre) i invertim la direcció d'aquests
		bndryF = F[cylinder, :]
		bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]

		#calculem la densitat
		rho = np.sum(F,2)
		#momentum
		ux  = np.sum(F*cxs,2) / rho
		uy  = np.sum(F*cys,2) / rho

		#invertim totes les velocitats dins del cilindre
		F[cylinder,:] = bndryF
		ux[cylinder] = 0
		uy[cylinder] = 0

		#colisió
		#calculem el F equilibrium
		Feq = np.zeros(F.shape)
		for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
			Feq[:, :, i] = rho * w * (
				1+3*(cx*ux + cy*uy)+9*(cx*ux + cy*uy)**2 / 2-3 * (ux**2 + uy**2)/2
			)
		#apliquel el fequilibrium a la altra ecuació
		F = F + -(1/tau) * (F-Feq)

		if(it%plot_every==0):
			#identifiquem els punts que creen vòrtex
			dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
			dfydy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
			curl = dfydx - dfydy

			magnitude = np.abs(curl)
			vmax = np.max(magnitude)
			vmin = 0.001
			#twilight_shifted
			#plotting amb el mapa de colors bwr que dona vermell als valors positius i blau als valors negatius

			plt.imshow(magnitude, cmap="twilight", vmin=vmin)
			plt.pause(.01)
			plt.cla()

if __name__ == '__main__':
	main()

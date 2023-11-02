import matplotlib.pyplot as plt
import numpy as np

# Ensenyarem cada 10 timesteps
plot_every = 10
# Calcular distancia entre particules
def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():
    # Simulation parameters
    Nx = 300  # resolució x
    Ny = 100  # resolució y
    rho0 = 20  # densitat mitja
    tau = 0.6  # col·lisions
    Nt = 10000  # iteracions
    plotRealTime = True

    # Numero de particules del sistema
    NL = 9
    # Direccions X de les particules
    cxs = np.array([0, 0, 1,
                    1, 1, 0,
                   -1, -1, -1])
    # Direccions Y de les particules
    cys = np.array([0, 1, 1,
                    0, -1, -1,
                   -1, 0, 1])
    # Proabilitat que te la particula central de moure's en cada direcció o quedar-se quieta
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])  # sums to 1

    # Initial Conditions
    # Interferencies per a que la simulació sigui més realista (crear un efecte caos)
    F = np.ones((Ny, Nx, NL)) + 0.1*np.random.randn(Ny, Nx, NL)
    # Fer que el fluit vagi cap a la dreta afegint un valor a les particules de la dreta
    F[:, :, 3] = 2.4
    # Si el punt es "false" es un espai buit, si es "true" es un cube
    cube = np.full((Ny, Nx), False)
    # Fijamos el cuadrado en el centro del eje Y y a 3/4 del eje X
    start_x = Nx // 4
    end_x = Nx // 4 + Ny // 2
    start_y = Ny // 4
    end_y = Ny // 4 + Ny // 2
    cube[start_y:end_y, start_x:end_x] = True

    #################COLORBAR##########################################################
    vmin = 0.001
    rho = np.sum(F, 2)
    ux = np.sum(F*cxs, 2) / rho
    uy = np.sum(F*cys, 2) / rho
    # Identifiquem els punts que creen vòrtex
    dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
    dfydy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
    curl = dfydx - dfydy
    magnitude = np.abs(curl)
    plt.imshow(magnitude, cmap="twilight", vmin=vmin)
    plt.colorbar(orientation="horizontal")
    ###################################################################################

    # Main Loop
    # Cada timestep (Nt), la particula es moura en una de les 9 direccions cap al seu veí
    for it in range(Nt):
        print(it)

        # Contrarrestem els valors que van cap a l'esquerra (les ones reboten a les parets)
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        # Movem les partícules
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # Calculem les colisions (punts que estan dintre del quadrat) i invertim la direcció d'aquests
        bndryF = F[cube, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Calculem la densitat
        rho = np.sum(F, 2)
        # Momentum
        ux = np.sum(F*cxs, 2) / rho
        uy = np.sum(F*cys, 2) / rho

        # Invertim totes les velocitats dins del quadrat
        F[cube, :] = bndryF
        ux[cube] = 0
        uy[cube] = 0

        # Colisió
        # Calculem el F equilibrium
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                1 + 3*(cx*ux + cy*uy) + 9*(cx*ux + cy*uy)**2 / 2 - 3*(ux**2 + uy**2)/2)

        # Apliquel el Fequilibrium a la altra ecuació
        F = F + -(1/tau) * (F-Feq)

        if it % plot_every == 0:
            # Identifiquem els punts que creen vòrtex
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfydy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfydy

            magnitude = np.abs(curl)
            vmax = 0.1
            vmin = 0.0001

            # Twilight_shifted

            plt.imshow(magnitude, cmap="twilight", vmin=vmin, vmax=vmax)
            plt.pause(.01)
            plt.cla()

if __name__ == '__main__':
    main()

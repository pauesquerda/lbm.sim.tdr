import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

# fer plot de la simulació cada 20 timesteps
plot_every = 20

# calcul vectorial de distància entre partícules
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    # paràmetres de la simulació
    Nx = 200   # resolució en la direcció X
    Ny = 100   # resolució en la direcció Y
    rho0 = 1.225   # densitat del fluid
    tau = 0.6   # temps de col·lisió
    Nt = 10000   # número d'iteracions
    plotRealTime = True   # traçar la simulació en temps real

    # número de partícules del sistema
    NL = 9
    # matriu de direccions X de les partícules
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    # matriu de direccions Y de les partícules
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    # matriu de probabilitats de mobilitat de les partícules
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # condicions inicials
    F = np.ones((Ny, Nx, NL)) + 0.015*np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.4  # Hacer que el fluido vaya hacia la derecha añadiendo un valor a las partículas de la derecha

    wing = np.full((Ny, Nx), False)

    # crear un perfil alar simple utilitzanr la funció del sinus
    for y in range(Ny):
        for x in range(Nx):
            if y > 0.3 * Ny and y < 0.6 * Ny and x > 0.2 * Nx and x < 0.8 * Nx:
                wing[y][x] = y < 0.3 * Ny + 10 * np.sin(np.pi * (x - 0.01 * Nx) / (0.6 * Nx))+1

    rotated_wing = rotate(wing, 20, reshape=False)

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

    # bucle principal
    for it in range(Nt):
        print(it)

        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]  # condicions de frontera amb els límits de la simulació
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        # Mover las partículas
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # calcular els punts que estàn dins del sòlid i invertir-los (condicions de frontera)
        bndryF = F[rotated_wing, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # calcular densitat
        rho = np.sum(F, 2)
        # Momentum
        ux = np.sum(F*cxs, 2) / rho
        uy = np.sum(F*cys, 2) / rho

        # invertir les velocitats que estàn dins del sòlid (condicions de frontera)
        F[rotated_wing, :] = bndryF
        ux[rotated_wing] = 0
        uy[rotated_wing] = 0

        # col·lisió
        # calcular f equilibriuim
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                 1 + 3*(cx*ux + cy*uy) + 9*(cx*ux + cy*uy)**2 / 2 - 3*(ux**2 + uy**2)/2
            )
        # aplicar f equilibrium a l'altra equació
        F = F + -(1/tau) * (F - Feq)

        if it % plot_every == 0:
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfydy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfydy
            #
            magnitude = np.abs(curl)
            vmax = np.max(magnitude)
            vmin = 0.001
            plt.imshow(magnitude, cmap="twilight", vmin=vmin, vmax=0.11)
            plt.imshow(rotated_wing, cmap="binary", alpha=0.2)
            plt.pause(0.01)
            plt.cla()

if __name__ == '__main__':
    main()

import taichi as ti
from shared.parameters import res, diffusion_rate, dt, h

@ti.kernel
def set_boundaries(dens: ti.template()):
    for i in range(1, res - 1):
        # Bordes superior e inferior
        dens[i, 0] = dens[i, 1]
        dens[i, res - 1] = dens[i, res - 2]

    for j in range(1, res - 1):
        # Bordes izquierdo y derecho
        dens[0, j] = dens[1, j]
        dens[res - 1, j] = dens[res - 2, j]

    # Esquinas
    dens[0, 0] = 0.5 * (dens[1, 0] + dens[0, 1])
    dens[0, res - 1] = 0.5 * (dens[1, res - 1] + dens[0, res - 2])
    dens[res - 1, 0] = 0.5 * (dens[res - 2, 0] + dens[res - 1, 1])
    dens[res - 1, res - 1] = 0.5 * (dens[res - 2, res - 1] + dens[res - 1, res - 2])

@ti.kernel
def jacobi_iter(dens: ti.template(), dens0: ti.template()):
    a = diffusion_rate * dt / (h * h)
    for i, j in ti.ndrange((1, res - 1), (1, res - 1)):
        dens[i, j] = (dens0[i, j] + a * (dens0[i - 1, j] + dens0[i + 1, j] + dens0[i, j - 1] + dens0[i, j + 1])) / (1 + 4 * a)


def diffuse(dens, iterations=100):
    for _ in range(iterations):
        jacobi_iter(dens.nxt, dens.cur)
        set_boundaries(dens.nxt)
        dens.cur, dens.nxt = dens.nxt, dens.cur

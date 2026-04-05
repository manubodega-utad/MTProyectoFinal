import taichi as ti
from shared.parameters import res, diffusion_rate, dt, h


@ti.kernel
def copy_field(dst: ti.template(), src: ti.template()):
    for i, j in dst:
        dst[i, j] = src[i, j]


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
def jacobi_iter(dens_new: ti.template(), dens: ti.template(), dens0: ti.template()):
    # dens0 es la densidad antes de difundir.
    a = diffusion_rate * dt / (h * h)
    for i, j in ti.ndrange((1, res - 1), (1, res - 1)):
        # Ecuación Implícita "Diffusion Step":
        #   dens0[i,j] = dens[i,j] - a*(dens[i-1,j]+dens[i+1,j]+dens[i,j-1]+dens[i,j+1] - 4*dens[i,j])

        # Densidad de los vecinos
        dens_vecinos = (dens[i - 1, j] + dens[i + 1, j] + dens[i, j - 1] + dens[i, j + 1])

        # dens0 = dens - a*dens_vecinos + 4a*dens = (1+4a)*dens - a*dens_vecinos
        # Despejamos dens[i,j]:
        dens_new[i, j] = (dens0[i, j] + a * dens_vecinos) / (1 + 4 * a)


def diffuse(dens, dens0, iterations=100):
    copy_field(dens0, dens.cur)
    for _ in range(iterations):
        jacobi_iter(dens.nxt, dens.cur, dens0)
        set_boundaries(dens.nxt)
        dens.swap()

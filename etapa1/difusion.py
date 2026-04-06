import taichi as ti
from shared.parameters import res, diffusion_rate, dt, h
from shared.utils import set_boundaries


@ti.kernel
def copy_field(dst: ti.template(), src: ti.template()):
    for i, j in dst:
        dst[i, j] = src[i, j]


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

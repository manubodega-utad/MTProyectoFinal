import taichi as ti
from shared.parameters import res, dt


@ti.func
def bilerp(p, dens0):
    x, y = p[0], p[1]

    i0, i1 = int(x), int(x) + 1
    j0, j1 = int(y), int(y) + 1
    s = x - i0
    t = y - j0

    # Interpolación bilineal
    return (1 - s) * ((1 - t) * dens0[i0, j0] + t * dens0[i0, j1]) + s * ((1 - t) * dens0[i1, j0] + t * dens0[i1, j1])


@ti.kernel
def advect(d: ti.template(), d0: ti.template(), vel: ti.template()):
    # Para cada celda (i,j), buscamos de dónde vino la partícula hace un dt.
    for i, j in ti.ndrange((1, res - 1), (1, res - 1)):
        # Trazar hacia atrás:
        x = i - dt * vel[i, j][0]
        y = j - dt * vel[i, j][1]

        # Permanecer dentro del dominio
        # No salirse por izquierda
        if x < 0.5:
            x = 0.5
        # No salirse por derecha
        if x > res - 1.5:
            x = res - 1.5
        # No salirse por abajo
        if y < 0.5:
            y = 0.5
        # No salirse por arriba
        if y > res - 1.5:
            y = res - 1.5

        # Interpolación bilineal en el punto origen
        p = ti.Vector([x, y])
        d[i, j] = bilerp(p, d0)

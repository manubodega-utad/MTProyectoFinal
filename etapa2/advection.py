import taichi as ti
from shared.parameters import res, dt
@ti.func
def bilerp(p, f):
    x, y = p[0], p[1]
    i0 = int(x)
    j0 = int(y)
    i1 = i0 + 1
    j1 = j0 + 1
    s1 = x - i0
    s0 = 1.0 - s1
    t1 = y - j0
    t0 = 1.0 - t1
    return s0 * (t0 * f[i0, j0] + t1 * f[i0, j1]) + \
           s1 * (t0 * f[i1, j0] + t1 * f[i1, j1])

@ti.kernel
def advect(d: ti.template(), d0: ti.template(), vel: ti.template()):
    for i, j in ti.ndrange((1, res - 1), (1, res - 1)):
        x = i - dt * vel[i, j][0] * res
        y = j - dt * vel[i, j][1] * res

        # Clamp dentro del dominio
        x = min(max(0.5, x), res - 1.5)
        y = min(max(0.5, y), res - 1.5)

        p = ti.Vector([x, y])
        d[i, j] = bilerp(p, d0)
import taichi as ti
import numpy as np
from shared.parameters import res, velocity_scale
from shared.utils import density_source, velocity_source, set_boundaries
from etapa1.difusion import diffuse
from advection import advect

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

# Campos
density_0 = ti.field(dtype=ti.f32, shape=(res, res))  # Densidad de la fuente
density_1 = ti.field(dtype=ti.f32, shape=(res, res))  # Densidad iteración anterior
density_2 = ti.field(dtype=ti.f32, shape=(res, res))  # Densidad nueva
velocity = ti.Vector.field(2, dtype=ti.f32, shape=(res, res))


class FieldPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


dens = FieldPair(density_1, density_2)


@ti.kernel
def init_vel(vel: ti.template()):
    franja_inferior = res // 3
    franja_superior = 2 * res // 3
    for i, j in vel:
        if j < franja_inferior:
            # Franja de abajo: derecha → izquierda
            vel[i, j][0] = -velocity_scale
        elif j < franja_superior:
            # Franja central: izquierda → derecha
            vel[i, j][0] = velocity_scale
        else:
            # Franja de arriba: derecha → izquierda
            vel[i, j][0] = -velocity_scale


def init():
    density_0.fill(0)
    dens.cur.fill(0)
    dens.nxt.fill(0)
    init_vel(velocity)


def step(input_data):
    # 1. Fuente de densidad y Velocidad
    density_source(dens.cur, input_data)
    velocity_source(velocity, input_data)
    set_boundaries(dens.cur)

    # 2. Difusión de densidad
    diffuse(dens, density_0)

    # 3. Advección de densidad
    advect(dens.nxt, dens.cur, velocity)
    set_boundaries(dens.nxt)
    dens.swap()


def main():
    paused = False
    window = ti.ui.Window("Etapa 2: Advección", (res, res))
    canvas = window.get_canvas()

    init()

    while window.running:
        input_data = np.zeros(3, dtype=np.float32)

        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            if e.key == "r":
                paused = False
                init()
            elif e.key == "p":
                paused = not paused

        mouse = window.get_cursor_pos()
        mx = mouse[0] * res
        my = mouse[1] * res

        if window.is_pressed(ti.ui.RMB):
            input_data[0] = mx
            input_data[1] = my
            input_data[2] = 1.0

        if not paused:
            step(input_data)

        canvas.set_image(dens.cur)
        window.show()


if __name__ == "__main__":
    main()
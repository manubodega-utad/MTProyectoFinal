import taichi as ti
import numpy as np
from shared.parameters import res, s_force
from shared.utils import density_source, set_boundaries, add_forces
from etapa1.difusion import diffuse
from etapa2.advection import advect

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


def init():
    density_0.fill(0)
    dens.cur.fill(0)
    dens.nxt.fill(0)
    velocity.fill(0)


def step(input_data):
    # 1. Fuente de densidad y fuerza
    density_source(dens.cur, input_data)
    add_forces(velocity, input_data)
    set_boundaries(dens.cur)

    # 2. Difusión de densidad
    diffuse(dens, density_0)

    # 3. Advección de densidad
    advect(dens.nxt, dens.cur, velocity)
    set_boundaries(dens.nxt)
    dens.swap()


def main():
    paused = False
    window = ti.ui.Window("Etapa 3: Fuerzas", (res, res))
    canvas = window.get_canvas()
    init()

    prev_mouse = None
    while window.running:
        input_data = np.zeros(5, dtype=np.float32)

        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "r":
                paused = False
                init()
            elif e.key == "p":
                paused = not paused

        mouse = window.get_cursor_pos()
        mx = mouse[0] * res
        my = mouse[1] * res

        # 1. BOTÓN DERECHO (RMB)
        if window.is_pressed(ti.ui.RMB):
            input_data[0] = mx
            input_data[1] = my
            input_data[2] = 1.0
        else:
            input_data[2] = 0.0

        # 2. BOTÓN IZQUIERDO (LMB)
        if window.is_pressed(ti.ui.LMB):
            input_data[0] = mx
            input_data[1] = my
            if prev_mouse is not None:
                dx = mx - prev_mouse[0]
                dy = my - prev_mouse[1]
                input_data[3] = dx * s_force
                input_data[4] = dy * s_force
            prev_mouse = (mx, my)
        else:
            prev_mouse = None

        if not paused:
            step(input_data)

        canvas.set_image(dens.cur)
        window.show()


if __name__ == "__main__":
    main()
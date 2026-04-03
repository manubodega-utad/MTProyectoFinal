import taichi as ti
import numpy as np
from shared.parameters import res, dt, s_dens, s_radius
from shared.utils import density_source, velocity_source, swap
from etapa1.difusion import diffuse
from advection import advect

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

# Campos
velocity = ti.Vector.field(2, dtype=ti.f32, shape=(res, res))
density_1 = ti.field(dtype=ti.f32, shape=(res, res))
density_2 = ti.field(dtype=ti.f32, shape=(res, res))


class FieldPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


dens = FieldPair(density_1, density_2)


def init():
    dens.cur.fill(0)
    dens.nxt.fill(0)
    velocity.fill(0)


def step(input_data):
    density_source(dens.cur, input_data)
    velocity_source(velocity, input_data)
    debug_velocity()
    diffuse(dens)
    dens.swap()
    advect(dens.nxt, dens.cur, velocity)
    dens.swap()

@ti.kernel
def debug_velocity():
    for i, j in velocity:
        if velocity[i, j].norm() > 0.1:
            print(f"vel[{i},{j}] = ", velocity[i, j])

def main():
    paused = False
    window = ti.ui.Window("Etapa 2: Advección", (res, res))
    canvas = window.get_canvas()

    init()

    while window.running:
        input_data = np.zeros(5, dtype=np.float32)

        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            if e.key == "r":
                paused = False
                init()
            elif e.key == "p":
                paused = not paused

        if window.is_pressed(ti.ui.RMB):
            mouse = window.get_cursor_pos()
            input_data[0] = mouse[0] * res
            input_data[1] = mouse[1] * res
            input_data[2] = 1.0
            input_data[3] = 0.005    # velocidad x
            input_data[4] = 0.0    # velocidad y

        if not paused:
            step(input_data)

        canvas.set_image(dens.cur)
        window.show()


if __name__ == "__main__":
    main()
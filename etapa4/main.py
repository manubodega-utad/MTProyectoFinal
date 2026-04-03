import taichi as ti
import numpy as np
from shared.parameters import res
from shared.utils import density_source, swap, add_forces
from etapa1.difusion import diffuse
from etapa2.advection import advect

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

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

def step(input_data, force):
    density_source(dens.cur, input_data)
    add_forces(velocity, force[0], force[1])
    diffuse(0, dens)
    dens.swap()
    advect(0, dens.nxt, dens.cur, velocity)
    dens.swap()

def main():
    paused = False
    window = ti.ui.Window("Etapa 3: Campo de velocidades", (res, res))
    canvas = window.get_canvas()
    init()

    force = np.array([0.0, 0.0], dtype=np.float32)
    vel = 0.2

    while window.running:
        input_data = np.zeros(5, dtype=np.float32)
        force[:] = 0.0

        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "r":
                paused = False
                init()
            elif e.key == "p":
                paused = not paused
            elif e.key == ti.ui.UP:
                force[1] = vel
            elif e.key == ti.ui.DOWN:
                force[1] = -vel
            elif e.key == ti.ui.LEFT:
                force[0] = -vel
            elif e.key == ti.ui.RIGHT:
                force[0] = vel

        if window.is_pressed(ti.ui.RMB):
            mouse = window.get_cursor_pos()
            input_data[0] = mouse[0] * res
            input_data[1] = mouse[1] * res
            input_data[2] = 1.0
            input_data[3] = 0.0
            input_data[4] = 0.0

        if not paused:
            step(input_data, force)

        canvas.set_image(dens.cur)
        window.show()

if __name__ == "__main__":
    main()
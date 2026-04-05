import taichi as ti
import numpy as np
from shared.parameters import res, density_color
from shared.utils import density_source_rgb
from difusion import diffuse, set_boundaries

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

density_0 = ti.Vector.field(3, dtype=ti.f32, shape=(res, res))  # Densidad RGB de la fuente
density_1 = ti.Vector.field(3, dtype=ti.f32, shape=(res, res))  # Densidad RGB iteración anterior
density_2 = ti.Vector.field(3, dtype=ti.f32, shape=(res, res))  # Densidad RGB nueva
density_color = np.array(density_color, dtype=np.float32)


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


def step(input_data):
    density_source_rgb(dens.cur, input_data)
    set_boundaries(dens.cur)
    diffuse(dens, density_0)


def main():
    paused = False
    window = ti.ui.Window("Etapa 1: Difusión", (res, res))
    canvas = window.get_canvas()

    init()

    while window.running:
        input_data = np.zeros(6, dtype=np.float32)

        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == "r":
                paused = False
                init()
            elif e.key == "p":
                paused = not paused

        if window.is_pressed(ti.ui.RMB):
            mouse = window.get_cursor_pos()
            input_data[0] = mouse[0] * res
            input_data[1] = mouse[1] * res
            input_data[2] = 1.0
            input_data[3:] = density_color

        if not paused:
            step(input_data)

        canvas.set_image(dens.cur)
        window.show()


if __name__ == "__main__":
    main()

import taichi as ti
from shared.parameters import s_dens, s_radius, dt, density_color


@ti.kernel
def density_source(dens: ti.template(), input_data: ti.types.ndarray()):
    for i, j in dens:
        densidad = input_data[2] * s_dens
        mx, my = input_data[0], input_data[1]
        cx = i + 0.5
        cy = j + 0.5
        d2 = (cx - mx) ** 2 + (cy - my) ** 2
        dens[i, j] += dt * densidad * ti.exp(-6 * d2 / (s_radius * s_radius))


@ti.kernel
def density_source_rgb(dens: ti.template(), input_data: ti.types.ndarray()):
    for i, j in dens:
        if input_data[2] == 0:
            continue
        densidad = input_data[2] * s_dens
        mx, my = input_data[0], input_data[1]
        cx = i + 0.5
        cy = j + 0.5
        d2 = (cx - mx) ** 2 + (cy - my) ** 2
        dens[i, j] += dt * densidad * ti.exp(-6 * d2 / (s_radius * s_radius)) * ti.Vector(density_color)


@ti.kernel
def velocity_source(vel: ti.template(), input_data: ti.types.ndarray()):
    for i, j in vel:
        if 1 < i < vel.shape[0] - 2 and 1 < j < vel.shape[1] - 2:
            mx, my = input_data[0], input_data[1]
            cx, cy = i + 0.5, j + 0.5
            d2 = (cx - mx)**2 + (cy - my)**2
            f = ti.exp(-6 * d2 / (s_radius * s_radius))
            vel[i, j][0] += f * input_data[3]
            vel[i, j][1] += f * input_data[4]


@ti.kernel
def add_forces(vel: ti.template(), fx: float, fy: float):
    for i, j in vel:
        vel[i, j][0] += fx * dt
        vel[i, j][1] += fy * dt

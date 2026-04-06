import taichi as ti
from shared.parameters import s_dens, s_radius, dt, density_color, res


@ti.kernel
def density_source(dens: ti.template(), input_data: ti.types.ndarray()):
    # Posición del ratón
    mx = input_data[0]
    my = input_data[1]

    densidad = input_data[2] * s_dens
    for i, j in dens:
        cx = i + 0.5
        cy = j + 0.5
        d2 = (cx - mx) ** 2 + (cy - my) ** 2
        dens[i, j] += dt * densidad * ti.exp(-6 * d2 / (s_radius * s_radius))


@ti.kernel
def density_source_rgb(dens: ti.template(), input_data: ti.types.ndarray()):
    # Posición del ratón
    mx = input_data[0]
    my = input_data[1]

    densidad = input_data[2] * s_dens
    for i, j in dens:
        if input_data[2] == 0:
            continue

        cx = i + 0.5
        cy = j + 0.5
        d2 = (cx - mx) ** 2 + (cy - my) ** 2
        dens[i, j] += dt * densidad * ti.exp(-6 * d2 / (s_radius * s_radius)) * ti.Vector(density_color)


@ti.kernel
def velocity_source(vel: ti.template(), input_data: ti.types.ndarray()):
    # Posición del ratón
    mx = input_data[0]
    my = input_data[1]

    for i, j in vel:
        if 1 < i < vel.shape[0] - 2 and 1 < j < vel.shape[1] - 2:
            cx, cy = i + 0.5, j + 0.5
            d2 = (cx - mx)**2 + (cy - my)**2
            f = ti.exp(-6 * d2 / (s_radius * s_radius))
            vel[i, j][0] += f * input_data[3]
            vel[i, j][1] += f * input_data[4]


@ti.kernel
def add_forces(vel: ti.template(), input_data: ti.types.ndarray()):
    mx = input_data[0]
    my = input_data[1]
    fx = input_data[3]
    fy = input_data[4]
    for i, j in vel:
        # Verificamos que hay fuerza aplicándose
        if 1 < i < vel.shape[0] - 2 and 1 < j < vel.shape[1] - 2:
            cx, cy = i + 0.5, j + 0.5
            d2 = (cx - mx) ** 2 + (cy - my) ** 2
            f = ti.exp(-6 * d2 / (s_radius * s_radius))
            vel[i, j][0] += f * fx * dt
            vel[i, j][1] += f * fy * dt


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


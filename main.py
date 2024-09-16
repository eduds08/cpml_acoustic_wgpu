import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import findiff
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
from WebGpuHandler import WebGpuHandler
import wgpu


def derivative_iter(dim, u, boundary_factor):
    coeff_diff = [1.19629, -0.07975, 0.00957, -0.00070]

    # Swap das dimensões dependendo de dim
    if dim == 0:
        shifted_u = u
    else:
        shifted_u = [[u[j][i] for j in range(grid_size_z)] for i in range(grid_size_x)]  # Transpor para usar a outra dimensão

    # Criação do array para armazenar o resultado
    derivative_result = [[0.0 for _ in range(len(shifted_u[0]))] for _ in range(len(shifted_u) + 1)]

    # Cálculo da derivada usando o FDM
    for i, coeff in enumerate(coeff_diff):
        for z in range(len(shifted_u) - i):
            for x in range(len(shifted_u[0])):
                derivative_result[z][x] += coeff * shifted_u[z + i][x]
                derivative_result[z + i + 1][x] -= coeff * shifted_u[z][x]

    # Aplicação do boundary_factor
    derivative_result = derivative_result[boundary_factor:len(derivative_result) + boundary_factor - 1]

    # Se dim == 1, faz o swap de volta
    if dim == 1:
        derivative_result = [[derivative_result[j][i] for j in range(len(derivative_result))] for i in range(len(derivative_result[0]))]

    return derivative_result


# Calcula a derivada utilizando o Finite Difference Method (FDM)
def derivative(dim, u, boundary_factor):
    coeff_diff = np.asarray([1.19629, -0.07975, 0.00957, -0.00070], dtype=np.float32)

    shifted_u = u.swapaxes(0, dim)
    derivative_result = np.zeros((shifted_u.shape[0] + 1, shifted_u.shape[1]), dtype=np.float32)

    for i, coeff in enumerate(coeff_diff):
        derivative_result[:-i - 1, ...] += coeff * shifted_u[i:, ...]
        derivative_result[i + 1:, ...] -= coeff * shifted_u[:shifted_u.shape[0] - i, ...]

    return derivative_result[boundary_factor:derivative_result.shape[0] + boundary_factor - 1, ...].swapaxes(0, dim)


""" Parameters """

gpu_mode = True
derivative_fast = True

dt = np.float32(5e-8)  # Time step (s)
c = np.float32(5000.)  # Velocity (m/s)
dz = np.float32(5e-4)  # Grid Steps - z (m/px)
dx = np.float32(5e-4)  # Grid Steps - x (m/px)
size = np.int32(600)  # Grid Size (z, x) (px)
grid_size_z = np.int32(size)
grid_size_x = np.int32(size)
total_time = np.int32(6000)  # Total amount of time steps

ws_derivative = None
for i in range(15, 0, -1):
    if ((size + 1) % i) == 0:
        ws_derivative = i
        break

# Simplify typing
grid_size_shape = (grid_size_z, grid_size_x)

source_z = np.int32(grid_size_z / 2)  # Source position - z
source_x = np.int32(grid_size_x / 2)  # Source position - x

c = np.full(grid_size_shape, c, dtype=np.float32)
c_squared = (c ** 2).astype(np.float32)
cfl = (c_squared * (dt ** 2 / dz ** 2)).astype(np.float32)

if gpu_mode:
    wgpu_handler = WebGpuHandler(grid_size_z, grid_size_x)

# CFL
cfl_z = c * (dt / dz)
cfl_x = c * (dt / dx)
print(f'CFL-Z: {np.amax(cfl_z)}')
print(f'CFL-X: {np.amax(cfl_x)}')

# Source
time_arr = np.arange(total_time, dtype=np.float32) * dt
t0 = 4e-6
f0 = 1e6
source = np.exp(-((time_arr - t0) * f0) ** 2) * np.cos(2 * np.pi * f0 * time_arr)
source = (source / np.amax(np.abs(source))).astype(np.float32)

if gpu_mode:
    info_int = np.array(
        [
            grid_size_z,
            grid_size_x,
            source_z,
            source_x,
            0,
        ],
        dtype=np.int32
    )

    info_float = np.array(
        [
            dz,
            dx,
            dt,
        ],
        dtype=np.float32
    )

p_future = np.zeros(grid_size_shape, dtype=np.float32)
p_present = np.zeros(grid_size_shape, dtype=np.float32)
p_past = np.zeros(grid_size_shape, dtype=np.float32)
z_diff_1 = np.zeros(grid_size_shape, dtype=np.float32)
z_diff_2 = np.zeros(grid_size_shape, dtype=np.float32)
x_diff_1 = np.zeros(grid_size_shape, dtype=np.float32)
x_diff_2 = np.zeros(grid_size_shape, dtype=np.float32)

"""CPML começa aqui"""

absorption_layer_size = 50
damping_coefficient = 3e6

x, z = np.meshgrid(np.arange(grid_size_x, dtype=np.float32), np.arange(grid_size_z, dtype=np.float32))

# Aqui escolhemos as bordas que queremos absorver
is_x_absorption = (x > grid_size_x - absorption_layer_size) | (x < absorption_layer_size)
is_z_absorption = (z > grid_size_z - absorption_layer_size) | (z < absorption_layer_size)

is_x_absorption_int32 = ((x > grid_size_x - absorption_layer_size) | (x < absorption_layer_size)).astype(np.int32)
is_z_absorption_int32 = ((z > grid_size_z - absorption_layer_size) | (z < absorption_layer_size)).astype(np.int32)

absorption_coefficient = np.exp(
    -(damping_coefficient * (np.arange(absorption_layer_size) / absorption_layer_size) ** 2) * dt).astype(np.float32)

psi_x = np.zeros(is_x_absorption.sum(), dtype=np.float32)
psi_z = np.zeros(is_z_absorption.sum(), dtype=np.float32)
phi_x = np.zeros(is_x_absorption.sum(), dtype=np.float32)
phi_z = np.zeros(is_z_absorption.sum(), dtype=np.float32)

absorption_x = np.ones((grid_size_z, grid_size_x), dtype=np.float32)
absorption_z = np.ones((grid_size_z, grid_size_x), dtype=np.float32)

# Aqui também escolhemos as bordas que queremos absorver
absorption_x[:, :absorption_layer_size] = absorption_coefficient[::-1]
absorption_x[:, -absorption_layer_size:] = absorption_coefficient
absorption_z[:absorption_layer_size, :] = absorption_coefficient[:, np.newaxis][::-1]
absorption_z[-absorption_layer_size:, :] = absorption_coefficient[:, np.newaxis]

absorption_x = absorption_x[is_x_absorption]
absorption_z = absorption_z[is_z_absorption]

# GUI (animação)
vminmax = 1e-4
vscale = 1
surface_format = pg.QtGui.QSurfaceFormat()
surface_format.setSwapInterval(0)
pg.QtGui.QSurfaceFormat.setDefaultFormat(surface_format)
app = pg.QtWidgets.QApplication([])
raw_image_widget = RawImageGLWidget()
raw_image_widget.setWindowFlags(pg.QtCore.Qt.WindowType.FramelessWindowHint)
raw_image_widget.resize(vscale * grid_size_x, vscale * grid_size_z)
raw_image_widget.show()
colormap = plt.get_cmap("bwr")
norm = matplotlib.colors.Normalize(vmin=-vminmax, vmax=vminmax)

if gpu_mode:
    shader_file = open('./main.wgsl')
    shader_string = (shader_file.read()
                     .replace('wsz', f'{wgpu_handler.ws[0]}')
                     .replace('wsx', f'{wgpu_handler.ws[1]}')
                     .replace('ws_derivative', f'{ws_derivative}'))
    shader_file.close()

    wgpu_handler.shader_module = wgpu_handler.device.create_shader_module(code=shader_string)

    wgsl_data = {
        'infoI32': info_int,
        'infoF32': info_float,
        'source': source,
        'p_future': p_future,
        'p_present': p_present,
        'p_past': p_past,
        'cfl': cfl,
        'phi_z': phi_z,
        'phi_x': phi_x,
        'absorption_z': absorption_z,
        'absorption_x': absorption_x,
        'is_z_absorption_int32': is_z_absorption_int32,
        'is_x_absorption_int32': is_x_absorption_int32,
        'psi_z': psi_z,
        'psi_x': psi_x,
        'z_diff_1': z_diff_1,
        'z_diff_2': z_diff_2,
        'x_diff_1': x_diff_1,
        'x_diff_2': x_diff_2,
    }

    shader_lines = list(shader_string.split('\n'))
    buffers = wgpu_handler.create_buffers(wgsl_data, shader_lines)

    compute_derivative_z1 = wgpu_handler.create_compute_pipeline("derivative_z1")
    compute_derivative_x1 = wgpu_handler.create_compute_pipeline("derivative_x1")
    compute_derivative_z2 = wgpu_handler.create_compute_pipeline("derivative_z2")
    compute_derivative_x2 = wgpu_handler.create_compute_pipeline("derivative_x2")
    compute_sim = wgpu_handler.create_compute_pipeline("sim")
    compute_incr = wgpu_handler.create_compute_pipeline("incr_time")

# Loop principal
for i in range(total_time):
    if gpu_mode:
        command_encoder = wgpu_handler.device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()

        for index, bind_group in enumerate(wgpu_handler.bind_groups):
            compute_pass.set_bind_group(index, bind_group, [], 0, 999999)

        compute_pass.set_pipeline(compute_derivative_z1)
        compute_pass.dispatch_workgroups(grid_size_z // wgpu_handler.ws[0],
                                         grid_size_x // wgpu_handler.ws[1])

        compute_pass.set_pipeline(compute_derivative_x1)
        compute_pass.dispatch_workgroups(grid_size_z // wgpu_handler.ws[0],
                                         grid_size_x // wgpu_handler.ws[1])

        compute_pass.set_pipeline(compute_derivative_z2)
        compute_pass.dispatch_workgroups(grid_size_z // wgpu_handler.ws[0],
                                         grid_size_x // wgpu_handler.ws[1])

        compute_pass.set_pipeline(compute_derivative_x2)
        compute_pass.dispatch_workgroups(grid_size_z // wgpu_handler.ws[0],
                                         grid_size_x // wgpu_handler.ws[1])

        compute_pass.set_pipeline(compute_sim)
        compute_pass.dispatch_workgroups(grid_size_z // wgpu_handler.ws[0],
                                         grid_size_x // wgpu_handler.ws[1])

        compute_pass.set_pipeline(compute_incr)
        compute_pass.dispatch_workgroups(1)

        compute_pass.end()
        wgpu_handler.device.queue.submit([command_encoder.finish()])

        """ READ BUFFERS """
        p_future = (np.asarray(wgpu_handler.device.queue.read_buffer(buffers['b3']).cast("f"))
                    .reshape(grid_size_shape))

    else:
        if not derivative_fast:
            z_diff_1 = derivative_iter(dim=0, u=p_present, boundary_factor=1)
            x_diff_1 = derivative_iter(dim=1, u=p_present, boundary_factor=1)

            z_diff_2 = derivative_iter(dim=0, u=z_diff_1, boundary_factor=0)
            x_diff_2 = derivative_iter(dim=1, u=x_diff_1, boundary_factor=0)

            z_diff_2 = np.array(z_diff_2).reshape(grid_size_shape)
            x_diff_2 = np.array(x_diff_2).reshape(grid_size_shape)
        else:
            z_diff_1 = derivative(dim=0, u=p_present, boundary_factor=1)
            x_diff_1 = derivative(dim=1, u=p_present, boundary_factor=1)

            z_diff_2 = derivative(dim=0, u=z_diff_1, boundary_factor=0)
            x_diff_2 = derivative(dim=1, u=x_diff_1, boundary_factor=0)

        psi_z = absorption_z * psi_z + (absorption_z - 1) * z_diff_2[is_z_absorption]
        psi_x = absorption_x * psi_x + (absorption_x - 1) * x_diff_2[is_x_absorption]

        z_diff_2[is_z_absorption] += psi_z
        x_diff_2[is_x_absorption] += psi_x

        p_future = cfl * (z_diff_2 + x_diff_2)

        p_future += 2 * p_present - p_past

        p_past = p_present
        p_present = p_future

        p_future[source_z, source_x] += source[i]

    # Atualiza a GUI
    if not i % 3:
        raw_image_widget.setImage(colormap(norm(p_future.T)), levels=[0, 1])
        app.processEvents()
        plt.pause(1e-12)

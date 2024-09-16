import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget
from WebGpuHandler import WebGpuHandler

""" Parameters """

dt = np.float32(5e-8)  # Time step (s)
c = np.float32(5000.)  # Velocity (m/s)
dz = np.float32(5e-4)  # Grid Steps - z (m/px)
dx = np.float32(5e-4)  # Grid Steps - x (m/px)
size = np.int32(600)  # Grid Size (z, x) (px)
grid_size_z = np.int32(size)
grid_size_x = np.int32(size)
total_time = np.int32(3000)  # Total amount of time steps

# Simplify typing
grid_size_shape = (grid_size_z, grid_size_x)

source_z = np.int32(grid_size_z / 2)  # Source position - z
source_x = np.int32(grid_size_x / 2)  # Source position - x

# CFL
c = np.full(grid_size_shape, c, dtype=np.float32)
c_squared = (c ** 2).astype(np.float32)
cfl = (c_squared * (dt ** 2 / dz ** 2)).astype(np.float32)

wgpu_handler = WebGpuHandler(grid_size_z, grid_size_x)

# Source
time_arr = np.arange(total_time, dtype=np.float32) * dt
t0 = 4e-6
f0 = 1e6
source = np.exp(-((time_arr - t0) * f0) ** 2) * np.cos(2 * np.pi * f0 * time_arr)
source = (source / np.amax(np.abs(source))).astype(np.float32)

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

shader_file = open('no_cpml_gpu.wgsl')
shader_string = (shader_file.read()
                 .replace('wsz', f'{wgpu_handler.ws[0]}')
                 .replace('wsx', f'{wgpu_handler.ws[1]}'))
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
    p_future = (np.asarray(wgpu_handler.device.queue.read_buffer(buffers['b3']).cast("f")).reshape(grid_size_shape))

    # Atualiza a GUI
    if not i % 3:
        raw_image_widget.setImage(colormap(norm(p_future.T)), levels=[0, 1])
        app.processEvents()
        plt.pause(1e-12)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import findiff
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageGLWidget


# Calcula a derivada utilizando o Finite Difference Method (FDM)
def derivative(dim, u, boundary_factor):
    # accuracy = 8
    # offsets = (np.concatenate((np.arange(-accuracy // 2, 0), np.arange(accuracy // 2))) + 0.5).astype(np.float32)
    # coeff_diff = findiff.coefficients(deriv=1, offsets=list(offsets))["coefficients"][round(accuracy / 2):].astype(
    #     np.float32)

    coeff_diff = np.asarray([1.19629, -0.07975, 0.00957, -0.00070], dtype=np.float32)

    shifted_u = u.swapaxes(0, dim)
    derivative_result = np.zeros((shifted_u.shape[0] + 1, shifted_u.shape[1]), dtype=np.float32)

    for i, coeff in enumerate(coeff_diff):
        derivative_result[:-i - 1, ...] += coeff * shifted_u[i:, ...]
        derivative_result[i + 1:, ...] -= coeff * shifted_u[:shifted_u.shape[0] - i, ...]

    return derivative_result[boundary_factor:derivative_result.shape[0] + boundary_factor - 1, ...].swapaxes(0, dim)


""" Parameters """

dt = np.float32(5e-8)  # Time step (s)
c = np.float32(5000.)  # Velocity (m/s)
dz = np.float32(5e-4)  # Grid Steps - z (m/px)
dx = np.float32(5e-4)  # Grid Steps - x (m/px)
size = 600  # Grid Size (z, x) (px)
grid_size_z = size
grid_size_x = size
total_time = 3000  # Total amount of time steps

# Simplify typing
grid_size_shape = (grid_size_z, grid_size_x)

source_z = np.int32(grid_size_z / 2)  # Source position - z
source_x = np.int32(grid_size_x / 2)  # Source position - x

# CFL
cfl_z = c * (dt / dz)
cfl_x = c * (dt / dx)
print(f'CFL-Z: {cfl_z}')
print(f'CFL-X: {cfl_x}')

# Source
time_arr = np.arange(total_time, dtype=np.float32) * dt
t0 = 4e-6
f0 = 1e6
source = np.exp(-((time_arr - t0) * f0) ** 2) * np.cos(2 * np.pi * f0 * time_arr)
source = (source / np.amax(np.abs(source))).astype(np.float32)

p_future = np.zeros(grid_size_shape, dtype=np.float32)
p_present = np.zeros(grid_size_shape, dtype=np.float32)
p_past = np.zeros(grid_size_shape, dtype=np.float32)

"""CPML começa aqui"""

absorption_layer_size = 50
damping_coefficient = 3e6

x, z = np.meshgrid(np.arange(grid_size_x, dtype=np.float32), np.arange(grid_size_z, dtype=np.float32))

# Aqui escolhemos as bordas que queremos absorver
is_x_absorption = (x > grid_size_x - absorption_layer_size) | (x < absorption_layer_size)
is_z_absorption = (z > grid_size_z - absorption_layer_size) | (z < absorption_layer_size)

absorption_coefficient = np.exp(
    -(damping_coefficient * (np.arange(absorption_layer_size) / absorption_layer_size) ** 2) * dt).astype(np.float32)

psi_x = np.zeros(is_x_absorption.sum(), dtype=np.float32)
psi_z = np.zeros(is_z_absorption.sum(), dtype=np.float32)
phi_x = np.zeros(is_x_absorption.sum(), dtype=np.float32)
phi_z = np.zeros(is_z_absorption.sum(), dtype=np.float32)

absorption_x = np.ones((grid_size_z, grid_size_x), dtype=np.float32)
absorption_z = np.ones((grid_size_z, grid_size_x), dtype=np.float32)

# Aqui também escolhems as bordas que queremos absorver
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

# Loop principal
for i in range(total_time):
    z_diff_1 = derivative(dim=0, u=p_present, boundary_factor=1)
    x_diff_1 = derivative(dim=1, u=p_present, boundary_factor=1)

    phi_z = absorption_z * phi_z + (absorption_z - 1) * z_diff_1[is_z_absorption]
    phi_x = absorption_x * phi_x + (absorption_x - 1) * x_diff_1[is_x_absorption]

    z_diff_1[is_z_absorption] += phi_z
    x_diff_1[is_x_absorption] += phi_x

    z_diff_2 = derivative(dim=0, u=z_diff_1, boundary_factor=0)
    x_diff_2 = derivative(dim=1, u=x_diff_1, boundary_factor=0)

    psi_z = absorption_z * psi_z + (absorption_z - 1) * z_diff_2[is_z_absorption]
    psi_x = absorption_x * psi_x + (absorption_x - 1) * x_diff_2[is_x_absorption]

    z_diff_2[is_z_absorption] += psi_z
    x_diff_2[is_x_absorption] += psi_x

    p_future = (c ** 2) * (dt ** 2 / dz ** 2) * (z_diff_2 + x_diff_2)

    p_future += 2 * p_present - p_past

    p_past = p_present
    p_present = p_future

    p_future[source_z, source_x] += source[i]

    # Atualiza a GUI
    if not i % 3:
        raw_image_widget.setImage(colormap(norm(p_future.T)), levels=[0, 1])
        app.processEvents()
        plt.pause(1e-12)

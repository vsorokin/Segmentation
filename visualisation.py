from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import random

segmentation_cmap = colors.ListedColormap(
    ['none', 'red', 'green', 'yellow'])  # 'none' for 0 (background) to make it transparent

grid_cmap = colors.ListedColormap(
    ['none', 'yellow'])  # 'none' for 0 to make it transparent

grid_with_bg_cmap = colors.ListedColormap(['black', 'yellow'])

placeholder_cmap = colors.ListedColormap(['none'])


def create_grid(size=(240, 240), step=30, width=2):
    grid = np.zeros(size)
    for x in range(step, size[1], step):
        grid[:, x:x + width] = 1
    for y in range(step, size[0], step):
        grid[y:y + width, :] = 1
    return grid


def create_grid_3d(shape=(240, 240, 155), step=30, width=2):
    grid = np.zeros(shape)
    for x in range(step, shape[1], step):
        grid[:, x:x + width, :] = 1
    for y in range(step, shape[0], step):
        grid[y:y + width, :, :] = 1
    return grid


def draw_displacement_forces(shape, dx, dy, transpose=False):
    count = 12
    edge = 12
    xs = np.linspace(edge, shape[1] - edge, count, dtype=np.int32)
    ys = np.linspace(edge, shape[0] - edge, count, dtype=np.int32)
    dm = 2
    for x in xs:
        for y in ys:
            arrow_dx = -dx[y, x] * dm
            arrow_dy = -dy[y, x] * dm
            arrow_start_x = x - arrow_dx
            arrow_start_y = y - arrow_dy
            if transpose:
                arrow_start_x, arrow_start_y = arrow_start_y, arrow_start_x
                arrow_dx, arrow_dy = arrow_dy, arrow_dx
            plt.arrow(arrow_start_x, arrow_start_y, arrow_dx, arrow_dy, width=1.5, head_width=6,
                      color='dodgerblue')

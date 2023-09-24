"""
Elastic deformation of images as described in Simard, Steinkraus and Platt, "Best Practices for
Convolutional Neural Networks applied to Visual Document Analysis".
"""
import logging

import torch
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
import numpy as np

import visualisation
import skimage


class ElasticTransformation3D:
    def __init__(self, alpha, sigma, order=1, image_clipped=True, random_state=None):
        """
        Accepts image [C, H, W, D].
        Transforms channels equally and independently, then merges back.
        """
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.image_shape = (240, 240, 144 if image_clipped else 155)
        self.random_state = np.random.RandomState(None) if random_state is None else random_state

        def generate_displacement_():
            return generate_displacement(self.image_shape, self.random_state, alpha=alpha,
                                         sigma=sigma)

        self.dx = generate_displacement_()
        self.dy = generate_displacement_()
        self.dz = generate_displacement_()

    def transform_3d(self, image_3d):
        assert len(image_3d.shape) == 3
        assert image_3d.shape == self.image_shape
        # TODO remove redundant args
        result, _, _, _ = elastic_transform_3d(image_3d, alpha=self.alpha,
                                               sigma=self.sigma,
                                               order=self.order,
                                               random_state=self.random_state, dx=self.dx,
                                               dy=self.dy, dz=self.dz)
        assert len(result.shape) == 3
        assert result.shape == self.image_shape
        return result

    def restore_3d(self, transformed_image_3d, order=None):
        assert len(transformed_image_3d.shape) == 3
        assert transformed_image_3d.shape == self.image_shape
        if order is None:
            order = self.order
        # TODO remove redundant args
        result, _, _, _ = elastic_transform_3d(transformed_image_3d,
                                               alpha=self.alpha,
                                               sigma=self.sigma, order=order,
                                               random_state=self.random_state,
                                               dx=-self.dx, dy=-self.dy, dz=-self.dz)
        assert len(result.shape) == 3
        assert result.shape == self.image_shape
        return result

    def transform_4d(self, image_4d):
        assert len(image_4d.shape) == 4
        assert image_4d[0].shape == self.image_shape
        # TODO remove redundant args
        result, _, _, _ = elastic_transform_4d(image_4d, alpha=self.alpha,
                                               sigma=self.sigma,
                                               order=self.order,
                                               random_state=self.random_state, dx=self.dx,
                                               dy=self.dy, dz=self.dz)
        assert len(result.shape) == 4
        assert result[0].shape == self.image_shape
        return result

    def restore_4d(self, transformed_image_4d, order=None):
        assert len(transformed_image_4d.shape) == 4
        assert transformed_image_4d[0].shape == self.image_shape
        if order is None:
            order = self.order
        # TODO remove redundant args
        result, _, _, _ = elastic_transform_4d(transformed_image_4d,
                                               alpha=self.alpha,
                                               sigma=self.sigma, order=order,
                                               random_state=self.random_state,
                                               dx=-self.dx, dy=-self.dy, dz=-self.dz)
        assert len(result.shape) == 4
        assert result[0].shape == self.image_shape
        return result

    def draw_displacement_forces(self, z_projection, transpose=True, reverse=False):
        m = -1. if reverse else 1.
        visualisation.draw_displacement_forces(self.image_shape, dx=m * self.dx[:, :, z_projection],
                                               dy=m * self.dy[:, :, z_projection],
                                               transpose=transpose)


def elastic_transform(image, alpha, sigma, order=1, random_state=None, dx=None, dy=None):
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    def generate_displacement_():
        return generate_displacement(shape, random_state, alpha, sigma)

    dx = generate_displacement_() if dx is None else dx
    dy = generate_displacement_() if dy is None else dy

    xs = np.arange(shape[1])
    ys = np.arange(shape[0])
    x, y = np.meshgrid(xs, ys, indexing='xy')
    indices = [y + dy, x + dx]

    return map_coordinates(image, indices, order=order), dx, dy


def elastic_transform_NEW2(image, alpha, sigma, order=1, random_state=None, dx=None, dy=None):
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    def generate_displacement_():
        return generate_displacement_2(shape, random_state, alpha, sigma)

    dx, unfiltered_dx = generate_displacement_() if dx is None else dx
    dy, unfiltered_dy = generate_displacement_() if dy is None else dy

    xs = np.arange(shape[1])
    ys = np.arange(shape[0])
    x, y = np.meshgrid(xs, ys, indexing='xy')
    indices = [y + dy, x + dx]
    return map_coordinates(image, indices, order=order), dx, dy, unfiltered_dx, unfiltered_dy


def elastic_transform_3d(image, alpha, sigma, order=1, random_state=None, dx=None, dy=None,
                         dz=None):
    shape = image.shape
    assert len(shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    def generate_displacement_():
        return generate_displacement(shape, random_state, alpha, sigma)

    dx = generate_displacement_() if dx is None else dx
    dy = generate_displacement_() if dy is None else dy
    dz = generate_displacement_() if dz is None else dz

    xs = np.arange(shape[1])
    ys = np.arange(shape[0])
    zs = np.arange(shape[2])
    x, y, z = np.meshgrid(xs, ys, zs, indexing='xy')
    indices = [y + dy, x + dx, z + dz]
    image_as_numpy = torch.as_tensor(image).numpy()
    mapped = torch.as_tensor(map_coordinates(image_as_numpy, indices, order=order))
    return mapped, dx, dy, dz


def elastic_transform_4d(image_4d, alpha, sigma, order=1, random_state=None, dx=None,
                         dy=None,
                         dz=None):
    """Accepts [C, H, W, D]. Transforms channels equally and independently, then merges back."""
    shape = image_4d.shape
    num_channels = 4
    assert len(shape) == 4
    assert shape[0] == num_channels

    if random_state is None:
        random_state = np.random.RandomState(None)

    def generate_displacement_():
        return generate_displacement(shape[1:], random_state, alpha, sigma)

    dx = generate_displacement_() if dx is None else dx
    dy = generate_displacement_() if dy is None else dy
    dz = generate_displacement_() if dz is None else dz

    modalities = []
    for channel in range(num_channels):
        t, _, _, _ = elastic_transform_3d(image=image_4d[channel], alpha=alpha,
                                          sigma=sigma,
                                          order=order,
                                          random_state=random_state, dx=dx, dy=dy, dz=dz)
        modalities.append(t)

    modalities = torch.stack(modalities)
    print(f"modalities shape: {modalities.shape}")
    return modalities, dx, dy, dz


def generate_displacement(shape, random_state, alpha, sigma):
    original_displacement = (random_state.rand(*shape) * 2 - 1) * alpha
    return gaussian_filter(original_displacement, sigma, mode="constant", cval=0)


def generate_displacement_2(shape, random_state, alpha, sigma):
    original_displacement = (random_state.rand(*shape) * 2 - 1) * alpha
    return gaussian_filter(original_displacement, sigma,
                           mode="constant", cval=0), original_displacement


def elastic_transform_3d_HIGHER_DEF(image, alpha, sigma, order=1, random_state=None, dx=None,
                                    dy=None, dz=None):
    shape = image.shape
    assert len(shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    factor = 4
    new_shape = (np.asarray(shape) * factor).tolist()

    scaled_up = skimage.transform.resize(image, new_shape, mode="edge")

    def generate_displacement_():
        return generate_displacement(new_shape, random_state, alpha, sigma)

    dx = generate_displacement_() if dx is None else dx
    dy = generate_displacement_() if dy is None else dy
    dz = generate_displacement_() if dz is None else dz

    xs = np.arange(new_shape[1])
    ys = np.arange(new_shape[0])
    zs = np.arange(new_shape[2])
    x, y, z = np.meshgrid(xs, ys, zs, indexing='xy')
    indices = [y + dy, x + dx, z + dz]
    image_as_numpy = torch.as_tensor(scaled_up).numpy()
    mapped = map_coordinates(image_as_numpy, indices, order=order)

    scaled_down = skimage.transform.resize(mapped, shape, mode="edge")

    result = torch.as_tensor(scaled_down)
    return result, dx, dy, dz

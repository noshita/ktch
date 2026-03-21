"""Shared fixtures for plot tests."""

# Copyright 2026 Koji Noshita
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

#
# Mock descriptors
#


class MockDescriptor2D:
    """Mock 2D curve descriptor (EFA-like).

    Returns unit circles of shape ``(n, 50, 2)``.
    """

    def inverse_transform(self, X):
        n = X.shape[0]
        t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        coords = np.zeros((n, 50, 2))
        for i in range(n):
            coords[i, :, 0] = np.cos(t)
            coords[i, :, 1] = np.sin(t)
        return coords


class MockDescriptor3DCurve:
    """Mock 3D curve descriptor.

    Returns helices of shape ``(n, 50, 3)``.
    """

    def inverse_transform(self, X):
        n = X.shape[0]
        t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        coords = np.zeros((n, 50, 3))
        for i in range(n):
            coords[i, :, 0] = np.cos(t)
            coords[i, :, 1] = np.sin(t)
            coords[i, :, 2] = t / (2 * np.pi)
        return coords


class MockDescriptor3DSurface:
    """Mock 3D surface descriptor (SHA-like).

    Returns unit spheres of shape ``(n, 10, 20, 3)``.
    """

    def inverse_transform(self, X):
        n = X.shape[0]
        m, k = 10, 20
        theta = np.linspace(0, np.pi, m)
        phi = np.linspace(0, 2 * np.pi, k)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        coords = np.zeros((n, m, k, 3))
        for i in range(n):
            coords[i, :, :, 0] = np.sin(T) * np.cos(P)
            coords[i, :, :, 1] = np.sin(T) * np.sin(P)
            coords[i, :, :, 2] = np.cos(T)
        return coords


#
# Fixtures
#


@pytest.fixture
def mock_descriptor_2d():
    """Return a mock 2D curve descriptor."""
    return MockDescriptor2D()


@pytest.fixture
def mock_descriptor_3d_curve():
    """Return a mock 3D curve descriptor."""
    return MockDescriptor3DCurve()


@pytest.fixture
def mock_descriptor_3d_surface():
    """Return a mock 3D surface descriptor."""
    return MockDescriptor3DSurface()

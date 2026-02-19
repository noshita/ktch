"""Tests for plot._base module.

This module tests the shared dependency management utilities in ktch.plot._base,
which centralizes optional dependency handling for matplotlib, seaborn, and plotly.
"""

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

from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestBaseModuleStructure:
    """Test the structure and exports of _base module."""

    def test_module_has_require_dependencies(self):
        """Test that _base.py exports require_dependencies."""
        from ktch.plot import _base

        assert hasattr(_base, "require_dependencies")
        assert callable(_base.require_dependencies)

    def test_lazy_attributes_accessible(self):
        """Test that plt, sns, go, px are accessible as lazy attributes."""
        from ktch.plot import _base

        # These should be accessible via __getattr__ without raising
        for attr in ("plt", "sns", "go", "px"):
            # Should not raise AttributeError
            getattr(_base, attr)


class TestRequireDependencies:
    """Test the require_dependencies function."""

    def test_with_available_deps(self):
        """Test require_dependencies when dependencies are available."""
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")

        from ktch.plot._base import require_dependencies

        # Must not raise
        require_dependencies("matplotlib")
        require_dependencies("seaborn")
        require_dependencies("matplotlib", "seaborn")

    def test_with_missing_deps(self):
        """Test require_dependencies raises error for missing dependencies."""
        from ktch.plot import _base

        with patch.dict(_base._DEPS, {"matplotlib": None}):
            with pytest.raises(ImportError, match="matplotlib"):
                _base.require_dependencies("matplotlib")

        with patch.dict(
            _base._DEPS, {"matplotlib": None, "seaborn": None}
        ):
            with pytest.raises(
                ImportError, match="matplotlib.*seaborn|seaborn.*matplotlib"
            ):
                _base.require_dependencies("matplotlib", "seaborn")

    def test_error_messages_helpful(self):
        """Test that error messages provide installation instructions."""
        from ktch.plot import _base

        with patch.dict(_base._DEPS, {"matplotlib": None}):
            with pytest.raises(ImportError) as exc_info:
                _base.require_dependencies("matplotlib")

            error_msg = str(exc_info.value)
            assert "matplotlib" in error_msg
            assert "pip install ktch[plot]" in error_msg
            assert "conda install ktch-plot" in error_msg

    def test_unknown_dep_name_raises_value_error(self):
        """Test that unknown dependency names raise ValueError."""
        from ktch.plot._base import require_dependencies

        with pytest.raises(ValueError, match="Unknown dependency"):
            require_dependencies("matplotlb")  # typo

        with pytest.raises(ValueError, match="Unknown dependency"):
            require_dependencies("pandas")

    def test_unknown_dep_lists_valid_names(self):
        """Test that ValueError message lists valid dependency names."""
        from ktch.plot._base import require_dependencies

        with pytest.raises(ValueError) as exc_info:
            require_dependencies("nonexistent")

        error_msg = str(exc_info.value)
        assert "matplotlib" in error_msg
        assert "seaborn" in error_msg
        assert "plotly" in error_msg


class TestPlotlySupport:
    """Test plotly support including both go and px."""

    def test_plotly_imports(self):
        """Test that plotly imports work correctly when installed."""
        from ktch.plot import _base

        try:
            import plotly.express as px_direct
            import plotly.graph_objects as go_direct

            assert _base.go is go_direct
            assert _base.px is px_direct
        except ImportError:
            assert _base.go is None
            assert _base.px is None


class TestSubmoduleIntegration:
    """Test that _pca and _kriging modules correctly use _base."""

    def test_pca_module_uses_base(self):
        """Test that _pca.py imports require_dependencies from _base."""
        from ktch.plot import _pca

        assert hasattr(_pca, "require_dependencies")

    def test_kriging_module_uses_base(self):
        """Test that _kriging.py imports require_dependencies from _base."""
        from ktch.plot import _kriging

        assert hasattr(_kriging, "require_dependencies")

    def test_pca_function_requires_deps(self):
        """Test that PCA plotting function checks dependencies."""
        from ktch.plot import _base, _pca

        mock_pca = Mock()
        mock_pca.n_components_ = 3
        mock_pca.explained_variance_ratio_ = np.array([0.5, 0.3, 0.2])

        with patch.dict(
            _base._DEPS, {"matplotlib": None, "seaborn": None}
        ):
            with pytest.raises(
                ImportError, match="matplotlib.*seaborn|seaborn.*matplotlib"
            ):
                _pca.explained_variance_ratio_plot(mock_pca)

    def test_kriging_function_requires_deps(self):
        """Test that kriging plotting function checks dependencies."""
        from ktch.plot import _base, _kriging

        x_reference = np.array([[0, 0], [1, 0], [0, 1]])
        x_target = np.array([[0.1, 0], [1, 0.1], [0, 1]])

        with patch.dict(_base._DEPS, {"matplotlib": None}):
            with pytest.raises(ImportError, match="matplotlib"):
                _kriging.tps_grid_2d_plot(x_reference, x_target)


class TestBackwardCompatibility:
    """Test that the refactoring maintains backward compatibility."""

    def test_public_api_unchanged(self):
        """Test that public API functions are still available."""
        from ktch import plot

        assert hasattr(plot, "explained_variance_ratio_plot")
        assert hasattr(plot, "tps_grid_2d_plot")

    def test_module_imports_work(self):
        """Test that importing plot modules doesn't fail."""
        import ktch.plot._kriging
        import ktch.plot._pca

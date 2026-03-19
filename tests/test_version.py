"""Tests for UCRS version metadata."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import ucrs

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


class TestVersion:
    """Test version information."""

    def test_version_exists(self) -> None:
        """Test that __version__ attribute exists."""
        assert hasattr(ucrs, "__version__")

    def test_version_is_string(self) -> None:
        """Test that __version__ is a string."""
        assert isinstance(ucrs.__version__, str)

    def test_version_not_empty(self) -> None:
        """Test that __version__ is not an empty string."""
        assert ucrs.__version__ != ""

    def test_version_format(self) -> None:
        """Test that __version__ has expected format or is 'unknown'."""
        version = ucrs.__version__

        # Version should be either 'unknown' or follow semantic versioning pattern
        if version != "unknown":
            # Basic check: should contain digits
            assert any(char.isdigit() for char in version), \
                f"Version '{version}' should contain digits or be 'unknown'"

            # Should not contain spaces
            assert " " not in version, \
                f"Version '{version}' should not contain spaces"

    def test_version_matches_pyproject(self) -> None:
        """Test that installed version matches pyproject.toml."""
        with open(PYPROJECT, "rb") as f:
            pyproject_version = tomllib.load(f)["project"]["version"]
        assert ucrs.__version__ == pyproject_version, (
            f"Installed version '{ucrs.__version__}' does not match "
            f"pyproject.toml version '{pyproject_version}'. "
            f"Run: pip install -e . to sync."
        )

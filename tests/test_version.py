"""Tests for UCRS version metadata."""

from __future__ import annotations

import ucrs


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

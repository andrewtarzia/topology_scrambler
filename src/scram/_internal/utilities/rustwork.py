"""Module for working with rustworkx."""

from dataclasses import dataclass


@dataclass
class Rustwork:
    def get_connected_components(self) -> int:
        """Get the number of connected components."""

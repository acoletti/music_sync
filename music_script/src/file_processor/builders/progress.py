from abc import ABC, abstractmethod
from typing import Optional, Any
from tqdm import tqdm

class ProgressBuilder(ABC):
    """Base class for progress display builders."""
    def __init__(self):
        self._progress_bar: Optional[tqdm] = None
        self._total: int = 0
        self._description: str = ""
        self._unit: str = "items"

    def set_total(self, total: int) -> 'ProgressBuilder':
        """Set the total number of items to process."""
        self._total = total
        return self

    def set_description(self, description: str) -> 'ProgressBuilder':
        """Set the description for the progress bar."""
        self._description = description
        return self

    def set_unit(self, unit: str) -> 'ProgressBuilder':
        """Set the unit for the progress bar."""
        self._unit = unit
        return self

    @abstractmethod
    def build(self) -> Any:
        """Build and return the progress display object."""
        pass

class TqdmProgressBuilder(ProgressBuilder):
    """Builder for tqdm progress bars."""
    def build(self) -> tqdm:
        """Build and return a tqdm progress bar."""
        self._progress_bar = tqdm(
            total=self._total,
            desc=self._description,
            unit=self._unit,
            mininterval=0.1,  # Update at least every 0.1 seconds
            maxinterval=1.0,  # Update at most every 1 second
            miniters=1,       # Update for every iteration
            dynamic_ncols=True,  # Adjust width based on terminal size
            leave=True        # Keep the progress bar after completion
        )
        return self._progress_bar

class ProgressManager:
    """Manager for handling progress display."""
    def __init__(self):
        self._current_progress: Optional[tqdm] = None

    def create_progress(self, total: int, description: str, unit: str = "items") -> tqdm:
        """Create a new progress bar."""
        builder = TqdmProgressBuilder()
        self._current_progress = (
            builder
            .set_total(total)
            .set_description(description)
            .set_unit(unit)
            .build()
        )
        return self._current_progress

    def update(self, increment: int = 1) -> None:
        """Update the current progress bar."""
        if self._current_progress:
            self._current_progress.update(increment)

    def close(self) -> None:
        """Close the current progress bar."""
        if self._current_progress:
            self._current_progress.close()
            self._current_progress = None 
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class BaseDataset(ABC):
    """Abstract dataset interface for synthetic or real datasets."""
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load or generate the dataset. Returns (X, y)."""
        pass

    @abstractmethod
    def get_input_dim(self) -> int:
        """Return input dimension (for NN layer sizing)."""
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """Return number of output classes."""
        pass

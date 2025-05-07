from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class LinearSystem:
    matrix: NDArray
    rhs: NDArray
    solution: NDArray | None = None

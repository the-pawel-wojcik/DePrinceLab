from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class LinearSystem:
    matrix: NDArray
    rhs: NDArray
    solution: NDArray | None = None


    def __str__(self) -> str:
        lstr = f"{self.matrix=}\n" 
        lstr += f"{self.solution=}\n" 
        lstr += f"{self.rhs=}\n" 
        return lstr

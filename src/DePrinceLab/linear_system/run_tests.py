from DePrinceLab.linear_system.test_0 import test_atomic_energy
from DePrinceLab.linear_system.test_1 import test_diagonal_dominant
from DePrinceLab.linear_system.test_random import test_random


def main():
    test_atomic_energy()
    test_diagonal_dominant()
    test_random()


if __name__ == "__main__":
    main()

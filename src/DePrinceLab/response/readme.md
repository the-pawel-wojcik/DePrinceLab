# Polarizability of water

[CCCBDB](https://cccbdb.nist.gov/exp2x.asp) says that the experimental electric
dipole polarizability of water is 1.501 Å³, which I converted to 10.129 Bohr³.

The calculation in this directory uses HF/cc-pVDZ, which is also available in
[CCCBDB](https://cccbdb.nist.gov/polcalc3x.asp?method=1&basis=5)
|  ɑ (Å³) |   x   |   y   |   z   |
| ------- | ----- | ----- | ----- |
|    x    | 0.451 | 0.000 | 0.000 |
|    y    | 0.000 | 0.992 | 0.000 |
|    z    | 0.000 | 0.000 | 0.738 |

which I can translate to 
|  ɑ (au³) |   x   |   y   |   z   |
| -------- | ----- | ----- | ----- |
|     x    | 3.043 | 0.000 | 0.000 |
|     y    | 0.000 | 6.694 | 0.000 |
|     z    | 0.000 | 0.000 | 4.980 |
Things check out for now.

## Run calcualtions
```bash
python main.py > result.txt
```

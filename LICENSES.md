# Licenses

This repository's own code is MIT licensed - see `LICENSE`.

The pinned submodule `vendor/plant-shift-oee-report` (commit
`7e378facb410b529d8e2d72d329daf2eee515161`) is this company's own synthetic
demo, MIT licensed (its own `LICENSE` file). Only two functions are imported
from it; it is not modified.

## Python dependencies (direct and transitive)

Taken from the installed distributions' metadata (`License-Expression`, or
`License` / trove classifier where no expression is published) in a fresh
venv built from `requirements.lock`. Every package's own licence is
OSI-approved.

| Package | Version | Direct? | Licence (installed metadata) | OSI-approved |
|---|---|---|---|---|
| numpy | 2.5.3 | direct | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | package licence BSD-3-Clause: yes (see bundled-components note) |
| scipy | 1.18.1 | direct | BSD License (classifier), BSD-3-Clause text | yes (see bundled-components note) |
| scikit-learn | 1.9.1 | direct | BSD-3-Clause | yes |
| duckdb | 1.5.5 | direct | MIT (classifier) | yes |
| PyYAML | 6.0.3 | direct | MIT | yes |
| pytest | 9.1.1 | direct (tests) | MIT | yes |
| joblib | 1.6.0 | transitive (scikit-learn) | BSD-3-Clause | yes |
| threadpoolctl | 3.7.0 | transitive (scikit-learn) | BSD-3-Clause | yes |
| narwhals | 2.26.0 | transitive (scikit-learn) | MIT | yes |
| cloudpickle | 3.1.2 | transitive (scikit-learn) | BSD-3-Clause | yes |
| iniconfig | 2.3.0 | transitive (pytest) | MIT | yes |
| pluggy | 1.6.0 | transitive (pytest) | MIT | yes |
| packaging | 26.3 | transitive (pytest) | Apache-2.0 OR BSD-2-Clause | yes |
| Pygments | 2.21.0 | transitive (pytest) | BSD-2-Clause | yes |

## Bundled components inside binary wheels (flagged, decision pending)

The numpy, scipy and scikit-learn **wheels** vendor third-party code whose
licences are declared in their `*.dist-info/licenses` / `LICENSE.txt`. Most
are OSI-approved (BSD-3-Clause OpenBLAS, GPL-3.0-or-later WITH
GCC-exception-3.1 libgfortran/libgomp, LGPL-2.1-or-later libquadmath, Zlib,
0BSD, MIT, Apache-2.0/BSD-3-Clause Highway). Some are **not on the OSI
list**, even though they are permissive:

| Where | Component | Licence | OSI list |
|---|---|---|---|
| numpy wheel | Highway `hwy/contrib/random/random-inl.h` (one file) | CC0-1.0 (public-domain dedication) | not OSI-approved |
| numpy, scipy wheels (OpenBLAS) | reference LAPACK | BSD-3-Clause-Open-MPI (BSD-3 variant) | not listed separately |
| scipy wheel | Qhull (`scipy/spatial/qhull_src/COPYING_QHULL.txt`) | Qhull licence (permissive) | not OSI-approved |
| scipy wheel | DOP853 (`scipy/integrate/LICENSE_DOP`) | BSD-style (Hairer) | not listed separately |

The protocol requires scikit-learn, which depends on numpy and scipy, so
these components cannot be dropped while following the protocol. None of
them is paid or closed-source. Whether the OSI-only rule covers vendored
sub-components as well as the packages is a decision for the CEO/reviewer.
It is recorded here, not decided here.

No paid or closed-source dependency is used anywhere in this repository.

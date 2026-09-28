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

## Dashboard dependencies (`dashboard/requirements.txt`, direct and transitive)

Additional to the table above; only used by `dashboard/` and `monitoring/`,
never by the frozen `src/` modules or `runner/`. Taken from installed
distribution metadata in the same venv. Every package's own licence is
OSI-approved; none is paid or closed-source.

| Package | Version | Direct? | Licence (installed metadata) | OSI-approved |
|---|---|---|---|---|
| streamlit | 1.64.0 | direct | Apache-2.0 | yes |
| altair | 6.3.0 | direct | BSD License (classifier) | yes |
| pandas | 3.0.6 | direct | BSD License (classifier) | yes |
| MarkupSafe | 3.0.3 | transitive (jinja2) | BSD-3-Clause | yes |
| anyio | 4.15.1 | transitive (starlette) | MIT | yes |
| attrs | 26.1.0 | transitive (jsonschema) | MIT | yes |
| certifi | 2026.7.22 | transitive (requests) | Mozilla Public License 2.0 (MPL 2.0) | yes |
| charset-normalizer | 3.5.1 | transitive (requests) | MIT | yes |
| click | 8.5.0 | transitive (streamlit) | BSD-3-Clause | yes |
| h11 | 0.16.0 | transitive (uvicorn) | MIT | yes |
| httptools | 0.8.0 | transitive (uvicorn) | MIT | yes |
| idna | 3.20 | transitive (requests) | BSD-3-Clause | yes |
| itsdangerous | 2.2.0 | transitive (streamlit) | BSD License (classifier) | yes |
| jinja2 | 3.1.6 | transitive (streamlit) | BSD License (classifier) | yes |
| jsonschema | 4.26.0 | transitive (altair) | MIT | yes |
| jsonschema-specifications | 2025.9.1 | transitive (jsonschema) | MIT | yes |
| pillow | 12.3.0 | transitive (streamlit) | MIT-CMU (HPND family) | yes |
| protobuf | 7.36.2 | transitive (streamlit) | 3-Clause BSD | yes |
| pyarrow | 25.0.1 | transitive (streamlit) | Apache-2.0 | yes |
| pydeck | 0.9.3 | transitive (streamlit) | Apache License 2.0 | yes |
| python-dateutil | 2.9.0.post0 | transitive (pandas) | BSD License / Apache Software License (dual) | yes |
| python-multipart | 0.0.32 | transitive (streamlit) | Apache Software License | yes |
| referencing | 0.37.0 | transitive (jsonschema) | MIT | yes |
| requests | 2.34.2 | transitive (streamlit) | Apache Software License | yes |
| rpds-py | 2026.6.3 | transitive (jsonschema) | MIT | yes |
| six | 1.17.0 | transitive (python-dateutil) | MIT | yes |
| starlette | 1.7.0 | transitive (streamlit) | BSD-3-Clause | yes |
| toml | 0.10.2 | transitive (streamlit) | MIT | yes |
| typing-extensions | 4.16.0 | transitive (streamlit) | PSF-2.0 | yes |
| urllib3 | 2.8.0 | transitive (requests) | MIT | yes |
| uvicorn | 0.54.0 | transitive (streamlit) | BSD-3-Clause | yes |
| watchdog | 6.0.0 | transitive (streamlit) | Apache Software License | yes |
| websockets | 16.1.1 | transitive (streamlit) | BSD-3-Clause | yes |

Not hash-locked like `requirements.lock` (see `dashboard/requirements.txt`'s
own header for why); version-pinned only.

## Bundled components inside binary wheels (covered by the licence exception below)

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

## Licence exception (company decision `20-decisions/2026-09-28-licence-exception-wheel-bundled-components.md`)

Accepted 2026-09-28, CEO by delegation, with Astra concurring (vault 9fd116d §2); the CEO may still veto. This is a
standing, **enumerated** exception to the company's OSI-only rule, and it covers only the components listed below, in
the pinned wheels named. It does not declare these licences OSI-approved.

This repository does **not** vendor or redistribute any of these wheels: users install them from PyPI via
`requirements*.txt`. Bundled notices stay exactly as shipped inside each wheel, unmodified. Where screenshots or
charts render a bundled font, the font licence permits using it to produce documents and images.

| Component | Package / version | File(s) in the wheel | Licence | Licence text / source | Option chosen | Obligations |
|---|---|---|---|---|---|---|
| Highway random-inl.h | numpy 2.5.3 | `hwy/contrib/random/random-inl.h` (compiled into numpy) | CC0-1.0 | `numpy-2.5.3.dist-info/licenses/numpy/_core/src/highway/LICENSE` | n/a | none (public-domain dedication); notice kept |
| Reference LAPACK (via OpenBLAS) | numpy 2.5.3, scipy 1.18.1 | compiled into the bundled OpenBLAS | BSD-3-Clause-Open-MPI | `numpy-2.5.3.dist-info/licenses/LICENSE.txt`, `scipy-1.18.1.dist-info/LICENSE.txt` | n/a | keep copyright and licence notice (as shipped) |
| Qhull | scipy 1.18.1 | `scipy/spatial/qhull_src/COPYING_QHULL.txt` (compiled `_qhull`) | Qhull licence | that file | n/a | keep notice; modified versions must be marked (we modify nothing) |
| DOP853 | scipy 1.18.1 | `scipy/integrate/LICENSE_DOP` | BSD-style (Hairer) | that file | n/a | keep notice |

The twin's cover chart is rendered with matplotlib in the vault tooling, not by this repository, which does not depend on matplotlib.

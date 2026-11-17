# Licenses and attribution

This extraction preserves existing source headers, notices and licenses.
It assigns no new umbrella license to project-owned kernels or patches.

- `vendor/internal-kyber/LICENSE` and `vendor/pq-crystals-kyber/LICENSE`:
  public domain / Apache-2.0 alternatives as stated by the upstream project.
- Keccak sources retain the authors' public-domain/CC0 notices in their files.
- wolfSSL is fetched at its pinned commit by `tools/build_single.py` with its
  GPL and other component license files intact. The B1 patches create derived
  copies while preserving those notices. Consult that license before reuse.
- liboqs is fetched at its pinned commit, retaining `LICENSE.txt` and component
  licenses in its source tree.

The B4 kernel and polynomial sources retain their existing attribution and
originate from the author's avxmtls implementation.
See `provenance.json` for the source inventory and hashes. Redistribution
of generated linked artifacts must account for all linked component licenses.

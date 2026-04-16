# overlay

Files in this directory are copied into the generated LMCache workspace after
the upstream tarball is unpacked and before patches are applied.

Layout:

- `common/`
  Shared Wings-owned logic copied into every workspace.
- `ascend/`
  Ascend-only additions layered on top of `common/`.

Use this area for Wings-owned logic that should not live as large diffs against
upstream LMCache.

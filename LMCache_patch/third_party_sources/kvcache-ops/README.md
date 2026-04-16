# third_party_sources/kvcache-ops

This directory stores the prepared `kvcache-ops` source snapshot used by the
Ascend build path.

Current expected input:

- a tarball prepared from `ssh://git@git.codehub.xfusion.com:2222/OpenSourceCenter/kvcache-ops.git`
- the tarball file name locked in `manifest/lmcache.lock.json`

The current workflow does not keep a vendored `kvcache-ops` checkout in the
repo. `python3 install.py prepare-ascend-sources` clones the locked ref over
SSH, archives it into this directory, and later expands it into the generated
workspace as `third_party/kvcache-ops`.

If you change the upstream repository, ref, tarball name, or expected extracted
directory name, update `extra_sources.kvcache_ops` in
`manifest/lmcache.lock.json`.

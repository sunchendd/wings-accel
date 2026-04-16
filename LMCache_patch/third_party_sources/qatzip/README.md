# third_party_sources/qatzip

This directory stores the pinned `QATzip` source tarball for the common-path
`kv-agent` build.

Required artifact:

- `https://artifactrepo.wux-g.tools.xfusion.com/artifactory/opensource_general/QAT/v1.3.2/package/QATzip-1.3.2.tar.gz`

Supported input:

- `QATzip-1.3.2.tar.gz`

Expected follow-up after placing the tarball:

- update `extra_sources.qatzip.tarball_name`
- update `extra_sources.qatzip.tarball_sha256`
- update `extra_sources.qatzip.root_dir_name_in_tar`
- update `extra_sources.qatzip.source_url`

`scripts/build_kv_agent.sh` only uses this snapshot when `qatzip.h` or
`libqatzip.so` is not already available in standard paths or via explicit
`QATZIP_INCLUDE_DIR` and `QATZIP_LIB_DIR`.

The build does not auto-discover live `QATzip` worktrees outside this locked
offline snapshot.

Download the pinned tarball in a clean build environment:

```bash
cd LMCache_patch
python3 install.py prepare-common-sources
```
# upstream_sources

This directory stores the pinned LMCache source tarball consumed by the
patch-first workflow.

Required artifact:

- `https://artifactrepo.wux-g.tools.xfusion.com/artifactory/opensource_general/LMCache/v0.3.15/package/LMCache-0.3.15.tar.gz`

You can still inspect or reuse an existing tarball here locally, but the source
of truth for build preparation is the pinned artifact above.

## Example

```text
LMCache_patch/upstream_sources/
└── LMCache-0.3.15.tar.gz
```

## Required Follow-Up

After preparing or replacing a tarball here, update
[manifest/lmcache.lock.json](/workspace/wings-accel/LMCache_patch/manifest/lmcache.lock.json):

- `tarball_name`
- `tarball_sha256`
- `root_dir_name_in_tar`
- `source_url`
- `version`
- `patchset_version`

Ascend-specific extra sources such as `kvcache-ops` do not belong here. Put
those under
[third_party_sources/](/workspace/wings-accel/LMCache_patch/third_party_sources).

## Helpful Commands

Compute the tarball hash:

```bash
sha256sum LMCache_patch/upstream_sources/LMCache-0.3.15.tar.gz
```

List top-level entries in the tarball:

```bash
tar -tf LMCache_patch/upstream_sources/LMCache-0.3.15.tar.gz | head
```

Download the pinned tarball in a clean build environment:

```bash
cd LMCache_patch
python3 install.py prepare-common-sources
```

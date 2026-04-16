# patches/ascend

This directory contains Ascend-only hook patches. They are applied after the
shared `patches/common/<version>/` patch set.

Keep these patches small and focused on:

- build integration for Ascend-only dependencies
- minimal runtime hooks for NPU/Ascend enablement
- explicit unsupported-QAT behavior on Ascend

Do not duplicate common changes here.

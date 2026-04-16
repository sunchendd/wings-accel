# overlay/ascend

This directory is reserved for Ascend-only source additions layered on top of
`overlay/common`.

Intended contents:

- `torch_npu` and CANN-specific Python integration helpers
- Ascend connector glue
- Ascend system-detection helpers
- build helpers that should not affect the common GPU/x86 path

Keep shared feature logic in `overlay/common`. Only place code here when it is
truly Ascend-specific.

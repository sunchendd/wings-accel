# third_party

This directory stores vendored third-party source trees that are copied into
the generated LMCache workspace.

Rules:

- keep upstream third-party code separate from `lmcache/v1/wings_ext`
- prefer adapting third-party code through thin Wings hooks instead of editing
  it directly
- when local changes to a vendored dependency are unavoidable, keep them small
  and document them nearby

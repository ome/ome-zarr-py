Changelog
=========

Unreleased
----------

Bug Fixes
~~~~~~~~~
- Fixed chunking when a scalar value for chunks is given. Previously
  passing ``storage_options={"chunks": <int>}`` only set the chunk
  size of the final dimension. Now the chunk size of all dimensions is
  set to this value, which is identical behaviour to ``zarr-python``.

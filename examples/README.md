# Examples

This folder contains minimal examples and notes about expected input formats.

Input columns:
- aaSeqCDR3
- readCount
(optional) readFraction

File formats:
- TSV, CSV, or extension-less (separator auto-detected).

Null training expects replicate pairs named like:
  <subject>_<time>-1 and <subject>_<time>-2
## Toy dataset included

- `examples/toy_null/` contains small replicate pairs for null training
- `examples/toy_stim/` contains a minimal UNSTIM/STIM example

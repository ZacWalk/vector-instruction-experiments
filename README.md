# SIMD optimisation and performance 

Some experiments implementing SIMD algorithms on SSE2 and NEON intrinsics.

Implements a routine to calculate the distance (difference) between two 64-byte vectors. Using C, SSE2 and NEON intrinsics.

Current perf results:

| | C	| SSE	| NEON |
| --- | ---	| ---	| --- |
| ARM64 PC emulation	| 4,032,500	| 2,642,500	| |
| ARM64 PC native	| 3,463,900	| | 1,812,900 |
| Intel PC | 1,763,700	| 1,214,600	 | |

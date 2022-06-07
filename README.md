# SIMD optimisation and performance 

Some experiments implementing SIMD algorithms on SSE2 and NEON intrinsics.

Implements a routine to calculate the distance (difference) between two 64-byte vectors. Using C, SSE2 and NEON intrinsics.

Performance (Milliseconds for 100,000,000 iterations):

| | C   | SSE | AVX2 | AVX512 | NEON |
| --- | --- | --- | --- | --- | --- |
| Intel PC x64 | 182 | 102 | 92 | 94 | 0 |
| Intel PC x86 | 188 | 155 | 159 | 203 | 0 |
| ARM64 PC | 3524 | 0 | 0 | 0 | 201 |
| ARM64 PC (Emulated x64) | 408 | 285 | 68 | 67 | 0 |
| ARM64 PC (Emulated x86) | 492 | 437 | 173 | 146 | 0 |
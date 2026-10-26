# matadd

Adds two `float` matrices, C = A + B, 9999 × 6001 (fixed in the source code). Only the shape of thread block (`bx × by`) changes.

## Run

```bash
make ARCH=sm_86
./matadd 16 16              # bx by, block shape
./matadd 16 16 --csv
./run.sh                    # reproduces the Result table
```

Every element of C is checked against the CPU.

## Result

RTX 3090, 9999 × 6001, kernel time only, 1 warm-up + 10 runs.
GB/s = 3 × 9999 × 6001 × 4 bytes ÷ kernel time.

| Block (bx × by) | GB/s | vs 16x16 | Note |
| --- | ---: | ---: | --- |
| 1x256 | 203 | 0.24 | every thread in a warp is on a different row; 4 useful bytes per 32-byte sector |
| 2x128 | 393 | 0.46 | a warp spans several rows, so each sector is only partly used |
| 4x64 | 720 | 0.84 | a warp spans several rows, so each sector is only partly used |
| 8x32 | 855 | 1.00 | |
| 16x16 | 855 | 1.00 | a warp reads at least 8 consecutive floats per row, filling whole 32-byte sectors |
| 32x8 | 854 | 1.00 | |
| 64x4 | 853 | 1.00 | |
| 128x2 | 853 | 1.00 | |
| 256x1 | 851 | 1.00 | |
| 8x8 | 824 | 0.96 | |
| 32x32 | 825 | 0.96 | |

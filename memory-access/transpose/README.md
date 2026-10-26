# transpose

Transposes a 4096×4096 `float` matrix, B = Aᵀ. A plain copy with the same
access pattern on both sides is the baseline for what the transposes could reach.

| Variant | How |
| --- | --- |
| copy-only | no transpose, contiguous read and write (baseline) |
| read-coalesce | reads along rows of A, writes down columns of B |
| write-coalesce | reads down columns of A, writes along rows of B |
| tiled | 32×32 tile through shared `[32][32]`, both global accesses contiguous |
| tiled-pad | same, shared `[32][33]` |

## Run

```bash
make ARCH=sm_86
./transpose 4096 4096       # M N, matrix shape
./transpose 4096 4096 --csv
./run.sh                    # reproduces the Result table
```

Every element of B is checked against the CPU (copy-only is unchecked).

## Result

RTX 3090, 4096×4096, kernel time only, 1 warm-up + 10 runs.

GB/s = 2×M×N×4 bytes ÷ kernel time.

| Variant | GB/s | % of copy-only | Note |
| --- | ---: | ---: | --- |
| copy-only | 725 | 100 | |
| read-coalesce | 267 | 36.9 | writes jump a whole row apart; stores are not cached in L1 (write-through), so every 4-byte store is its own 32-byte sector write to L2 |
| write-coalesce | 508 | 70.1 | reads jump a whole row apart, but a loaded sector stays in L1, so the other 7 floats in it are hits for neighbouring threads |
| tiled | 496 | 68.5 | both global accesses are contiguous, but reading a shared column is a 32-way bank conflict |
| tiled-pad | 620 | 85.6 | one extra column shifts each row by one bank, removing the conflict |

Nsight Compute on the two naive kernels shows why scattered writes cost more than
scattered reads (global loads and stores at L1; 2.1M sectors is what a coalesced
access needs, 16.8M is 8× that):

| | load sectors | load L1 hit | store sectors | store L1 hit | sectors L1 → L2 (read / write) |
| --- | ---: | ---: | ---: | ---: | ---: |
| read-coalesce | 2.1M | 0 % | 16.8M (scattered) | 0 % | 2.1M / 16.8M |
| write-coalesce | 16.8M (scattered) | 87.5 % | 2.1M | 0 % | 2.1M / 2.1M |

A scattered load brings a whole 32-byte sector into L1, and the other 7 floats in it are
used by neighbouring threads soon after, so 7 of 8 sector accesses hit (87.5%) and
L2 sees only 2.1M sectors. L1 is write-through for global stores and keeps nothing it
stores, so a scattered store gets no such reuse: each 4-byte store goes to L2 as its own
sector write, 8× the coalesced count. L2 merges them before DRAM, so both kernels move
the same DRAM traffic (67 MB read, 64 MB write). The gap is the extra L1 → L2 store
traffic. The coalesced sides hit 0% because each sector is touched only once.

## NCU Command

```bash
ncu -k naiveReadCoalescedMatrixTranspose -s 1 -c 1 --metrics l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum ./transpose 4096 4096
```
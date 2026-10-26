# Builds every benchmark. Settings pass straight through to common/common.mk,
# so `make ARCH=sm_86`, `make REPS=50` and `make clean` all work from here.
#
#     make                 build everything
#     make gemm            build one group
#     make clean           remove every binary
#
# A single benchmark is quickest built in its own directory:
#
#     cd primitives/reduction && make && ./reduction 16777216

BENCH_DIRS := gemm/01-naive \
              gemm/02-tiled \
              gemm/03-register-tiling \
              gemm/04-tensor-core \
              gemm/cublas-gemm \
              intro/device-query \
              intro/divergence \
              intro/matadd \
              intro/vecadd \
              llm-inference/decoder \
              llm-inference/graph-launch \
              llm-inference/quantized-gemv \
              llm-inference/normalization \
              memory-access/p-chase \
              memory-access/stride \
              memory-access/transfer \
              memory-access/transpose \
              primitives/histogram \
              primitives/reduction \
              primitives/scan

GROUPS := $(sort $(foreach d,$(BENCH_DIRS),$(firstword $(subst /, ,$(d)))))

.PHONY: all clean $(BENCH_DIRS) $(GROUPS)

all: $(BENCH_DIRS)

$(BENCH_DIRS):
	$(MAKE) -C $@

# `make gemm` builds every directory under gemm/
$(GROUPS):
	$(foreach d,$(filter $@/%,$(BENCH_DIRS)),$(MAKE) -C $(d) &&) true

clean:
	$(foreach d,$(BENCH_DIRS),$(MAKE) -C $(d) clean &&) true

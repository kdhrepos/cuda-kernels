# Shared build settings for every benchmark directory.
#
# A leaf Makefile is normally just:
#
#     include ../../common/common.mk
#
# which builds one binary per .cu in that directory, named after the source.
# A directory needing more sets variables before the include:
#
#     LDLIBS := -lcublas
#     include ../../common/common.mk
#
# Overridable from the command line, so `make ARCH=sm_86` or
# `make NVCCFLAGS_EXTRA=-DREPS=50` work without editing anything:
#
#     NVCC       compiler                      (default nvcc)
#     ARCH       target architecture           (default sm_80, the A100s here)
#     OPT        optimisation level            (default -O3)
#     STD        C++ standard                  (default c++14, what CUDA 11 takes)
#     REPS       measured iterations per run   (unset: each source defaults to 10)
#     GEMM_CHECK_CORNER  how deep the GEMM CPU check goes (default 1024)
#     DEFINES    extra -D macros, e.g. DEFINES='TILE=32'
#     SRCS       sources to build              (default: every .cu here)
#     LDLIBS     extra libraries to link
#     NVCCFLAGS_EXTRA  anything else to pass through

COMMON_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

NVCC ?= nvcc
ARCH ?= sm_80
OPT  ?= -O3
STD  ?= c++14

NVCCFLAGS ?= $(OPT) -arch=$(ARCH) -std=$(STD)

# Tunables that exist in the sources as -D overrides. Named here so `make REPS=50`
# and `make GEMM_CHECK_CORNER=64` work, with DEFINES as the general escape hatch:
#   make DEFINES='TILE=32 REPS=5'
ifdef REPS
NVCCFLAGS += -DREPS=$(REPS)
endif
ifdef GEMM_CHECK_CORNER
NVCCFLAGS += -DGEMM_CHECK_CORNER=$(GEMM_CHECK_CORNER)
endif
NVCCFLAGS += $(addprefix -D,$(DEFINES))
NVCCFLAGS += $(NVCCFLAGS_EXTRA)

SRCS    ?= $(wildcard *.cu)
TARGETS ?= $(basename $(SRCS))

# Rebuild when a shared header changes, not just when the source does.
COMMON_HEADERS := $(wildcard $(COMMON_DIR)/*.cuh)

.PHONY: all clean
all: $(TARGETS)

$(TARGETS): %: %.cu $(COMMON_HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(TARGETS)

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
#     SRCS       sources to build              (default: every .cu here)
#     LDLIBS     extra libraries to link
#     NVCCFLAGS_EXTRA  anything else to pass through

COMMON_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

NVCC ?= nvcc
ARCH ?= sm_80
OPT  ?= -O3
STD  ?= c++14

NVCCFLAGS ?= $(OPT) -arch=$(ARCH) -std=$(STD)
ifdef REPS
NVCCFLAGS += -DREPS=$(REPS)
endif
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

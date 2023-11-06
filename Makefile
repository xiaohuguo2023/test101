# Compiler and options
HIPCC = hipcc

ROCM_INSTALL_DIR := /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

# Source files and output binary
SRCS = main.cpp
OUT = hello_world

# Compilation flags
CXXFLAGS = -std=c++11 --rocm-path=$(ROCM_INSTALL_DIR) -I$(HIP_INCLUDE_DIR)

all: $(OUT)

$(OUT): $(SRCS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(OUT)

clean:
	rm -f $(OUT)


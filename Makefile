# Compiler and options
HIPCC = hipcc

ROCM_INSTALL_DIR := /opt/rocm
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include

# Source files and output binary
SRCS = main.cpp
SRCS1 = addVectors.cpp
OUT = hello_world
OUT1= addvectors

# Compilation flags
CXXFLAGS = -std=c++11 --rocm-path=$(ROCM_INSTALL_DIR) -I$(HIP_INCLUDE_DIR)

all: $(OUT) $(OUT1)

$(OUT): $(SRCS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(OUT)

$(OUT1): $(SRCS1)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(OUT)
clean:
	rm -f $(OUT) $(OUT1)


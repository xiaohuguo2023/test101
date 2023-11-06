# Compiler and options
HIPCC = hipcc

# Source files and output binary
SRCS = main.cpp
OUT = hello_world

# Compilation flags
CXXFLAGS = -std=c++11

all: $(OUT)

$(OUT): $(SRCS)
	$(HIPCC) $(CXXFLAGS) $(SRCS) -o $(OUT)

clean:
	rm -f $(OUT)


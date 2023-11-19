# Compiler and Flags
CXX = g++ # or you can use clang++ or other C++ compilers
CXXFLAGS = -Wall -std=c++11 # adjust your C++ standard as needed
INC = -Isrc/core -Isrc/kernels -Ilibs
I = -Isrc/core -Isrc/kernels -Ilibs
# Directories
CORE_DIR = src/core
KERNELS_DIR = src/kernels
BUILD_DIR = build

# Files
OBJECTS = src/core/GPModel.o  src/kernels/KernelBase.o  src/kernels/RBFKernel.o

# Target binary
TARGET = gaussian_process

all: $(TARGET)

$(TARGET): $(BUILD_DIR)/main.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(I) $^ -o $@

$(BUILD_DIR)/main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(I) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

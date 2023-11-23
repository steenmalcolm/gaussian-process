# Compiler and Flags
CXX = g++ # or you can use clang++ or other C++ compilers
CXXFLAGS = -Wall -std=c++20 # adjust your C++ standard as needed
INC_DIRS := $(shell find src -type d) libs
INC := $(addprefix -I,$(INC_DIRS))
# INC = -Isrc/core -Isrc/kernels -Isrc/optimizers -Ilibs 
# Directories
BUILD_DIR = build

# Files
OBJECTS = src/optimizers/GradientDescent.o src/core/GPModel.o  src/kernels/KernelBase.o  src/kernels/RBFKernel.o 

# Target binary
TARGET = gaussian_process

all: $(TARGET)

$(TARGET): $(BUILD_DIR)/main.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INC) $^ -o $@

$(BUILD_DIR)/main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

clean:
	rm  $(BUILD_DIR)/* $(TARGET)

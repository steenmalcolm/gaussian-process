# Compiler and compiler flags
CXX = g++
CXXFLAGS = -Wall -pedantic -std=c++14
INC = -I../libs -Ikernels

# Object files
OBJS = core/GPModel.o kernels/KernelBase.o kernels/RBFKernel.o optimizers/GradientDescent.o

all: $(OBJS)

kernels/RBFKernel.o: kernels/RBFKernel.cpp kernels/RBFKernel.h kernels/KernelBase.h
	$(CXX) $(CXXFLAGS) $(INC) -c kernels/RBFKernel.cpp -o kernels/RBFKernel.o

kernels/KernelBase.o: kernels/KernelBase.cpp kernels/KernelBase.h
	$(CXX) $(CXXFLAGS) $(INC) -c kernels/KernelBase.cpp -o kernels/KernelBase.o

core/GPModel.o: core/GPModel.cpp core/GPModel.h kernels/KernelBase.h
	$(CXX) $(CXXFLAGS) $(INC) -c core/GPModel.cpp -o core/GPModel.o

optimizers/GradientDescent.o: optimizers/GradientDescent.cpp optimizers/GradientDescent.h
	$(CXX) $(CXXFLAGS) $(INC) -c optimizers/GradientDescent.cpp -o optimizers/GradientDescent.o

clean:
	rm -f $(OBJS)

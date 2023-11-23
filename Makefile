# Compiler settings - Can be customized.
CXX = g++
CXXFLAGS = -Wall -Iinclude -Ilibs -std=c++11

# Define the directories.
SRCDIR = src
OBJDIR = obj
LIBDIR = libs
INCDIR = include

# Define the target executable
TARGET = main

# Find all the CPP files in the src/ directory.
SRCS = $(wildcard $(SRCDIR)/*/*.cpp)

# For each CPP file, generate the corresponding object file.
OBJS = $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Default target.
all: $(TARGET)

# Link the target with all the object files.
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile the source files into object files.
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create the directory for object files.
$(OBJDIR):
	mkdir $@

# Clean up build.
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Include dependencies.
-include $(OBJS:.o=.d)

# Calculate dependencies.
$(OBJDIR)/%.d: $(SRCDIR)/%.cpp | $(OBJDIR)
	@$(CXX) $(CXXFLAGS) -MM -MT '$(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$<)' $< > $@

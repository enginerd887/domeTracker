# Makefile for Basler pylon sample program
.PHONY: all clean
	# The program to build
NAME       := blobTrack

# Installation directories for pylon
PYLON_ROOT ?= /opt/pylon5



# Rules for building
all: $(NAME)
# Build tools and flags
LD         := g++
CPPFLAGS   := -I /usr/local/tbb/include $(shell $(PYLON_ROOT)/bin/pylon-config --cflags)
CXXFLAGS   := -std=c++11#e.g., CXXFLAGS=-g -O0 for debugging
LDFLAGS    := -L /usr/local/tbb/lib/intel64/gcc4.8 $(shell $(PYLON_ROOT)/bin/pylon-config --libs-rpath)
LDLIBS     := -ltbb $(shell $(PYLON_ROOT)/bin/pylon-config --libs)

$(NAME): $(NAME).o
	$(LD)   -o $@ $^ $(LDFLAGS) $(LDLIBS) `pkg-config --cflags --libs opencv`

$(NAME).o: $(NAME).cpp
	$(CXX) $(CXXFLAGS) -O3 -c -o $@ $< $(CPPFLAGS)

clean:
	$(RM) $(NAME).o $(NAME)

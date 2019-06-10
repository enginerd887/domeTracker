# Makefile for Basler pylon sample program
.PHONY: all clean
	# The program to build
NAME       := blob

# Installation directories for pylon
PYLON_ROOT ?= /opt/pylon5



# Rules for building
all: $(NAME)
# Build tools and flags
LD         := g++
CPPFLAGS   := $(shell $(PYLON_ROOT)/bin/pylon-config --cflags)
CXXFLAGS   := #e.g., CXXFLAGS=-g -O0 for debugging
LDFLAGS    := $(shell $(PYLON_ROOT)/bin/pylon-config --libs-rpath)
LDLIBS     := $(shell $(PYLON_ROOT)/bin/pylon-config --libs)

$(NAME): $(NAME).o
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS) `pkg-config --cflags --libs opencv`

$(NAME).o: $(NAME).cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	$(RM) $(NAME).o $(NAME)

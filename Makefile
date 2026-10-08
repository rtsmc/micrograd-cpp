CXX := g++
CXXFLAGS := -I. -std=c++23 -Wall -Wextra -Wpedantic -g -O0
ASAN_FLAGS := -O1 -fsanitize=address -fno-omit-frame-pointer

SOURCES := test/test.cpp micrograd/engine.cpp micrograd/nn.cpp
HEADERS := micrograd/engine.hpp micrograd/nn.hpp

.PHONY: test asan clean

test: build/test
	./build/test

build/test: $(SOURCES) $(HEADERS) Makefile
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@

asan: build/test-asan
	./build/test-asan

build/test-asan: $(SOURCES) $(HEADERS) Makefile
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(ASAN_FLAGS) $(SOURCES) -o $@

clean:
	rm -rf ./build

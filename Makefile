CXX := g++
CXXFLAGS := -std=c++23 -Wall -Wextra -Wpedantic -g -O0
ASAN_FLAGS := -O1 -fsanitize=address -fno-omit-frame-pointer

.PHONY: clean asan

test: test.cpp engine.cpp engine.hpp Makefile
	$(CXX) $(CXXFLAGS) test.cpp engine.cpp -o $@

test-asan: test.cpp engine.cpp engine.hpp Makefile
	$(CXX) $(CXXFLAGS) $(ASAN_FLAGS) test.cpp engine.cpp -o $@

asan: test-asan
	./test-asan

clean:
	$(RM) test test-asan

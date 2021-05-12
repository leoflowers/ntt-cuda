CPP_FILES = $(wildcard src/*.cpp)
CU_FILES = $(wildcard src/*.cu)

H_FILES = $(wildcard include/*.h)
CUH_FILES = $(wildcard include/*.cuh)

OBJ_FILES = $(CPP_FILES:.cpp=.o)
COBJ_FILES = $(CU_FILES:.cu=.cu.o)

NVCC = nvcc

%.o: %.cpp $(DEPS)
	$(CXX) -g -c -o $@ $< -Iinclude

%.cu.o: %.cu $(DEPS)
	$(NVCC) -g -G -c -o $@ $< -Iinclude -dc

ntt-kernel: $(COBJ_FILES) $(OBJ_FILES)
	$(NVCC) -g -o $@ $< -Iinclude

clean:
	rm -rfv ntt ntt-kernel $(OBJ_FILES) $(COBJ_FILES)

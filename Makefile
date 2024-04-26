CPP_FILES 	= $(wildcard src/*.cpp)
CU_FILES 	= $(wildcard src/*.cu)

H_FILES 	= $(wildcard include/*.h)
CUH_FILES 	= $(wildcard include/*.cuh)

OBJ_FILES 	= $(CPP_FILES:.cpp=.o)
COBJ_FILES 	= $(CU_FILES:.cu=.cu.o)

NVCC = nvcc
NVCC_OPTS = -g -G --std=c++20

%.o: %.cpp $(DEPS)
	$(CXX) -g -c -o $@ $< -Iinclude -I/usr/local/cuda/include

%.cu.o: %.cu $(DEPS)
	$(NVCC) $(NVCC_OPTS) -c -o $@ $< -Iinclude -dc

ntt-kernel: $(OBJ_FILES) $(COBJ_FILES)
	$(NVCC) $(NVCC_OPTS) -o $@ $(OBJ_FILES) $(COBJ_FILES) -Iinclude

clean:
	rm -rfv ntt ntt-kernel $(OBJ_FILES) $(COBJ_FILES)

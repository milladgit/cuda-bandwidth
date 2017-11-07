
all: bw bw-pinned bw-acc bw-acc-pinned bandwidthtest

bw: bw.cu
	nvcc -std c++11 -O3 bw.cu -o bw

bw-pinned: bw.cu
	nvcc -DPINNED -std c++11 -O3 bw.cu -o bw-pinned



OPENACC_FLAGS=-ta=tesla -Minfo=accel
OPENACC_PINNED_FLAGS=-ta=tesla:pinned -Minfo=accel


bw-acc: bw-acc.cpp
	pgc++ $(OPENACC_FLAGS) --c++11 -O3 bw-acc.cpp -o bw-acc

bw-acc-pinned: bw-acc.cpp
	pgc++ $(OPENACC_PINNED_FLAGS) --c++11 -O3 bw-acc.cpp -o bw-acc-pinned


bandwidthtest: bandwidthtest.cu
	nvcc -w -std c++11 -O3 bandwidthtest.cu -o bandwidthtest

clean:
	rm -rf bw bw-pinned bw-acc bw-acc-pinned bandwidthtest


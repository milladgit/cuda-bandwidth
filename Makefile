
all: bw bandwidthtest

bw:
	nvcc -std c++11 -O3 bw.cu -o bw

bandwidthtest:
	nvcc -w -std c++11 -O3 bandwidthtest.cu

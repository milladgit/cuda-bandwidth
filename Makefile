
all: bw bandwidthtest

bw: bw.cu
	nvcc -std c++11 -O3 bw.cu -o bw

bandwidthtest: bandwidthtest.cu
	nvcc -w -std c++11 -O3 bandwidthtest.cu -o bandwidthtest

clean:
	rm -rf bw bandwidthtest


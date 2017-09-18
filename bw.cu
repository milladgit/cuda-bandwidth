
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(cmd) {\
	cudaError_t err = cmd; \
	if(err != cudaSuccess) fprintf(stderr, "Error in %s (%d) - Name: %s - String: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
}


int cmpfunc (const void * a, const void * b) {
	return ( *(double*)a - *(double*)b );
}

int main(int argc, char **argv) {

	int count_per_piece = atoi(argv[1]);
	int total_pieces = atoi(argv[2]);
	int iterations = atoi(argv[3]);

	int sz = sizeof(double) * count_per_piece;

	double *timings_h2d = (double*) malloc(sizeof(double) * iterations);
	double *timings_d2h = (double*) malloc(sizeof(double) * iterations);

	struct timespec t1, t2;
	double diff;

	for(int q=0;q < iterations; q++) {

		double **data_host;
		data_host = (double **) malloc(sizeof(double*) * total_pieces);

		double **data_device;
		data_device = (double **) malloc(sizeof(double*) * total_pieces);


		for(int i=0;i<total_pieces;i++) {
			data_host[i] = (double *) malloc(sz);

			for(int j=0; j < count_per_piece;j++) 
				data_host[i][j] = (i+1) * (j+1);
		}


		// Allocation on GPU
		for(int i=0;i<total_pieces;i++) {
			double *d;
			CUDA_CHECK(cudaMalloc((void**)&d, sz));
			data_device[i] = d;
		}


		if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1) != 0) {fprintf(stderr, "Error in timer.\n"); exit(1);}
		// Host to device
		for(int i=0;i<total_pieces;i++) {
			CUDA_CHECK(cudaMemcpy(data_device[i], data_host[i], sz, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2) != 0) {fprintf(stderr, "Error in timer.\n"); exit(1);}
		diff = (t2.tv_sec - t1.tv_sec) * 1E9 + (t2.tv_nsec - t1.tv_nsec);
		timings_h2d[q] = diff;



		if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1) != 0) {fprintf(stderr, "Error in timer.\n"); exit(1);}
		// Device to host
		for(int i=0;i<total_pieces;i++) {
			CUDA_CHECK(cudaMemcpy(data_host[i], data_device[i], sz, cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2) != 0) {fprintf(stderr, "Error in timer.\n"); exit(1);}
		diff = (t2.tv_sec - t1.tv_sec) * 1E9 + (t2.tv_nsec - t1.tv_nsec);
		timings_d2h[q] = diff;




		for(int i=0;i<total_pieces;i++) {
			free(data_host[i]);
			cudaFree(data_device[i]);
		}

		free(data_host);
		free(data_device);
	}

	qsort(timings_h2d, iterations, sizeof(double), cmpfunc);
	qsort(timings_d2h, iterations, sizeof(double), cmpfunc);

	printf("H2D median: %.2fms\n", timings_h2d[iterations/2] / 1E6);
	printf("D2H median: %.2fms\n", timings_d2h[iterations/2] / 1E6);

	return 0;
}
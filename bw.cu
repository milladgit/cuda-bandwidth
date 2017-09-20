
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

void measure_data_transfer(int count_per_piece, int total_pieces, int iterations, double *timings_h2d, double *timings_d2h) {
	int sz = sizeof(double) * count_per_piece;

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
		timings_h2d[q] = diff / total_pieces;



		if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1) != 0) {fprintf(stderr, "Error in timer.\n"); exit(1);}
		// Device to host
		for(int i=0;i<total_pieces;i++) {
			CUDA_CHECK(cudaMemcpy(data_host[i], data_device[i], sz, cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2) != 0) {fprintf(stderr, "Error in timer.\n"); exit(1);}
		diff = (t2.tv_sec - t1.tv_sec) * 1E9 + (t2.tv_nsec - t1.tv_nsec);
		timings_d2h[q] = diff / total_pieces;




		for(int i=0;i<total_pieces;i++) {
			free(data_host[i]);
			cudaFree(data_device[i]);
		}

		free(data_host);
		free(data_device);
	}	
}

int main(int argc, char **argv) {

	if(argc < 3) {
		fprintf(stderr, "Usage: %s <count_per_piece> <iterations>\n", argv[0]);
		return 1;
	}

	int count_per_piece = atoi(argv[1]);
	int iterations = atoi(argv[2]);
	int total_pieces;

	if(iterations < 2) {
		fprintf(stderr, "Unacceptable value for <iterations> parameter.\n");
		return 1;
	}

	double *timings_h2d = (double*) malloc(sizeof(double) * iterations);
	double *timings_d2h = (double*) malloc(sizeof(double) * iterations);

	fprintf(stderr, "Warmup...\n");
	measure_data_transfer(count_per_piece, 10, 1, &timings_h2d[0], &timings_d2h[0]);
	fprintf(stderr, "Warmup...Done\n");
	double t1 = 0, t2 = 0;
	for(int i=0;i<10;i++) { 
		t1 += timings_h2d[i];
		t2 += timings_d2h[i];
	}
	if(t2 < t1) 
		t1 = t2;

	t1 *= 1E-9;

	printf("t: %.10f\n", t1);

	int max_total_pieces = 500000;
	total_pieces = (int) ceil(1.0 / t1);
	if(total_pieces > max_total_pieces) {
		fprintf(stderr, "Number of iterations has exceeded more than %d (it was %d)\n", max_total_pieces, total_pieces);
		total_pieces = max_total_pieces;
	}
	fprintf(stderr, "Number of iterations to reach at least 1 second: %d\n", total_pieces);


	fprintf(stderr, "Main operations...\n");
	measure_data_transfer(count_per_piece, total_pieces, iterations, &timings_h2d[0], &timings_d2h[0]);
	fprintf(stderr, "Main operations...Done\n");


	qsort(timings_h2d, iterations, sizeof(double), cmpfunc);
	qsort(timings_d2h, iterations, sizeof(double), cmpfunc);

	double t_h2d, t_d2h;
	if(iterations % 2 == 1) {
		t_h2d = timings_h2d[iterations/2];
		t_d2h = timings_d2h[iterations/2];
	} else {
		t_h2d = timings_h2d[iterations/2];
		t_h2d += timings_h2d[iterations/2 - 1];
		t_h2d /= 2;

		t_d2h = timings_d2h[iterations/2];
		t_d2h += timings_d2h[iterations/2 - 1];
		t_d2h /= 2;
	}

	printf("H2D median: %.2fus\n", t_h2d / 1E3);
	printf("D2H median: %.2fus\n", t_d2h / 1E3);

	printf("------------------------\n");

	printf("H2D median: %.2fms\n", t_h2d / 1E6);
	printf("D2H median: %.2fms\n", t_d2h / 1E6);

	return 0;
}

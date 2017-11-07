
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <vector>

using namespace std;


#define CUDA_CHECK(cmd) {\
	cudaError_t err = cmd; \
	if(err != cudaSuccess) fprintf(stderr, "Error in %s (%d) - Name: %s - String: %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
}


#define GET_CLOCK(t) if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t) != 0) {fprintf(stderr, "%s (%d): Error in timer.\n", __FILE__, __LINE__); exit(1);}
#define COMPUTE_DELTA_T(diff, t2, t1) diff = (t2.tv_sec - t1.tv_sec) * 1E9 + (t2.tv_nsec - t1.tv_nsec);




bool cmpfunc (double a, double b) { return a < b; }



void measure_data_transfer(int NumberOfDoubleValuesAsVector, int NumberOfVectors, int InnerIterationCount, int OuterIterationCount, 
		vector<double> *timings_h2d, vector<double> *timings_d2h) {


	int sz = sizeof(double) * NumberOfDoubleValuesAsVector;

	struct timespec t1, t2;
	double diff;



	double **data_host;
	data_host = (double **) malloc(sizeof(double*) * NumberOfVectors);

	for(int i=0;i<NumberOfVectors;i++) {
		data_host[i] = (double *) malloc(sz);
	}

	// Allocation on GPU
	#pragma acc enter data copyin(data_host[0:NumberOfVectors][0:NumberOfDoubleValuesAsVector])
	// for(int i=0;i<NumberOfVectors;i++) {
	// 	double *d = data_host[i];
	// 	#pragma acc enter data create(d[0:NumberOfDoubleValuesAsVector])
	// }

	for(int i=0;i<NumberOfVectors; i++) {
		for(int j=0;j<NumberOfDoubleValuesAsVector; j++) {
			data_host[i][j] = i*j;
		}
	}


	for(int q=0;q < OuterIterationCount; q++) {


		GET_CLOCK(t1);
		// Host to device
		for(int i=0;i<InnerIterationCount;i++) {
			int index = i % NumberOfVectors;
			#pragma acc update device(data_host[index:1][0:NumberOfDoubleValuesAsVector]) async
			#pragma acc wait
		}
		GET_CLOCK(t2);
		COMPUTE_DELTA_T(diff, t2, t1);
		timings_h2d->push_back(diff / InnerIterationCount);

		#pragma acc parallel loop collapse(2) gang vector present(data_host[0:1][0:1])
		for(int i=0;i<NumberOfVectors; i++) {
			for(int j=0;j<NumberOfDoubleValuesAsVector; j++) {
				data_host[i][j] = i*j;
			}
		}


		GET_CLOCK(t1);
		// Device to host
		for(int i=0;i<InnerIterationCount;i++) {
			int index = i % NumberOfVectors;
			#pragma acc update host(data_host[index:1][0:NumberOfDoubleValuesAsVector]) async
			#pragma acc wait
		}
		GET_CLOCK(t2);
		COMPUTE_DELTA_T(diff, t2, t1);
		timings_d2h->push_back(diff / InnerIterationCount);


	}	



	#pragma acc exit data delete(data_host[0:NumberOfVectors][0:NumberOfDoubleValuesAsVector])
	for(int i=0;i<NumberOfVectors;i++) {
		free(data_host[i]);
	}

	free(data_host);

}

int main(int argc, char **argv) {

	if(argc < 5) {
		fprintf(stderr, "Usage: %s <NumberOfDoubleValuesAsVector> <NumberOfVectors> <InnerIterationCount> <OuterIterationCount>\n", argv[0]);
		return 1;
	}

	int NumberOfDoubleValuesAsVector = atoi(argv[1]);
	int NumberOfVectors = atoi(argv[2]);
	int InnerIterationCount = atoi(argv[3]);
	int OuterIterationCount = atoi(argv[4]);

	if(NumberOfVectors < 2) {
		fprintf(stderr, "Unacceptable value for <NumberOfVectors> parameter.\n");
		return 1;
	}

	printf("\n\n");
	printf("------------------------\n");
	printf("NumberOfDoubleValuesAsVector: %d\n", NumberOfDoubleValuesAsVector);
	printf("NumberOfVectors: %d\n", NumberOfVectors);
	printf("InnerIterationCount: %d\n", InnerIterationCount);
	printf("OuterIterationCount: %d\n", OuterIterationCount);
	printf("------------------------\n");
	printf("\n\n");

	// double *timings_h2d = (double*) malloc(sizeof(double) * NumberOfVectors);
	// double *timings_d2h = (double*) malloc(sizeof(double) * NumberOfVectors);
	vector<double> timings_d2h, timings_h2d;

	fprintf(stderr, "Warmup...\n");
	measure_data_transfer(NumberOfDoubleValuesAsVector, NumberOfVectors, 1, 1, &timings_h2d, &timings_d2h)	;
	fprintf(stderr, "Warmup...Done\n");

	timings_h2d.clear();
	timings_d2h.clear();


	fprintf(stderr, "Main operations...\n");
	measure_data_transfer(NumberOfDoubleValuesAsVector, NumberOfVectors, InnerIterationCount, OuterIterationCount, &timings_h2d, &timings_d2h)	;
	fprintf(stderr, "Main operations...Done\n");


	sort(timings_h2d.begin(), timings_h2d.end(), cmpfunc);
	sort(timings_d2h.begin(), timings_d2h.end(), cmpfunc);


	double t_h2d, t_d2h;
	if(NumberOfVectors % 2 == 1) {
		t_h2d = timings_h2d[NumberOfVectors/2];
		t_d2h = timings_d2h[NumberOfVectors/2];
	} else {
		t_h2d = timings_h2d[NumberOfVectors/2];
		t_h2d += timings_h2d[NumberOfVectors/2 - 1];
		t_h2d /= 2;

		t_d2h = timings_d2h[NumberOfVectors/2];
		t_d2h += timings_d2h[NumberOfVectors/2 - 1];
		t_d2h /= 2;
	}

	int sz = NumberOfDoubleValuesAsVector * sizeof(double);

	printf("\n\nResults:\n");
	printf("========================\n");
	printf("Vector size: %d Bytes\n", sz);
	printf("========================\n");

	printf("H2D median: %.2fus\n", t_h2d / 1E3);
	printf("D2H median: %.2fus\n", t_d2h / 1E3);

	printf("------------------------\n");

	printf("H2D median: %.2fms\n", t_h2d / 1E6);
	printf("D2H median: %.2fms\n", t_d2h / 1E6);

	printf("------------------------\n");

	printf("csv_output,%dB,%.2fKB,%.2fMB,%.2fGB,%.2fus,%.2fus\n", sz, 1.0*sz/1024, 1.0*sz/(1024*1024), 1.0*sz/(1024*1024*1024), t_h2d / 1E3, t_d2h / 1E3);

	printf("------------------------\n");

	return 0;
}

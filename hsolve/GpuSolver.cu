#include <iostream>
#include <cuda.h>
#include "GpuSolver.h"

using namespace std;
 
__global__ 

void findSumToN(int *n, int limit)
{
	int tId = threadIdx.x;
	
	for (int i=0; i<=(int)log2((double)limit); i++)
	{
		if (tId%(int)(pow(2.0,(double)(i+1))) == 0){
			if (tId+(int)pow(2.0, (double)i) >= limit) break;
			n[tId] += n[tId+(int)pow(2.0, (double)i)];
		}
		__syncthreads();
	}
}

GpuInterface::GpuInterface()
{
	y = 20;
	asize = y*sizeof(int);
	for (int i=0; i<y; i++)
		n[i] = i;
}

void GpuInterface::sayHi()
{
	cout << "Hello there\n";
}

double GpuInterface::lookupTables(double *tab, double v)
{
/*
	if (v <= xmin_) return tab[0];
	if (v >= xmax_) return tab[tabSize];

	return tab[10];
*/
}

void GpuInterface::setupTables(double *A, double *B, double ASize, double BSize, double xmin, double xmax)
{
	cudaMalloc( (void**)&A_d, ASize*sizeof(double));
	cudaMalloc( (void**)&B_d, BSize*sizeof(double));

	cudaMemcpy(A_d, A, ASize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, BSize*sizeof(double), cudaMemcpyHostToDevice);

	xmin_ = xmin;
	xmax_ = xmax;

	A_ = A;
	B_ = B;
/*
	for (int i=0; i<ASize; i++)
		cout << A[i];
	cout << endl;

	for (int i=0; i<BSize; i++)
		cout << B[i];
	cout << endl;	
*/
}


int GpuInterface::calculateSum()
{
	int *n_d;
	cudaMalloc( (void**)&n_d, asize );

        cudaMemcpy(n_d, n, asize, cudaMemcpyHostToDevice );

        dim3 dimBlock( y, 1 );
        dim3 dimGrid( 1, 1 );
        findSumToN<<<dimGrid, dimBlock>>>(n_d, y);
        cudaMemcpy(n, n_d, asize, cudaMemcpyDeviceToHost);
        cudaFree (n_d);
        return n[0];
}

void GpuInterface::setY(int newVal)
{
	y = newVal;
	asize = y*sizeof(int);
	for (int i=0; i<y; i++)
                n[i] = i;

}
/*
int main()
{
        GpuInterface obj;
        obj.setY(20);
        std::cout << obj.calculateSum();
        return EXIT_SUCCESS;
}
*/

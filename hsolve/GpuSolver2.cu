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

void GpuInterface::lookupTables(double &v, double* A, double* B) const
{
	if (v <= xmin_){
		*A = A[0];
		*B = B[0];
	}
	else if (v >= xmax_){
		*A = A[ASize_];
		*B = B[BSize_];
	}
	else{	
		unsigned int index = (v-xmin_) * invDx_;
		//assert(ASize_ > index && BSize_ > index);
		//Check for lookupByInterpolation in the HHGate code
		double frac = (v-xmin_-(index/invDx_)) * invDx_;
		*A = A_[index]*(1-frac) + A_[index+1] * frac;
		*B = B_[index]*(1-frac) + B_[index+1] * frac;
	}
}

void GpuInterface::setupTables(double *A, double *B, double ASize, double BSize, double xmin, double xmax, double invDx)
{
	cudaMalloc( (void**)&A_d, ASize*sizeof(double));
	cudaMalloc( (void**)&B_d, BSize*sizeof(double));
	cudaMalloc( (void**)&xmin_d, sizeof(double));
	cudaMalloc( (void**)&xmax_d, sizeof(double));
	cudaMalloc( (void**)&invDx_d, sizeof(double));
	cudaMalloc( (void**)&ASize_d, sizeof(int));
	cudaMalloc( (void**)&BSize_d, sizeof(int));

	cudaMemcpy(A_d, A, ASize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, BSize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(xmin_d, &xmin, BSize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(xmax_d, xmax, BSize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, BSize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, BSize*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, BSize*sizeof(double), cudaMemcpyHostToDevice);
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

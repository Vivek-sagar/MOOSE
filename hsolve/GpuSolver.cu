#include <iostream>
#include <cuda.h>
#include "GpuSolver.h"
 
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

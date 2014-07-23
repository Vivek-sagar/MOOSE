#include <cuda.h>
#include <iostream>
#include "GpuLookup.h"

__global__ void lookup_kernel(double row, double column, double *table_d, unsigned int *nColumns_d, double *result_d)
{
	double *table = table_d;
	table += (int)(row*(2* *nColumns_d) + column);
	*result_d = *table;
}


GpuLookupTable::GpuLookupTable()
{

}

GpuLookupTable::GpuLookupTable(double *min, double *max, int *nDivs, unsigned int nSpecies)
{
	min_ = *min;
	max_ = *max;
	// Number of points is 1 more than number of divisions.
	// Then add one more since we may interpolate at the last point in the table.
	nPts_ = *nDivs + 1 + 1;
	dx_= ( *max - *min ) / *nDivs;
	// Every row has 2 entries for each type of gate
	nColumns_ = 2 * nSpecies;

	cudaMalloc((void **)&min_d, sizeof(double));
	cudaMalloc((void **)&max_d, sizeof(double));
	cudaMalloc((void **)&nPts_d, sizeof(unsigned int));
	cudaMalloc((void **)&dx_d, sizeof(double));
	cudaMalloc((void **)&nColumns_d, sizeof(unsigned int));
	cudaMalloc((void **)&result_d, sizeof(double));

 	cudaMemcpy( min_d, min, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( max_d, max, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( nPts_d, &nPts_, sizeof(unsigned int), cudaMemcpyHostToDevice);
 	cudaMemcpy( dx_d, &dx_, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( nColumns_d, &nColumns_, sizeof(unsigned int), cudaMemcpyHostToDevice);

 	// Just randomly assumes that there will be only 50 species and allocates memory for that. BAD IDEA!
	cudaMalloc((void **)&table_d, (nPts_ * 100) * sizeof(double));

	cudaMemset(table_d, 0x00, (nPts_ * 100) * sizeof(double));
}

void GpuLookupTable::sayHi()
{
	std::cout << "Hi there! ";
}

void GpuLookupTable::addColumns(int species, double *C1, double *C2)
{
	//Get iTable to point to last element in the table
	double *iTable = table_d;//+(nPts_ * nColumns_);
	// Loop until last but one point
	for (int i=0; i<nPts_-1; i++ )
	{
		//std::cout << i << " " << C1[i] << " " << C2[i] << "\n";
		cudaMemcpy(iTable, &C1[i], sizeof(double), cudaMemcpyHostToDevice);
		iTable++;
		cudaMemcpy(iTable, &C2[i], sizeof(double), cudaMemcpyHostToDevice);
		iTable++;
	}

	// Then duplicate the last point
	cudaMemset(iTable, C1[nPts_-2], sizeof(double));
	iTable++;
	cudaMemset(iTable, C2[nPts_-2], sizeof(double));
	iTable++;
}

void GpuLookupTable::lookup()
{
	lookup_kernel<<<1,1>>>(20.0, 2.0, table_d, nColumns_d, result_d);
	cudaMemcpy(result_, result_d, sizeof(double), cudaMemcpyDeviceToHost);
	std::cout << "%%%%%%%%" << *result_ << "\n";
}

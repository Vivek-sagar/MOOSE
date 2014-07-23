#include <cuda.h>
#include <iostream>
#include "GpuLookup.h"


GpuLookupTable::GpuLookupTable()
{

}

GpuLookupTable::GpuLookupTable(double *min, double *max, int *nDivs, unsigned int nSpecies)
{
	// min_ = *min;
	// max_ = *max;
	// nPts_ = *nDivs + 1 + 1;
	// dx_= ( *max - *min ) / *nDivs;
	// nColumns_ = 2 * nSpecies;

	cudaMalloc((void **)&min_d, sizeof(double));
	cudaMalloc((void **)&max_d, sizeof(double));
	cudaMalloc((void **)&nPts_d, sizeof(unsigned int));
	cudaMalloc((void **)&dx_d, sizeof(double));
	cudaMalloc((void **)&nColumns_d, sizeof(unsigned int));

	// Number of points is 1 more than number of divisions.
	// Then add one more since we may interpolate at the last point in the table.
	// Every row has 2 entries for each type of gate

	unsigned int nPts_ = *nDivs + 1 + 1;
	double dx_= ( *max - *min ) / *nDivs;
	unsigned int nColumns_ = 2 * nSpecies;

 	cudaMemcpy( min_d, min, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( max_d, max, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( nPts_d, &nPts_, sizeof(unsigned int), cudaMemcpyHostToDevice);
 	cudaMemcpy( dx_d, &dx_, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( nColumns_d, &nColumns_, sizeof(unsigned int), cudaMemcpyHostToDevice);

 	// Just randomly assumes that there will be only 50 species and allocates memory for that. BAD IDEA!
	cudaMalloc((void **)&table_d, (nPts_ * 100) * sizeof(double));

}

void GpuLookupTable::sayHi()
{
	std::cout << "Hi there! ";
}

void GpuLookupTable::addColumns(int species, double *C1, double *C2)
{
	// double *iTable = table_d+nColumns_;
	// for (int i=0)

}

void GpuLookupTable::lookup(double row, double column, double& C1, double& C2)
{

}

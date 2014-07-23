#include <cuda.h>
#include <iostream>
#include "GpuLookup.h"

__global__ void lookup_kernel(double *row_array, double *column_array, double *table_d, unsigned int *nColumns_d, double *result_d)
{

	int tId = threadIdx.x;

	double row = row_array[tId];
	double column = column_array[tId];

	double *table = table_d;

	table += (int)(row*2 + column); // This formula ignores column considerations. Assumes a single type of gate with 2 columns.
	// table += (int)(row*(2* *nColumns_d) + column);
	result_d[tId] = *table;
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

	//Hardcoded. Max number of parallel lookups is 1000. BAD IDEA!
	cudaMalloc((void **)&result_d, 1000*sizeof(double));

 	cudaMemcpy( min_d, min, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( max_d, max, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( nPts_d, &nPts_, sizeof(unsigned int), cudaMemcpyHostToDevice);
 	cudaMemcpy( dx_d, &dx_, sizeof(double), cudaMemcpyHostToDevice);
 	cudaMemcpy( nColumns_d, &nColumns_, sizeof(unsigned int), cudaMemcpyHostToDevice);

 	// Just randomly assumes that there will be only 50 species and allocates memory accordingly. BAD IDEA!
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

	std::cout << "********" << nColumns_ << "\n" ;
	for (int i=0; i<nPts_-1; i++ )
	{
		//std::cout << i << " " << C1[i] << " " << C2[i] << "\n";
		cudaMemcpy(iTable, &C1[i], sizeof(double), cudaMemcpyHostToDevice);
		iTable++;
		cudaMemcpy(iTable, &C2[i], sizeof(double), cudaMemcpyHostToDevice);
		iTable++;
	}

	//std::cout << "ggggg" << C2[0] << " " << C2[40] << "\n";

	// Then duplicate the last point
	cudaMemcpy(iTable, &C1[nPts_-2], sizeof(double), cudaMemcpyHostToDevice);
	iTable++;
	cudaMemcpy(iTable, &C2[nPts_-2], sizeof(double), cudaMemcpyHostToDevice);
	iTable++;
}

void GpuLookupTable::lookup(double *row, double *column)
{
	 //std::cout << "%%%%%%%%" << row[0] << " " << column[0] << " " << row[1] << " " << column[1] << "\n";
	double *row_array_d;
	double *column_array_d;

	int set_size = 200;

	cudaMalloc((void **)&row_array_d, set_size*sizeof(double));
	cudaMalloc((void **)&column_array_d, set_size*sizeof(double));

	cudaMemcpy(row_array_d, row, set_size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(column_array_d, column, set_size*sizeof(double), cudaMemcpyHostToDevice);

	lookup_kernel<<<1,set_size>>>(row_array_d, column_array_d, table_d, nColumns_d, result_d);
	
	cudaMemcpy(result_, result_d, set_size*sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << "%%%%%%%% "; 
	for (int i=0; i<set_size; i++)
		std::cout << result_[i] << " ";
	std::cout << "\n";

}

#include <cuda.h>
#include <iostream>
#include "GpuLookup.h"

__global__ void lookup_kernel(double *row_array, double *column_array, double *table_d, unsigned int *nRows_d, double *result_d)
{

	int tId = threadIdx.x;

	double row = row_array[tId];
	double column = column_array[tId];

	double *table = table_d;

	table += (int)(row + column * (*nRows_d));

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
	nColumns_ = 0;//2 * nSpecies;

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

// void GpuLookupTable::row(double V, GpuLookupRow& row)
// {
// 	if (V < min_) V = min_;
// 	else if (V > max_) V = max_;
	
// 	double div = ( x - min_ ) / dx_;
// 	unsigned int integer = ( unsigned int )( div );

// 	row.fraction = div - integer;
// 	row.row = table_d + integer * nColumns_;
// }

void GpuLookupTable::sayHi()
{
	std::cout << "Hi there! ";
}

// Columns are arranged in memory as    |	They are visible as
// 										|	Column 1 	Column 2 	Column 3 	Column 4
// C1(Type 1)							|	C1(Type 1) 	C2(Type 1)	C1(Type 2)	C2(Type 2)
// C2(Type 1)							|
// C1(Type 2)							|
// C2(Type 2)							|
// .									|
// .									|
// .									|


void GpuLookupTable::addColumns(int species, double *C1, double *C2)
{
	//Get iTable to point to last element in the table
	double *iTable = table_d + (nPts_ * nColumns_);
	
	// Loop until last but one point
	for (int i=0; i<nPts_-1; i++ )
	{
		//std::cout << i << " " << C1[i] << " " << C2[i] << "\n";
		cudaMemcpy(iTable, &C1[i], sizeof(double), cudaMemcpyHostToDevice);
		iTable++;
		
	}
	// Then duplicate the last point
	cudaMemcpy(iTable, &C1[nPts_-2], sizeof(double), cudaMemcpyHostToDevice);
	iTable++;

	//Similarly for C2
	for (int i=0; i<nPts_-1; i++ )
	{
		cudaMemcpy(iTable, &C2[i], sizeof(double), cudaMemcpyHostToDevice);
		iTable++;
	}
	cudaMemcpy(iTable, &C2[nPts_-2], sizeof(double), cudaMemcpyHostToDevice);
	iTable++;

	nColumns_ += 2;
}

void GpuLookupTable::lookup(double *row, double *column, unsigned int set_size)
{
	double *row_array_d;
	double *column_array_d;

	cudaMalloc((void **)&row_array_d, set_size*sizeof(double));
	cudaMalloc((void **)&column_array_d, set_size*sizeof(double));

	cudaMemcpy(row_array_d, row, set_size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(column_array_d, column, set_size*sizeof(double), cudaMemcpyHostToDevice);

	lookup_kernel<<<1,set_size>>>(row_array_d, column_array_d, table_d, nPts_d, result_d);
	
	cudaMemcpy(result_, result_d, set_size*sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << "%%%%%%%% "; 
	for (int i=0; i<set_size; i++)
		std::cout << result_[i] << "\n";
	std::cout << "\n";

}

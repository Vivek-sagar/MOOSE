#ifndef GPU_LOOKUP_H
#define GPU_LOOKUP_H

class GpuLookupTable
{
	public:
		double *min_d, *max_d, *dx_d;
		unsigned int *nPts_d, *nColumns_d;
		double *table_d;

		GpuLookupTable();
		GpuLookupTable(double *min, double *max, int *nDivs, unsigned int nSpecies);
		void addColumns(int species, double *C1, double *C2);
		
		void lookup(double row, double column, double& C1, double& C2);

		void sayHi();
		
};

#endif

#ifndef EXAMPLE6_H
#define EXAMPLE6_H

class GpuInterface
{
        public:
                int n[20];
                int y;
                int asize;
		double *A_d, *B_d;
		double *A_, *B_;
		double xmin_, xmax_;

                GpuInterface();
		void sayHi();
		void setupTables(double*, double*, double, double, double, double);
		double lookupTables(double*, double);
                int calculateSum();
                void setY(int);
};
#endif

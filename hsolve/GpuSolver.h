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
		int ASize_, BSize_;
		double xmin_, xmax_;
		double invDx_;

                GpuInterface();
		void sayHi();
		void setupTables(double*, double*, double, double, double, double, double);
		void lookupTables(double&, double*, double*) const;
                int calculateSum();
                void setY(int);
};
#endif

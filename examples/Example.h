#ifndef EXAMPLE_H
#define EXAMPLE_H

#include "../hsolve/GpuSolver.h"

class Example {

    private:
        double x_;
        double y_;
        double output_;

	GpuInterface gpu;

    public:

        Example();
        
        double getX() const;
        void setX( double x );
        double getY() const;
        void setY( double y );

        void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

        void handleX(double arg);
        void handleY(double arg);
        
        vector< Id > getNeighbors( const Eref& e, string field ) const;

        static const Cinfo* initCinfo();

}; 
#endif

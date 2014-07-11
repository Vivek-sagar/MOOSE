#ifndef EXAMPLE6_H
#define EXAMPLE6_H
class GpuInterface
{
        public:
                int n[20];
                int y;
                int asize;

                GpuInterface();
                int calculateSum();
                void setY(int);
};
#endif

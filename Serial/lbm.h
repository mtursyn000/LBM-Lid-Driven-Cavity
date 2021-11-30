#ifndef _LBM_H_
#define _LBM_H_
#define q 9
#define dim 2

#include <cmath>
#include <fstream>
#include <iomanip>

class LBM
{
public:
    LBM(int sx, int sy, double &tau, double &rho0, double &u0) : lx(sx + 5), ly(sy + 5), tau0(tau), rho0(rho0), u0(u0)
    {
        // weight function
        t = (double *)malloc(sizeof(double) * q);
        e = (int *)malloc(sizeof(int) * q * dim);
        // distribution function
        f = (double *)malloc(sizeof(double) * this->lx * this->ly * q);
        fb = (double *)malloc(sizeof(double) * this->lx * this->ly * q);
        ft = (double *)malloc(sizeof(double) * this->lx * this->ly * q);
        // macroscopic function
        ua = (double *)malloc(sizeof(double) * this->lx * this->ly);
        va = (double *)malloc(sizeof(double) * this->lx * this->ly);
        p = (double *)malloc(sizeof(double) * this->lx * this->ly);
        rho = (double *)malloc(sizeof(double) * this->lx * this->ly);
        cs2 = 1.0 / 3.0;
    };
    void Initialize();
    void Collision();
    void Streaming();
    void BC();
    void Update();
    void Output(int n);

    // 1D array for distribution function
    inline int l3i(int iq, int x, int y)
    {

        return (x + y * this->lx) * q + iq;
    }

    // 1D array for macroscopic function
    inline int l2i(int x, int y)
    {

        return x + y * this->lx;
    }
    inline int l2e(int x, int y)
    {

        return x + y * q;
    }

    // domain size and relaxation time
    int lx, ly;
    double rho0;
    double mu;
    double cs2;
    double tau0;

    // distribution function
    double *f;
    double *ft;
    double *fb;
    double feq;

    // macroscopic
    double *ua;
    double *va;
    double *p;
    double *rho;

    // initial velocity
    double u0;

    // discrete velocity and weight function
    double *t;
    int *e;
};
#endif
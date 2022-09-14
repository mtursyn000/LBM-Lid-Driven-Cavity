#include "lbm.h"

void LBM::Initialize()
{

    // weight and discrete velocity
    t[0] = 4.0 / 9.0;
    t[1] = 1.0 / 9.0;
    t[2] = 1.0 / 9.0;
    t[3] = 1.0 / 9.0;
    t[4] = 1.0 / 9.0;
    t[5] = 1.0 / 36.0;
    t[6] = 1.0 / 36.0;
    t[7] = 1.0 / 36.0;
    t[8] = 1.0 / 36.0;

    e[l2e(0, 0)] = 0;
    e[l2e(1, 0)] = 1;
    e[l2e(2, 0)] = 0;
    e[l2e(3, 0)] = -1;
    e[l2e(4, 0)] = 0;
    e[l2e(5, 0)] = 1;
    e[l2e(6, 0)] = -1;
    e[l2e(7, 0)] = -1;
    e[l2e(8, 0)] = 1;

    e[l2e(0, 1)] = 0;
    e[l2e(1, 1)] = 0;
    e[l2e(2, 1)] = 1;
    e[l2e(3, 1)] = 0;
    e[l2e(4, 1)] = -1;
    e[l2e(5, 1)] = 1;
    e[l2e(6, 1)] = 1;
    e[l2e(7, 1)] = -1;
    e[l2e(8, 1)] = -1;

    // macroscopic function initialization
    for (int j = 0; j < this->ly; j++)
    {
        for (int i = 0; i < this->lx; i++)
        {
            ua[l2i(i, j)] = 0.0;
            va[l2i(i, j)] = 0.0;
            p[l2i(i, j)] = 0.0;
            rho[l2i(i, j)] = rho0;
        }
    }

    // distribution function initialization
    for (int j = 2; j < ly - 2; j++)
    {
        for (int i = 2; i < lx - 2; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {
                f[l3i(ii, i, j)] = t[ii] * rho[l2i(i, j)] +
                                   t[ii] * (3.0 * (e[l2e(ii, 0)] * ua[l2i(i, j)] + e[l2e(ii, 1)] * va[l2i(i, j)]) +
                                            4.5 * pow((e[l2e(ii, 0)] * ua[l2i(i, j)] + e[l2e(ii, 1)] * va[l2i(i, j)]), 2) -
                                            1.5 * (pow(ua[l2i(i, j)], 2) + pow(va[l2i(i, j)], 2)));
                fb[l3i(ii, i, j)] = 0.0;
                ft[l3i(ii, i, j)] = 0.0;
            }
        }
    }
};
void LBM::Collision()
{

    for (int j = 2; j < this->ly - 2; j++)
    {
        for (int i = 2; i < this->lx - 2; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {
                feq = t[ii] * rho[l2i(i, j)] + t[ii] * (3.0 * (e[l2e(ii, 0)] * ua[l2i(i, j)] + e[l2e(ii, 1)] * va[l2i(i, j)]) +
                                                        4.5 * pow((e[l2e(ii, 0)] * ua[l2i(i, j)] + e[l2e(ii, 1)] * va[l2i(i, j)]), 2) -
                                                        1.5 * (pow(ua[l2i(i, j)], 2) + pow(va[l2i(i, j)], 2)));
                f[l3i(ii, i, j)] = f[l3i(ii, i, j)] - (f[l3i(ii, i, j)] - feq) / (this->tau0 + 0.5);
            }
        }
    }
};
void LBM::Streaming()
{

    for (int j = 1; j <= this->ly - 2; j++)
    {
        for (int i = 1; i <= this->lx - 2; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {
                int im1 = i - e[l2e(ii, 0)];
                int jm1 = j - e[l2e(ii, 1)];
                ft[l3i(ii, i, j)] = f[l3i(ii, im1, jm1)];
            }
        }
    }
    for (int j = 2; j <= this->ly - 3; j++)
    {
        for (int i = 2; i <= this->lx - 3; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {
                f[l3i(ii, i, j)] = ft[l3i(ii, i, j)];
            }
        }
    }

    for (int j = 2; j <= this->ly - 3; j++)
    {
        f[l3i(1, 2, j)] = ft[l3i(3, 1, j)];
        f[l3i(5, 2, j)] = ft[l3i(7, 1, j - 1)];
        f[l3i(8, 2, j)] = ft[l3i(6, 1, j + 1)];

        f[l3i(3, lx - 3, j)] = ft[l3i(1, lx - 2, j)];
        f[l3i(7, lx - 3, j)] = ft[l3i(5, lx - 2, j + 1)];
        f[l3i(6, lx - 3, j)] = ft[l3i(8, lx - 2, j - 1)];
    }

    // up and bottom boundary condition

    for (int i = 2; i <= this->lx - 3; i++)
    {
        f[l3i(2, i, 2)] = ft[l3i(4, i, 1)];
        f[l3i(5, i, 2)] = ft[l3i(7, i - 1, 1)];
        f[l3i(6, i, 2)] = ft[l3i(8, i + 1, 1)];

        f[l3i(4, i, ly - 3)] = ft[l3i(2, i, ly - 2)];
        f[l3i(7, i, ly - 3)] = ft[l3i(5, i + 1, ly - 2)] - 6.0 * t[5] * this->u0;
        f[l3i(8, i, ly - 3)] = ft[l3i(6, i - 1, ly - 2)] + 6.0 * t[6] * this->u0;
    }
};
void LBM::BC(){

};
void LBM::Update()
{

    for (int j = 2; j < this->ly - 2; j++)
    {
        for (int i = 2; i < this->lx - 2; i++)
        {
            rho[l2i(i, j)] = 0.0;
            for (int ii = 0; ii < q; ii++)
            {
                rho[l2i(i, j)] += f[l3i(ii, i, j)];
            }
        }
    }
    for (int j = 2; j < this->ly - 2; j++)
    {
        for (int i = 2; i < this->lx - 2; i++)
        {
            ua[l2i(i, j)] = 0.0;
            va[l2i(i, j)] = 0.0;
            for (int ii = 0; ii < q; ii++)
            {
                ua[l2i(i, j)] += f[l3i(ii, i, j)] * e[l2e(ii, 0)] / this->rho0;
                va[l2i(i, j)] += f[l3i(ii, i, j)] * e[l2e(ii, 1)] / this->rho0;
            }
        }
    }
};
void LBM::Output(int n)
{

    std::ofstream outfile;
    std::string str = "output" + std::to_string(n);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,u,v,rho" << std::endl;
    outfile << "zone I=" << this->lx - 4 << ",J=" << this->ly - 4 << std::endl;

    for (int j = 2; j < this->ly - 2; j++)
    {
        for (int i = 2; i < this->lx - 2; i++)
        {

            outfile << std::setprecision(8) << setiosflags(std::ios::left) << (i - 2.0) << " " << (j - 2.0)<< " " << ua[l2i(i, j)] << " " << va[l2i(i, j)] << " " << rho[l2i(i, j)] << std::endl;
        }
    }

    outfile.close();
    printf("\n");
    printf("The result %d is writen\n", n);
    printf("\n");
    printf("============================\n");
};
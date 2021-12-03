#include "lbm.hpp"

void LBM::Initialize()
{

    // weight and discrete velocity
    t(0) = 4.0 / 9.0;
    t(1) = 1.0 / 9.0;
    t(2) = 1.0 / 9.0;
    t(3) = 1.0 / 9.0;
    t(4) = 1.0 / 9.0;
    t(5) = 1.0 / 36.0;
    t(6) = 1.0 / 36.0;
    t(7) = 1.0 / 36.0;
    t(8) = 1.0 / 36.0;

    e(0, 0) = 0;
    e(1, 0) = 1;
    e(2, 0) = 0;
    e(3, 0) = -1;
    e(4, 0) = 0;
    e(5, 0) = 1;
    e(6, 0) = -1;
    e(7, 0) = -1;
    e(8, 0) = 1;

    e(0, 1) = 0;
    e(1, 1) = 0;
    e(2, 1) = 1;
    e(3, 1) = 0;
    e(4, 1) = -1;
    e(5, 1) = 1;
    e(6, 1) = 1;
    e(7, 1) = -1;
    e(8, 1) = -1;

    Kokkos::parallel_for(
        "initv", mdrange_policy2({0, 0}, {lx, ly}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            ua(i, j) = 0;
            va(i, j) = 0;
            p(i, j) = 0;
            rho(i, j) = rho0;
        });
    // distribution function initialization

    Kokkos::parallel_for(
        "initf", mdrange_policy3({0, 0, 0}, {q, lx, ly}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f(ii, i, j) = t(ii) * p(i, j) * 3.0 +
                          t(ii) * (3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                   4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                   1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));

            ft(ii, i, j) = 0;
        });
};
void LBM::Collision()
{

    Kokkos::parallel_for(
        "collision", mdrange_policy3({0, 2, 2}, {q, lx - 2, ly - 2}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            double feq = t(ii) * p(i, j) * 3.0 +
                         t(ii) * (3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                  4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                  1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));
            f(ii, i, j) -= (f(ii, i, j) - feq) / (tau0 + 0.5);
        });
};
void LBM::Streaming()
{

    Kokkos::parallel_for(
        "stream1", mdrange_policy3({0, 1, 1}, {q, lx - 1, ly - 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            ft(ii, i, j) = f(ii, i - e(ii, 0), j - e(ii, 1));
        });

    Kokkos::parallel_for(
        "stream2", mdrange_policy3({0, 2, 2}, {q, lx - 2, ly - 2}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f(ii, i, j) = ft(ii, i, j);
        });

    Kokkos::parallel_for(
        "stream3", range_policy(2, ly - 2), KOKKOS_CLASS_LAMBDA(const int j) {
            f(1, 2, j) = ft(3, 1, j);
            f(5, 2, j) = ft(7, 1, j - 1);
            f(8, 2, j) = ft(6, 1, j + 1);

            f(3, lx - 3, j) = ft(1, lx - 2, j);
            f(7, lx - 3, j) = ft(5, lx - 2, j + 1);
            f(6, lx - 3, j) = ft(8, lx - 2, j - 1);
        });

    Kokkos::parallel_for(
        "stream4", range_policy(2, lx - 2), KOKKOS_CLASS_LAMBDA(const int i) {
            f(2, i, 2) = ft(4, i, 1);
            f(5, i, 2) = ft(7, i - 1, 1);
            f(6, i, 2) = ft(8, i + 1, 1);

            f(4, i, ly - 3) = ft(2, i, ly - 2);
            f(7, i, ly - 3) = ft(5, i + 1, ly - 2) - 6.0 * t(5) * this->u0;
            f(8, i, ly - 3) = ft(6, i - 1, ly - 2) + 6.0 * t(6) * this->u0;
        });
};

void LBM::Update()
{

    Kokkos::parallel_for(
        "initv", mdrange_policy2({0, 0}, {lx, ly}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            ua(i, j) = 0;
            va(i, j) = 0;
            p(i, j) = 0;
            rho(i, j) = rho0;
        });

    for (int j = 2; j < this->ly - 2; j++)
    {
        for (int i = 2; i < this->lx - 2; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {
                p(i, j) = p(i, j) + f(ii, i, j)/3.0;
                ua(i, j) = ua(i, j) + f(ii, i, j) * e(ii, 0);
                va(i, j) = va(i, j) + f(ii, i, j) * e(ii, 1);
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

            outfile << std::setprecision(8) << setiosflags(std::ios::left) << (i - 2.0) / (lx - 5.0) << " " << (j - 2.0) / (ly - 5.0) << " " << ua(i, j) << " " << va(i, j) << " " << p(i, j) << std::endl;
        }
    }

    outfile.close();
    printf("\n");
    printf("The result %d is writen\n", n);
    printf("\n");
    printf("============================\n");
};

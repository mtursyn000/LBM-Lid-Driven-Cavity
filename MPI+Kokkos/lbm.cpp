#include "lbm.hpp"
#include <math.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>

void LBM::Initialize()
{
    x_lo = lx * comm.px;
    x_hi = lx * (comm.px + 1);
    y_lo = ly * comm.py;
    y_hi = ly * (comm.py + 1);

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
        "collision", mdrange_policy3({0, 1, 1}, {q, lx - 1, ly - 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            double feq = t(ii) * p(i, j) * 3.0 +
                         t(ii) * (3.0 * (e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j)) +
                                  4.5 * pow(e(ii, 0) * ua(i, j) + e(ii, 1) * va(i, j), 2) -
                                  1.5 * (pow(ua(i, j), 2) + pow(va(i, j), 2)));
            f(ii, i, j) -= (f(ii, i, j) - feq) / (tau0 + 0.5);
        });
};

void LBM::setup_subdomain()
{
    if (x_lo != 0)
        m_left = buffer_t("m_left", q, ly - 2);

    if (x_hi != glx)
        m_right = buffer_t("m_right", q, ly - 2);

    if (y_lo != 0)
        m_down = buffer_t("m_down", q, lx - 2);

    if (y_hi != gly)
        m_up = buffer_t("m_up", q, lx - 2);

    if (x_lo != 0 && y_hi != gly)
        m_leftup = buffer_ut("m_leftup", q);

    if (x_hi != glx && y_hi != gly)
        m_rightup = buffer_ut("m_rightup", q);

    if (x_lo != 0 && y_lo != 0)
        m_leftdown = buffer_ut("m_leftdown", q);

    if (x_hi != glx && y_lo != 0)
        m_rightdown = buffer_ut("m_rightdown", q);

    if (x_lo != 0)
        m_leftout = buffer_t("m_leftout", q, ly - 2);

    if (x_hi != glx)
        m_rightout = buffer_t("m_rightout", q, ly - 2);

    if (y_lo != 0)
        m_downout = buffer_t("m_downout", q, lx - 2);

    if (y_hi != gly)
        m_upout = buffer_t("m_upout", q, lx - 2);

    if (x_lo != 0 && y_hi != gly)
        m_leftupout = buffer_ut("m_leftupout", q);

    if (x_hi != glx && y_hi != gly)
        m_rightupout = buffer_ut("m_rightupout", q);

    if (x_lo != 0 && y_lo != 0)
        m_leftdownout = buffer_ut("m_leftdownout", q);

    if (x_hi != glx && y_lo != 0)
        m_rightdownout = buffer_ut("m_rightdownout", q);
}
void LBM::pack()
{
    if (x_lo != 0)
        Kokkos::deep_copy(m_leftout, Kokkos::subview(f, Kokkos::ALL, 1, std::make_pair(1, ly - 1)));

    if (x_hi != glx)
        Kokkos::deep_copy(m_rightout, Kokkos::subview(f, Kokkos::ALL, lx - 2, std::make_pair(1, ly - 1)));

    if (y_lo != 0)
        Kokkos::deep_copy(m_downout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(1, lx - 1), 1));

    if (y_hi != gly)
        Kokkos::deep_copy(m_upout, Kokkos::subview(f, Kokkos::ALL, std::make_pair(1, lx - 1), ly - 2));

    if (x_lo != 0 && y_hi != gly)
        Kokkos::deep_copy(m_leftupout, Kokkos::subview(f, Kokkos::ALL, 1, ly - 2));

    if (x_hi != glx && y_hi != gly)
        Kokkos::deep_copy(m_rightupout, Kokkos::subview(f, Kokkos::ALL, lx - 2, ly - 2));

    if (x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(m_leftdownout, Kokkos::subview(f, Kokkos::ALL, 1, 1));

    if (x_hi != glx && y_lo != 0)
        Kokkos::deep_copy(m_rightdownout, Kokkos::subview(f, Kokkos::ALL, lx - 2, 1));
}

void LBM::exchange()
{
    int mar = 1;

    if (x_lo != 0)
        MPI_Send(m_leftout.data(), m_leftout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);

    if (x_hi != glx)
        MPI_Recv(m_right.data(), m_right.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 2;
    if (x_hi != glx)
        MPI_Send(m_rightout.data(), m_rightout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);

    if (x_lo != 0)
        MPI_Recv(m_left.data(), m_left.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 3;

    if (y_lo != 0)
        MPI_Send(m_downout.data(), m_downout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);

    if (y_hi != gly)
        MPI_Recv(m_up.data(), m_up.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 4;
    if (y_hi != gly)
        MPI_Send(m_upout.data(), m_upout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);

    if (y_lo != 0)
        MPI_Recv(m_down.data(), m_down.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 5;

    if (x_lo != 0 && y_hi != gly)
        MPI_Send(m_leftupout.data(), m_leftupout.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm);

    if (x_hi != glx && y_lo != 0)
        MPI_Recv(m_rightdown.data(), m_rightdown.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 6;
    if (x_hi != glx && y_hi != gly)
        MPI_Send(m_rightupout.data(), m_rightupout.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm);

    if (x_lo != 0 && y_lo != 0)
        MPI_Recv(m_leftdown.data(), m_leftdown.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 7;
    if (x_lo != 0 && y_lo != 0)
        MPI_Send(m_leftdownout.data(), m_leftdownout.size(), MPI_DOUBLE, comm.leftdown, mar, comm.comm);

    if (x_hi != glx && y_hi != gly)
        MPI_Recv(m_rightup.data(), m_rightup.size(), MPI_DOUBLE, comm.rightup, mar, comm.comm, MPI_STATUSES_IGNORE);

    mar = 8;
    if (x_hi != glx && y_lo != 0)
        MPI_Send(m_rightdownout.data(), m_rightdownout.size(), MPI_DOUBLE, comm.rightdown, mar, comm.comm);

    if (x_lo != 0 && y_hi != gly)
        MPI_Recv(m_leftup.data(), m_leftup.size(), MPI_DOUBLE, comm.leftup, mar, comm.comm, MPI_STATUSES_IGNORE);
}

void LBM::unpack()
{
    if (x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, 0, std::make_pair(1, ly - 1)), m_left);

    if (x_hi != glx)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - 1, std::make_pair(1, ly - 1)), m_right);

    if (y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(1, lx - 1), 0), m_down);

    if (y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, std::make_pair(1, lx - 1), ly - 1), m_up);

    if (x_lo != 0 && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, 1, ly - 1), m_leftup);

    if (x_hi != glx && y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - 1, ly - 1), m_rightup);

    if (x_lo != 0 && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, 1, 1), m_leftdown);

    if (x_hi != glx && y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(f, Kokkos::ALL, lx - 1, 1), m_rightdown);
}
void LBM::Streaming()
{

    Kokkos::parallel_for(
        "stream11", mdrange_policy3({0, 1, 1}, {q, lx - 1, ly - 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            ft(ii, i, j) = f(ii, i - e(ii, 0), j - e(ii, 1));
        });

    Kokkos::parallel_for(
        "stream22", mdrange_policy3({0, 1, 1}, {q, lx - 1, ly - 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
            f(ii, i, j) = ft(ii, i, j);
        });

    if (x_lo == 0)
        Kokkos::parallel_for(
            "stream1", range_policy(1, ly - 1), KOKKOS_CLASS_LAMBDA(const int j) {
                f(1, 1, j) = ft(3, 0, j);
                f(5, 1, j) = ft(7, 0, j - 1);
                f(8, 1, j) = ft(6, 0, j + 1);
            });
    if (x_hi == glx)
        Kokkos::parallel_for(
            "stream2", range_policy(1, ly - 1), KOKKOS_CLASS_LAMBDA(const int j) {
                f(3, lx - 2, j) = ft(1, lx - 1, j);
                f(7, lx - 2, j) = ft(5, lx - 1, j + 1);
                f(6, lx - 2, j) = ft(8, lx - 1, j - 1);
            });
    if (y_lo == 0)
        Kokkos::parallel_for(
            "stream3", range_policy(1, lx - 1), KOKKOS_CLASS_LAMBDA(const int i) {
                f(2, i, 1) = ft(4, i, 0);
                f(5, i, 1) = ft(7, i - 1, 0);
                f(6, i, 1) = ft(8, i + 1, 0);
            });
    if (y_hi == gly)
        Kokkos::parallel_for(
            "stream4", range_policy(1, lx - 1), KOKKOS_CLASS_LAMBDA(const int i) {
                f(4, i, ly - 2) = ft(2, i, ly - 1);
                f(7, i, ly - 2) = ft(5, i + 1, ly - 1) - 6.0 * t(5) * u0;
                f(8, i, ly - 2) = ft(6, i - 1, ly - 1) + 6.0 * t(6) * u0;
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
    Kokkos::fence();
    for (int j = 1; j <= ly - 2; j++)
    {
        for (int i = 1; i <= lx - 2; i++)
        {
            for (int ii = 0; ii < q; ii++)
            {
                p(i, j) = p(i, j) + f(ii, i, j) / 3.0;
                ua(i, j) = ua(i, j) + f(ii, i, j) * e(ii, 0);
                va(i, j) = va(i, j) + f(ii, i, j) * e(ii, 1);
            }
        }
    }
};
void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n) + std::to_string(comm.me);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,u,v,p" << std::endl;
    outfile << "zone I=" << lx - 2 << ",J=" << ly - 2 << std::endl;

    for (int j = 1; j <= ly - 2; j++)
    {
        for (int i = 1; i <= lx - 2; i++)
        {

            outfile << std::setprecision(8) << setiosflags(std::ios::left) << ((i - 1.0) + x_lo) / (glx) << " " << ((j - 1.0) + y_lo) / (gly) << " " << ua(i, j) << " " << va(i, j) << " " << p(i, j) << std::endl;
        }
    }

    outfile.close();
    if (comm.me == 0)
    {
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
    }
};

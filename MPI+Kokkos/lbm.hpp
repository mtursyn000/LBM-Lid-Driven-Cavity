#ifndef _LBM_H_
#define _LBM_H_
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cmath>
#include <fstream>
#include <iomanip>

#define q 9
#define dim 2

struct CommHelper
{

    MPI_Comm comm;
    int rx, ry;
    int me;
    int px, py;
    int up, down, left, right, leftup, leftdown, rightup, rightdown;

    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;
        int nranks;
        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        rx = std::pow(1.0 * nranks, 1.0 / 2.0);
        while (nranks % rx != 0)
            rx++;

        ry = nranks / rx;

        px = me % rx;
        py = (me / rx) % ry;
        left = px == 0 ? -1 : me - 1;
        leftup = (px == 0 || py == ry - 1) ? -1 : me - 1 + rx;
        rightup = (px == rx - 1 || py == ry - 1) ? -1 : me + 1 + rx;
        leftdown = (px == 0 || py == 0) ? -1 : me - 1 - rx;
        rightdown = (px == rx - 1 || py == 0) ? -1 : me + 1 - rx;
        right = px == rx - 1 ? -1 : me + 1;
        down = py == 0 ? -1 : me - rx;
        up = py == ry - 1 ? -1 : me + rx;

        printf("Me:%i MyNeibors: %i %i %i %i %i %i %i %i\n", me, left, right, up, down, leftup, leftdown, rightup, rightdown);
    }
    template <class ViewType>
    void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer, MPI_Request *request_send, MPI_Request *request_recv)
    {
        MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_recv);
        MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_send);
    }
};
struct LBM
{
    typedef Kokkos::RangePolicy<> range_policy;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy3;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;

    CommHelper comm;
    MPI_Request mpi_requests_recv[8];
    MPI_Request mpi_requests_send[8];
    int mpi_active_requests;

    int glx = 200;
    int gly = 200;
    int lx = glx / comm.rx;
    int ly = gly / comm.ry;

    int x, y, x_lo, x_hi, y_lo, y_hi;
    double rho0 = 1.0;
    double mu;
    double cs2;
    double tau0 = 0.12;
    double u0 = 0.1;

    using buffer_t = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace>;
    buffer_t m_left, m_right, m_down, m_up;
    buffer_t m_leftout, m_rightout, m_downout, m_upout;
    using buffer_ut = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace>;
    buffer_ut m_leftup, m_rightup, m_leftdown, m_rightdown;
    buffer_ut m_leftupout, m_rightupout, m_leftdownout, m_rightdownout;
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> f = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("f", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> ft = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("ft", q, lx, ly);
    Kokkos::View<double ***, Kokkos::CudaUVMSpace> fb = Kokkos::View<double ***, Kokkos::CudaUVMSpace>("fb", q, lx, ly);

    Kokkos::View<double **, Kokkos::CudaUVMSpace> ua = Kokkos::View<double **, Kokkos::CudaUVMSpace>("u", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> va = Kokkos::View<double **, Kokkos::CudaUVMSpace>("v", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> rho = Kokkos::View<double **, Kokkos::CudaUVMSpace>("rho", lx, ly);
    Kokkos::View<double **, Kokkos::CudaUVMSpace> p = Kokkos::View<double **, Kokkos::CudaUVMSpace>("p", lx, ly);

    Kokkos::View<int **, Kokkos::CudaUVMSpace> e = Kokkos::View<int **, Kokkos::CudaUVMSpace>("e", q, dim);
    Kokkos::View<double *, Kokkos::CudaUVMSpace> t = Kokkos::View<double *, Kokkos::CudaUVMSpace>("t", q);

    LBM(MPI_Comm comm_) : comm(comm_)
    {
        mpi_active_requests = 0;
    };

    void Initialize();
    void Collision();
    void setup_subdomain();
    void pack();
    void exchange();
    void unpack();
    void Streaming();
    //  void BC();
    void Update();
    void Output(int n);
};
#endif
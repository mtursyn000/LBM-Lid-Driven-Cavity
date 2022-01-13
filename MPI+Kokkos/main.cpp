#include "mpi.h"
#include "lbm.hpp"
#include "System.hpp"
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[])
{
    int nx = 200;
    int ny = 200;
    // int rank;
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {

        // MPI_Comm_rank(comm, &rank);
        // if (rank == 0)
        //{
        System s1(nx, ny);
        s1.Initialize();
        s1.Monitor();
        //}
        LBM l1(MPI_COMM_WORLD);

        l1.Initialize();
        l1.setup_subdomain();

        for (int it = 1; it <= s1.Time; it++)
        {

            l1.Collision();
            l1.pack();
            l1.exchange();
            l1.unpack();
            l1.Streaming();

            l1.Update();
            if (it % s1.inter == 0)
            {
                l1.Output(it / s1.inter);
            }
        }
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
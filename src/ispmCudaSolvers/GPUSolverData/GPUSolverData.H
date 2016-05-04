/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | foam-extend: Open Source CFD
   \\    /   O peration     |
    \\  /    A nd           | For copyright notice see file Copyright
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of foam-extend.

    foam-extend is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    foam-extend is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with foam-extend.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::GPUSolverData

Description
    Container of data that is reused by the GPU solver across multiple time
    steps as long as mesh topology is unchanging

Author
    Alexander Monakov, ISP RAS

SourceFiles
    GPUSolverData.C

\*---------------------------------------------------------------------------*/

#ifndef GPUSolverData_H
#define GPUSolverData_H

#include <cassert>
#include <lduMatrix.H>

#include "AsyncPrecondProvider.H"

#include "fastainv/droptol-tuning.h"
#include "lduMatrixConversion.H"
#include "plan/plan.h"
#include "util/cuda/pinner.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class GPUSolverData Declaration
\*---------------------------------------------------------------------------*/

class GPUSolverData
{
public:
        typedef ::vector<scalar, device_memory_space_tag> dvec;

private:
    // Private data

        static const int n_gpu_scalars = 4;
        std::vector<scalar, cuda_pinned_allocator<scalar> > gpu_scalars;

        static const int n_gpu_scalars_dev = 2;
        dvec gpu_scalars_dev;

        static const int n_gpu_vectors = 11;
        dvec gpu_vectors[n_gpu_vectors];

        scalarField pinned1, pinned2;
        cuda_pinned_region<scalar> elms, gpuptr_pinned1, gpuptr_pinned2;

        csr_matrix<scalar> csrA;
        spmv_plan<scalar> amul_plan;

        std::vector<unsigned> csr_order, gpu_order;

        labelList coupledCells;
        ::vector<label, device_memory_space_tag> gpuCoupledCells;
        AsyncPrecondProvider precond;
        scalar dropTolerance;
        DroptolTuner droptune;

        static labelList recordCoupledCells(const lduMesh&);

        GPUSolverData(const lduMesh&);

public:

        static GPUSolverData* New(const lduMesh&);

    // Member functions

        void updateEarly(const lduMatrix &);

        void updatePrecond(const lduMatrix &, const dictionary &);

        void suspendPrecond();

        bool retestDroptol(const lduMatrix &);

        void notePerformance(double time, int iterations, int maxIter);

        void applyPrecond(dvec &x, dvec &Mx, dvec &tmp);

        void Amul(dvec& x, dvec& Ax, dvec& tmp, cudaEventScoped& computed_coupled,
                  const lduMatrix& matrix_,
                  const FieldField<Field, scalar>& coupleBouCoeffs_,
                  const lduInterfaceFieldPtrsList& interfaces_,
                  const direction cmpt);

        scalar& gpu_scalar_pinned(int idx)
        {
            assert(idx < n_gpu_scalars);
            return gpu_scalars.data()[idx];
        }
        devmem<scalar>& gpu_scalar_devmem(int idx)
        {
            assert(idx < n_gpu_scalars_dev);
            return gpu_scalars_dev.data()[idx];
        }
        dvec& gpu_vector(int idx)
        {
            assert(idx < n_gpu_vectors);
            return gpu_vectors[idx];
        }
};

}

#endif

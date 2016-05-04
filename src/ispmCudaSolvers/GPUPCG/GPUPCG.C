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

Description
    GPU (CUDA) Preconditioned Conjugate Gradient solver

Author
    Alexander Monakov, ISP RAS

\*---------------------------------------------------------------------------*/

#include "GPUPCG.H"

#include "util/cuda/timer.h"
#include "util/cuda/sblas.h"

namespace Foam
{

defineTypeNameAndDebug(GPUPCG, 0);

lduMatrix::solver::addsymMatrixConstructorToTable<GPUPCG>
  addGPUPCGSymMatrixConstructorToTable_;
}

Foam::GPUPCG::GPUPCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    persistent(solverPersistentData<lduMesh, GPUSolverData>::New(matrix.mesh()))
{}

solverPerformance Foam::GPUPCG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    addProfile(GPUPCG);
    GPUSolverData &p = *persistent(fieldName());
    {
        addProfile(GPUPCG_upload1);
        p.updateEarly(matrix_);
    }
    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        typeName,
        fieldName()
    );

    label nCells = psi.size();

    scalar normFactor;

    scalarField wA(nCells);

    {
    addProfile(normFactor);

    scalarField pA(nCells);

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, coupleBouCoeffs_, interfaces_, cmpt);

    // --- Calculate normalisation factor
    normFactor = this->normFactor(psi, source, wA, pA, cmpt);

    if (lduMatrix::debug >= 2)
    {
        Info<< "   Normalisation factor = " << normFactor << endl;
    }
    }

    // --- Calculate initial residual field
    scalarField rA(source - wA);

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() = gSumMag(rA)/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if (solverConverged(solverPerf))
        return solverPerf;

    p.updatePrecond(matrix_, dict());
    do
    {
        solverPerf.nIterations() = 0;
        solverPerf.finalResidual() = solverPerf.initialResidual();

        addProfile(GPUPCG_gpu);

        int i = 0;
        pinmemptr<scalar> gpu_gamma    = &p.gpu_scalar_pinned(i++);
        pinmemptr<scalar> gpu_gammaold = &p.gpu_scalar_pinned(i++);
        pinmemptr<scalar> gpu_delta    = &p.gpu_scalar_pinned(i++);
        pinmemptr<scalar> gpu_resnorm  = &p.gpu_scalar_pinned(i++);

        i = 0;
        devmem<scalar> *gpu_alpha      = &p.gpu_scalar_devmem(i++);
        devmem<scalar> *gpu_beta       = &p.gpu_scalar_devmem(i++);

        i = 0;
        GPUSolverData::dvec   &gpu_psi  = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_res  = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_tmp  = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_p    = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_q    = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_s    = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_z    = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_u    = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_w    = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_n    = p.gpu_vector(i++);
        GPUSolverData::dvec   &gpu_m    = p.gpu_vector(i++);

        gpu_psi.assign(psi.begin(), psi.end());
        gpu_res.assign(rA.begin(), rA.end());

        *gpu_gammaold = __builtin_inf();
        copy(gpu_alpha, &*gpu_gammaold, 1);

        scalar dummy = 0;
        reduce(dummy, sumOp<scalar>());

        cudaEventScoped computed_residual, computed_coupled;
        cudatimer cutimer; cutimer.start();

        // The following implements pipelined preconditioned conjugate gradient
        // method according to:
        // P. Ghysels, W. Vanroose: Hiding global synchronization latency in
        // the preconditioned Conjugate Gradient algorithm.
        //
        // The technique is useful for GPUs since it allows to reduce the
        // number of kernel launches and hide CPU-GPU latency
        p.applyPrecond(gpu_res, gpu_u, gpu_tmp);
        p.Amul(gpu_u, gpu_w, gpu_tmp, computed_coupled, matrix_,
               coupleBouCoeffs_, interfaces_, cmpt);
        sblas::dot(gpu_res.data(), gpu_u.data(), gpu_gamma(), gpu_u.size());
        sblas::dot(gpu_w.data(), gpu_u.data(), gpu_delta(), gpu_u.size());
        for (;;)
        {
            p.applyPrecond(gpu_w, gpu_m, gpu_tmp);
            p.Amul(gpu_m, gpu_n, gpu_tmp, computed_coupled, matrix_,
                   coupleBouCoeffs_, interfaces_, cmpt);

            if (solverPerf.nIterations() > 0)
            {
                computed_residual.await();
                reduce(*gpu_resnorm, sumOp<scalar>());
                solverPerf.finalResidual() = *gpu_resnorm / normFactor;
                if (solverConverged(solverPerf))
                {
                    break;
                }
            }

            if (Pstream::parRun())
            {
                reduce(*gpu_gamma, sumOp<scalar>());
                reduce(*gpu_delta, sumOp<scalar>());
            }

            sblas::ppcg_update_scalars(gpu_alpha, gpu_beta,
                                       gpu_gamma(), gpu_gammaold(), gpu_delta());
            std::swap(gpu_gammaold, gpu_gamma);
            sblas::ppcg_update_vectors
                (gpu_resnorm(), gpu_gamma(), gpu_delta(), gpu_alpha, gpu_beta,
                 gpu_n.data(), gpu_m.data(), gpu_p.data(), gpu_s.data(),
                 gpu_q.data(), gpu_z.data(), gpu_psi.data(), gpu_res.data(),
                 gpu_u.data(), gpu_w.data(), gpu_n.size());
            computed_residual.record();

            solverPerf.nIterations()++;

        }

        cutimer.stop();
        double time = cutimer.elapsed_seconds();
        p.notePerformance(time, solverPerf.nIterations(), maxIter());

        copy(psi.data(), gpu_psi.data(), nCells);

    } while (p.retestDroptol(matrix_));
    p.suspendPrecond();

    if (lduMatrix::debug >= 2)
    {
        matrix_.residual(rA, psi, source, coupleBouCoeffs_, interfaces_, cmpt);
        Info << "    CPU final residual: " << gSumMag(rA) / normFactor << endl;
    }

    return solverPerf;
}

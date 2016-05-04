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
    Container of data that is reused by the GPU solver across multiple time
    steps as long as mesh topology is unchanging

Author
    Alexander Monakov, ISP RAS

\*---------------------------------------------------------------------------*/

#include "GPUSolverData.H"
#include "util/cuda/sblas.h"
#include "util/cuda/initialize.h"
#include "FoamCompat.H"

namespace Foam
{

labelList GPUSolverData::recordCoupledCells(const lduMesh& mesh)
{
    labelList allBoundaryCells;
    forAll(mesh.interfaces(), patchI)
    {
        if (!mesh.interfaces().set(patchI))
            continue;
        allBoundaryCells.append(mesh.lduAddr().patchAddr(patchI));
    }
    sort(allBoundaryCells);
    label last = -1, n = 0;
    labelList compactedBoundaryCells(allBoundaryCells.size());
    forAll(allBoundaryCells, i)
    {
        label cell = allBoundaryCells[i];
        if (last != cell)
        {
            last = cell;
            compactedBoundaryCells[n++] = cell;
        }
    }
    compactedBoundaryCells.resize(n);
    return compactedBoundaryCells;
}

GPUSolverData::GPUSolverData(const lduMesh& mesh)
{
    addProfile(GPUPCGAux_ctor);
    ispm_initialize();
    Pout << "Using GPU at " << get_device_string().str << endl;

    csrA = csr_from_ldu(mesh.lduAddr());
    csr_order = build_order(csrA.elms.size(), csrA.elms);

    new(&amul_plan) spmv_plan<scalar>(csrA, PLAN_EXHAUSTIVE);
    gpu_order = build_order(csrA.elms.size(), amul_plan.host_matrix->elms);

    new(&elms) cuda_pinned_region<scalar>(amul_plan.host_matrix->elms.data(),
                                          amul_plan.host_matrix->elms.size());

    int nCells = csrA.n_rows;
    for (int i = 0; i < n_gpu_vectors; i++)
        gpu_vectors[i].resize(nCells);

    gpu_scalars.resize(n_gpu_scalars);
    gpu_scalars_dev.resize(n_gpu_scalars_dev);

    pinned1.resize(nCells, 0);
    pinned2.resize(nCells, 0);
    new(&gpuptr_pinned1) cuda_pinned_region<scalar>(pinned1.data(), nCells);
    new(&gpuptr_pinned2) cuda_pinned_region<scalar>(pinned2.data(), nCells);

    coupledCells = recordCoupledCells(mesh);
    gpuCoupledCells.assign(coupledCells.begin(), coupledCells.end());
}

GPUSolverData* GPUSolverData::New(const lduMesh& mesh)
{
    return new GPUSolverData(mesh);
}

void GPUSolverData::updateEarly(const lduMatrix &lduA)
{
    update_from_ldu(lduA, gpu_order, amul_plan.host_matrix->elms);
    amul_plan.device_matrix->elms.assign_async
        (&amul_plan.host_matrix->elms.front(),
         &amul_plan.host_matrix->elms.back());
}

void GPUSolverData::updatePrecond(const lduMatrix &lduA,
                                  const dictionary& dict)
{
    addProfile(updatePrecond);
    if (!precond.plan(0))
    {
        if (!precond.busy())
        {
            dropTolerance = dict.lookupOrDefault<scalar>("AINVdropTolerance",
                                                         -1);
            scalar initial_droptol = (dropTolerance > 0
                                      ? dropTolerance : droptune.getDroptol());
            precond.go(lduA, csrA, csr_order, initial_droptol);
        }
        precond.update();
    }
    else if (precond.busy())
    {
        precond.resume();
    }
    else
    {
        precond.update();
        precond.go(lduA, csrA, csr_order, dropTolerance);
    }
    droptune.resetPhase();
}

void GPUSolverData::suspendPrecond()
{
    precond.suspend();
}

bool GPUSolverData::retestDroptol(const lduMatrix &lduA)
{
    addProfile(retestDroptol);
    if (dropTolerance > 0)
        return false;
    // FIXME make configurable
    if (droptune.iterations() == 7)
    {
        dropTolerance = droptune.getBestDroptol();
        return false;
    }
    precond.clear();
    precond.go(lduA, csrA, csr_order, droptune.getDroptol());
    precond.update();
    return true;
}

void GPUSolverData::notePerformance(double time, int iterations, int maxIter)
{
    long bytes = sizeof(Foam::scalar) * csrA.n_rows * 18;
    bytes += amul_plan.device_matrix->spmv_bytes();
    bytes += precond.plan(0)->device_matrix->spmv_bytes(false, sizeof(scalar));
    bytes += precond.plan(1)->device_matrix->spmv_bytes(false, sizeof(scalar));
    bytes *= iterations;
    Foam::Info << "GPU: " << iterations << " iters: " << time << " s: ";
    Foam::Info << bytes * 1e-9 / time << " GB/s" << Foam::endl;

    if (iterations == maxIter)
        time = __builtin_inf();
    droptune.noteTime(time);
}

void GPUSolverData::applyPrecond(dvec &x, dvec &Mx, dvec &tmp)
{
    precond.plan(0)->execute_spmv(x, tmp);
    precond.plan(1)->execute_spmv(tmp, Mx);
}

void GPUSolverData::Amul(dvec& x, dvec& Ax, dvec& tmp,
                         cudaEventScoped& computed_coupled,
                         const lduMatrix& matrix_,
                         const FieldField<Field, scalar>& coupleBouCoeffs_,
                         const lduInterfaceFieldPtrsList& interfaces_,
                         const direction cmpt)
{
    if (gpuCoupledCells.size())
    {
        sblas::copy_indexed(tmp.data(), x.data(),
                            gpuCoupledCells.data(), gpuCoupledCells.size());
        copy_async(pinned2.data(), tmp.data(), gpuCoupledCells.size());
        computed_coupled.record();
    }
    amul_plan.execute_spmv(x, Ax);
    if (gpuCoupledCells.size())
    {
        computed_coupled.await();
        for (int i = 0; i < coupledCells.size(); i++)
            pinned1[coupledCells[i]] = pinned2[i];
        matrix_.initMatrixInterfaces(coupleBouCoeffs_, interfaces_,
                                     pinned1, pinned2, cmpt);
        for (int i = 0; i < coupledCells.size(); i++)
            pinned2[coupledCells[i]] = 0;
        matrix_.updateMatrixInterfaces(coupleBouCoeffs_, interfaces_,
                                       pinned1, pinned2, cmpt);
        for (int i = 0; i < coupledCells.size(); i++)
            pinned1[i] = pinned2[coupledCells[i]];
        copy_async(tmp.data(), pinned1.data(), gpuCoupledCells.size());
        sblas::add_indexed(Ax.data(), tmp.data(),
                           gpuCoupledCells.data(), gpuCoupledCells.size());
    }
}

}

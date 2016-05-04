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
    Macros that unify API differences between OpenFOAM and foam-extend to
    allow compiling the GPU solver with either of two

Author
    Alexander Monakov, ISP RAS

\*---------------------------------------------------------------------------*/

#ifndef FoamCompat_H
#define FoamCompat_H

#ifdef FOAM_DEV

// foam-extend 3.0 or newer

#include <profiling.H>

#define solverConverged(solverPerf) stop(solverPerf)

#define solverPerformance Foam::lduMatrix::solverPerformance

#else

// OpenFOAM

#define addProfile(ignored)

#define coupleBouCoeffs_ interfaceBouCoeffs_
#define normFactor(psi, source, wA, pA, cmpt) normFactor(psi, source, wA, pA)
#define maxIter() maxIter_
#define dict() controlDict_

// FIXME: check number of iterations
#define solverConverged(solverPerf) solverPerf.checkConvergence(tolerance_, relTol_)

#define solverPerformance Foam::solverPerformance

#endif

#endif

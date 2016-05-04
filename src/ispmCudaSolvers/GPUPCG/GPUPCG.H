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
    Foam::GPUPCG

Description
    GPU (CUDA) Preconditioned Conjugate Gradient solver

Author
    Alexander Monakov, ISP RAS

SourceFiles
    GPUPCG.C

\*---------------------------------------------------------------------------*/

#ifndef GPUPCG_H
#define GPUPCG_H

#include <lduMatrix.H>

#include "solverPersistentData.H"

#include "FoamCompat.H"

#include "GPUSolverData.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class GPUPCG Declaration
\*---------------------------------------------------------------------------*/

class GPUPCG
:
    public lduMatrix::solver
{
    // Private data

        const solverPersistentData<lduMesh, GPUSolverData>& persistent;

    // Private Member Functions

        //- Disallow default bitwise copy construct
        GPUPCG(const GPUPCG&);

        //- Disallow default bitwise assignment
        void operator=(const GPUPCG&);


public:

    //- Runtime type information
    TypeName("GPUPCG");


    // Constructors

        //- Construct from matrix components and solver controls
        GPUPCG
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~GPUPCG()
    {}


    // Member Functions

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt=0
        ) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

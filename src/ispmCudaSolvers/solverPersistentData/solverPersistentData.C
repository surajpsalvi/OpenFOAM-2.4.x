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
    Reference to mesh-specific data reused by a solver, invalidated upon
    mesh topology change

Author
    Alexander Monakov, ISP RAS

\*---------------------------------------------------------------------------*/

#ifndef solverPersistentDataC
#define solverPersistentDataC

#include "solverPersistentData.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //


// * * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * //

template<class MeshType, class EntryType>
Foam::solverPersistentData<MeshType, EntryType>::solverPersistentData
    (const MeshType& mesh)
:
    BaseMeshObject(mesh),
    dataPtrs_(1)
{}


// * * * * * * * * * * * * * * * * Destructor * * * * * * * * * * * * * * * //

template<class MeshType, class EntryType>
Foam::solverPersistentData<MeshType, EntryType>::~solverPersistentData()
{
    dataPtrs_.clear();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class MeshType, class EntryType>
EntryType* Foam::solverPersistentData<MeshType, EntryType>::operator()
    (const word& fieldName) const
{
    EntryType*& entry = dataPtrs_(fieldName);

    if (entry == NULL)
    {
        entry = EntryType::New(this->mesh());
    }

    return entry;
}

// ************************************************************************* //

#endif

// ************************************************************************* //

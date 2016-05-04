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
    Foam::solverPersistentData

Description
    Reference to mesh-specific data reused by a solver, invalidated upon
    mesh topology change

Author
    Alexander Monakov, ISP RAS

SourceFiles
    solverPersistentData.C

\*---------------------------------------------------------------------------*/

#ifndef solverPersistentData_H
#define solverPersistentData_H

#include "HashPtrTable.H"
#include "MeshObject.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class mapPolyMesh;

/*---------------------------------------------------------------------------*\
                     Class solverPersistentData Declaration
\*---------------------------------------------------------------------------*/

template<class MeshType, class EntryType>
class CompatMeshObject {
public:
#ifdef FOAM_DEV
    typedef MeshObject<MeshType, EntryType> Type;
#else
    typedef MeshObject<MeshType, UpdateableMeshObject, EntryType> Type;
#endif
};

TemplateName(solverPersistentData);

template<class MeshType, class EntryType>
class solverPersistentData
:
    public solverPersistentDataName,
    public CompatMeshObject
        <MeshType, solverPersistentData<MeshType, EntryType> >::Type
{
    typedef typename CompatMeshObject
        <MeshType, solverPersistentData<MeshType, EntryType> >::Type
        BaseMeshObject;
    // Private data

        //- Pointer to persistent data
        mutable HashPtrTable<EntryType> dataPtrs_;


    // Private member functions


public:

	using solverPersistentDataName::typeName;

    // Constructors

        //- Construct given a mesh
        explicit solverPersistentData(const MeshType&);


    // Destructor

        virtual ~solverPersistentData();


    // Member functions

        //- Return reference to persistent data for a given field
        EntryType* operator()(const word&) const;


#ifdef FOAM_DEV
        //- Update after mesh motion:
        //  Do nothing
        virtual bool movePoints() const
        {
            return true;
        }

        //- Update after topo change:
        //  Invalidate (delete) persistent data
        virtual bool updateMesh(const mapPolyMesh&) const
        {
            dataPtrs_.clear();
            return true;
        }
#else
        virtual bool movePoints()
        {
            return true;
        }
        virtual void updateMesh(const mapPolyMesh&)
        {
            dataPtrs_.clear();
        }
#endif
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
#   include "solverPersistentData.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

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
    Conversion from LDU matrix format to CSR

Author
    Alexander Monakov, ISP RAS

SourceFiles
    lduMatrixConversion.C

\*---------------------------------------------------------------------------*/

#ifndef lduMatrixConversion_H
#define lduMatrixConversion_H

#include <vector>

#include <formats/formats.h>

#include <lduMatrix.H>

namespace Foam
{

struct csr_matrix<scalar>
csr_from_ldu(const lduAddressing &lduAddr);

void
update_from_ldu(const lduMatrix &m,
    const std::vector<unsigned> &order,
    std::vector<scalar> &elts);

std::vector<unsigned>
build_order(int size, const std::vector<scalar> &elts);

}

#endif

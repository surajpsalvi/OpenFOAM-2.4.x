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

\*---------------------------------------------------------------------------*/

#include <algorithm>

#include "lduMatrixConversion.H"

namespace Foam
{

struct csr_matrix<scalar>
csr_from_ldu(const lduAddressing &lduAddr)
{
    unsigned counter = 1;

    csr_matrix<scalar> r;

    r.n_rows = lduAddr.size();
    r.n_cols = lduAddr.size();

    std::vector<std::vector<std::pair<unsigned, unsigned> > > rows(r.n_rows);

    for (int i = 0; i < r.n_rows; i++)
        rows[i].push_back(std::make_pair(i, counter++));

    for (int i = 0; i < lduAddr.upperAddr().size(); i++)
    {
        unsigned row = lduAddr.upperAddr()[i];
        unsigned col = lduAddr.lowerAddr()[i];

        rows[row].push_back(std::make_pair(col, counter++));
        rows[col].push_back(std::make_pair(row, counter++));
    }

    for (unsigned i = 0; i < rows.size(); i++)
    {
        std::sort(rows[i].begin(), rows[i].end());
        r.row_ptr.push_back(r.elms.size());
        for (unsigned j = 0; j < rows[i].size(); j++)
        {
            r.cols.push_back(rows[i][j].first);
            r.elms.push_back(rows[i][j].second);
        }
    }
    r.row_ptr.push_back(r.elms.size());

    r.n_nz = r.elms.size();

    return r;
}

void
update_from_ldu(const lduMatrix &m,
                const std::vector<unsigned> &order,
                std::vector<scalar> &elts)
{
    unsigned counter = 0;
    for (int i = 0; i < m.diag().size(); i++)
        elts[order[counter++]] = m.diag()[i];

    for (int i = 0; i < m.lower().size(); i++)
    {
        elts[order[counter++]] = m.lower()[i];
        elts[order[counter++]] = m.upper()[i];
    }
}

std::vector<unsigned>
build_order(int size, const std::vector<scalar> &elts)
{
    std::vector<unsigned> order(size);
    for (unsigned i = 0; i < elts.size(); i++)
    {
        unsigned counter = elts[i];
        if (counter)
            order[counter - 1] = i;
    }
    return order;
}

}

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
    Foam::AsyncPrecondProvider

Description
    Mechanism to compute preconditioners for the GPU solver asynchronously,
    doing work in a separate thread on the CPU

Author
    Alexander Monakov, ISP RAS

SourceFiles
    AsyncPrecondProvider.C

\*---------------------------------------------------------------------------*/

#ifndef AsyncPrecondProvider_H
#define AsyncPrecondProvider_H

#include <pthread.h>
#include <signal.h>
#include <semaphore.h>

#include "plan/plan.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class AsyncPrecondProvider Declaration
\*---------------------------------------------------------------------------*/

class AsyncPrecondProvider
{
    // Private data

        typedef float PrecType;

        typedef spmv_plan<scalar, PrecType> spmvType;
        spmvType *plan_[2];
        csr_matrix<PrecType> csrAinv1, csrAinv2;

        struct
        {
            int maxSize;
            double dropTolerance;
        } AINVParams;

        struct
        {
            const lduMatrix *lduA;
            csr_matrix<scalar> *csrA;
            std::vector<unsigned> *csr_order;

            pthread_t thread;
            sem_t lduAlock;
            bool busy, suspend;
        } worker;

    // Private member functions

        static void *workerJob(void *v);

public:

    // Constructors

        AsyncPrecondProvider()
        {
            plan_[0] = NULL;
            plan_[1] = NULL;
            worker.suspend = 0;
            worker.busy = 0;
            sem_init(&worker.lduAlock, 0, 1);
        }

    // Destructor

        ~AsyncPrecondProvider()
        {
            sem_destroy(&worker.lduAlock);
        }

    // Member functions

        spmvType *plan(int idx)
        {
            return plan_[idx];
        }

        bool busy()
        {
            return worker.busy;
        }

        void waitLduAlock()
        {
            sem_wait(&worker.lduAlock);
            sem_post(&worker.lduAlock);
        }

        void go(const lduMatrix &lduA, csr_matrix<scalar> &csrA,
                std::vector<unsigned> &csr_order, double dropTolerance)
        {
            sem_wait(&worker.lduAlock);
            worker.busy = true;
            worker.lduA = &lduA;
            worker.csrA = &csrA;
            worker.csr_order = &csr_order;
            worker.suspend = false;

            // FIXME make configurable
            AINVParams.maxSize = 1 << 26;
            AINVParams.dropTolerance = dropTolerance;

            pthread_create(&worker.thread, NULL, workerJob, this);

            Info << "Computing AINV with droptol " << dropTolerance << endl;
        }

        void update()
        {
            pthread_join(worker.thread, NULL);
            if (!plan_[0])
            {
                plan_[0] = new spmvType(csrAinv1, PLAN_EXHAUSTIVE);
                plan_[1] = new spmvType(csrAinv2, PLAN_EXHAUSTIVE);
            }
            else
            {
                plan_[0]->upload();
                plan_[1]->upload();
            }
        }

        void clear()
        {
            if (!plan_[0])
                return;
            delete plan_[0];
            delete plan_[1];
            plan_[0] = plan_[1] = NULL;
        }

        void suspend()
        {
            waitLduAlock();
            worker.suspend = true;
        }

        void resume()
        {
            worker.suspend = false;
            pthread_kill(worker.thread, SIGRTMIN);
        }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

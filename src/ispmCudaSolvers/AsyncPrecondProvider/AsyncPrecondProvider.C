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
    Mechanism to compute preconditioners for the GPU solver asynchronously,
    doing work in a separate thread on the CPU

Author
    Alexander Monakov, ISP RAS

\*---------------------------------------------------------------------------*/

#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>

#include "fastainv/fastainv.h"
#include "plan/plan.h"

#include "lduMatrixConversion.H"

#include "AsyncPrecondProvider.H"

namespace Foam
{

static void workerCallback(void *v)
{
    bool suspend = *static_cast<bool *>(v);
    if (suspend)
    {
	int sig;
	sigset_t sigset;
	sigemptyset(&sigset);
	sigaddset(&sigset, SIGRTMIN);
	sigwait(&sigset, &sig);
    }
}

void* AsyncPrecondProvider::workerJob(void *v)
{
    sigset_t sigset;
    sigfillset(&sigset);
    pthread_sigmask(SIG_SETMASK, &sigset, NULL);
    setpriority(PRIO_PROCESS, syscall(SYS_gettid), 19);

    AsyncPrecondProvider *this_ = static_cast<AsyncPrecondProvider *>(v);

    csr_matrix<Foam::scalar> &csrA = *this_->worker.csrA;
    update_from_ldu(*this_->worker.lduA, *this_->worker.csr_order, csrA.elms);
    sem_post(&this_->worker.lduAlock);

    fastainv_sym(this_->csrAinv1, this_->csrAinv2, csrA,
		 workerCallback, &this_->worker.suspend,
		 this_->AINVParams.maxSize,
		 this_->AINVParams.dropTolerance);

    if (this_->plan_[0])
    {
	this_->plan_[0]->change_host_matrix(this_->csrAinv1);
	this_->plan_[1]->change_host_matrix(this_->csrAinv2);
    }
    this_->worker.busy = false;
    return NULL;
}

}

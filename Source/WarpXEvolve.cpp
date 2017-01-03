
#include <cmath>
#include <limits>

#include <WarpX.H>
#include <WarpXConst.H>
#include <WarpX_f.H>

using namespace amrex;

void
WarpX::Evolve (int numsteps)
{
    BL_PROFILE("WarpX::Evolve()");

    Real cur_time = t_new[0];
    static int last_plot_file_step = 0;

    int numsteps_max = (numsteps >= 0 && numsteps <= max_step) ? numsteps : max_step;
    bool max_time_reached = false;

    for (int step = istep[0]; step < numsteps_max && cur_time < stop_time; ++step)
    {
	if (ParallelDescriptor::IOProcessor()) {
	    std::cout << "\nSTEP " << step+1 << " starts ..." << std::endl;
	}

	ComputeDt();

	// Advance level 0 by dt
	const int lev = 0;
	{
	    // At the beginning, we have B^{n-1/2} and E^{n}.
	    // Particles have p^{n-1/2} and x^{n}.
	    
	    EvolveB(lev, 0.5*dt[lev]); // We now B^{n}

	    if (WarpX::nox > 1 || WarpX::noy > 1 || WarpX::noz > 1) {
		WarpX::FillBoundary(*Bfield[lev][0], geom[lev], Bx_nodal_flag);
		WarpX::FillBoundary(*Bfield[lev][1], geom[lev], By_nodal_flag);
		WarpX::FillBoundary(*Bfield[lev][2], geom[lev], Bz_nodal_flag);
		WarpX::FillBoundary(*Efield[lev][0], geom[lev], Ex_nodal_flag);
		WarpX::FillBoundary(*Efield[lev][1], geom[lev], Ey_nodal_flag);
		WarpX::FillBoundary(*Efield[lev][2], geom[lev], Ez_nodal_flag);
	    }


	    // Evolve particles to p^{n+1/2} and x^{n+1}
	    // Depose current, j^{n+1/2}
	    mypc->Evolve(lev,
			 *Efield[lev][0],*Efield[lev][1],*Efield[lev][2],
			 *Bfield[lev][0],*Bfield[lev][1],*Bfield[lev][2],
			 *current[lev][0],*current[lev][1],*current[lev][2],dt[lev]);
	    
	    mypc->Redistribute(false,true);  // Redistribute particles
	    
	    EvolveB(lev, 0.5*dt[lev]); // We now B^{n+1/2}
	    
	    // Fill B's ghost cells because of the next step of evolving E.
	    WarpX::FillBoundary(*Bfield[lev][0], geom[lev], Bx_nodal_flag);
	    WarpX::FillBoundary(*Bfield[lev][1], geom[lev], By_nodal_flag);
	    WarpX::FillBoundary(*Bfield[lev][2], geom[lev], Bz_nodal_flag);

	    EvolveE(lev, dt[lev]); // We now have E^{n+1}

	    ++istep[lev];
	}

	cur_time += dt[0];

	if (ParallelDescriptor::IOProcessor()) {
	    std::cout << "STEP " << step+1 << " ends." << " TIME = " << cur_time << " DT = " << dt[0]
		      << std::endl;
	}

	// sync up time
	for (int i = 0; i <= finest_level; ++i) {
	    t_new[i] = cur_time;
	}

	if (plot_int > 0 && (step+1) % plot_int == 0) {
	    last_plot_file_step = step+1;
	    WritePlotFile();
	}

	if (cur_time >= stop_time - 1.e-6*dt[0]) {
	    max_time_reached = true;
	    break;
	}
    }

    if (plot_int > 0 && istep[0] > last_plot_file_step && (max_time_reached || istep[0] >= max_step)) {
	WritePlotFile();
    }
}

void
WarpX::EvolveB (int lev, Real dt)
{
    BL_PROFILE("WarpX::EvolveB()");

    const Real* dx = geom[lev].CellSize();

    Real dtsdx[3];
#if (BL_SPACEDIM == 3)
    dtsdx[0] = dt / dx[0];
    dtsdx[1] = dt / dx[1];
    dtsdx[2] = dt / dx[2];
#elif (BL_SPACEDIM == 2)
    dtsdx[0] = dt / dx[0];
    dtsdx[1] = std::numeric_limits<Real>::quiet_NaN();
    dtsdx[2] = dt / dx[1];
#endif

    long norder = 2;
    long nstart = 0;
    int l_nodal = false;

    long nguard = Efield[lev][0]->nGrow();
    BL_ASSERT(nguard == Efield[lev][1]->nGrow());
    BL_ASSERT(nguard == Efield[lev][2]->nGrow());
    BL_ASSERT(nguard == Bfield[lev][0]->nGrow());
    BL_ASSERT(nguard == Bfield[lev][1]->nGrow());
    BL_ASSERT(nguard == Bfield[lev][2]->nGrow());

#if (BL_SPACEDIM == 3)
    long nxguard = nguard;
    long nyguard = nguard;
    long nzguard = nguard; 
#elif (BL_SPACEDIM == 2)
    long nxguard = nguard;
    long nyguard = 0;
    long nzguard = nguard; 
#endif

    for ( MFIter mfi(*Bfield[lev][0]); mfi.isValid(); ++mfi )
    {
	const Box& bx = amrex::enclosedCells(mfi.validbox());
#if (BL_SPACEDIM == 3)
	long nx = bx.length(0);
	long ny = bx.length(1);
	long nz = bx.length(2); 
#elif (BL_SPACEDIM == 2)
	long nx = bx.length(0);
	long ny = 0;
	long nz = bx.length(1); 
#endif

	warpx_push_bvec( (*Efield[lev][0])[mfi].dataPtr(),
			 (*Efield[lev][1])[mfi].dataPtr(),
			 (*Efield[lev][2])[mfi].dataPtr(),
			 (*Bfield[lev][0])[mfi].dataPtr(),
			 (*Bfield[lev][1])[mfi].dataPtr(),
			 (*Bfield[lev][2])[mfi].dataPtr(), 
			 dtsdx, dtsdx+1, dtsdx+2,
			 &nx, &ny, &nz,
			 &norder, &norder, &norder,
			 &nxguard, &nyguard, &nzguard,
			 &nstart, &nstart, &nstart,
			 &l_nodal );
    }
}

void
WarpX::EvolveE (int lev, Real dt)
{
    BL_PROFILE("WarpX::EvolveE()");

    Real mu_c2_dt = (PhysConst::mu0*PhysConst::c*PhysConst::c) * dt;

    const Real* dx = geom[lev].CellSize();

    Real dtsdx_c2[3];
#if (BL_SPACEDIM == 3)
    dtsdx_c2[0] = (PhysConst::c*PhysConst::c) * dt / dx[0];
    dtsdx_c2[1] = (PhysConst::c*PhysConst::c) * dt / dx[1];
    dtsdx_c2[2] = (PhysConst::c*PhysConst::c) * dt / dx[2];
#else
    dtsdx_c2[0] = (PhysConst::c*PhysConst::c) * dt / dx[0];
    dtsdx_c2[1] = std::numeric_limits<Real>::quiet_NaN();
    dtsdx_c2[2] = (PhysConst::c*PhysConst::c) * dt / dx[1];
#endif

    long norder = 2;
    long nstart = 0;
    int l_nodal = false;

    long nguard = Efield[lev][0]->nGrow();
    BL_ASSERT(nguard == Efield[lev][1]->nGrow());
    BL_ASSERT(nguard == Efield[lev][2]->nGrow());
    BL_ASSERT(nguard == Bfield[lev][0]->nGrow());
    BL_ASSERT(nguard == Bfield[lev][1]->nGrow());
    BL_ASSERT(nguard == Bfield[lev][2]->nGrow());
    BL_ASSERT(nguard == current[lev][0]->nGrow());
    BL_ASSERT(nguard == current[lev][1]->nGrow());
    BL_ASSERT(nguard == current[lev][2]->nGrow());

#if (BL_SPACEDIM == 3)
    long nxguard = nguard;
    long nyguard = nguard;
    long nzguard = nguard; 
#elif (BL_SPACEDIM == 2)
    long nxguard = nguard;
    long nyguard = 0;
    long nzguard = nguard; 
#endif

    for ( MFIter mfi(*Efield[lev][0]); mfi.isValid(); ++mfi )
    {
	const Box & bx = amrex::enclosedCells(mfi.validbox());
#if (BL_SPACEDIM == 3)
	long nx = bx.length(0);
	long ny = bx.length(1);
	long nz = bx.length(2); 
#elif (BL_SPACEDIM == 2)
	long nx = bx.length(0);
	long ny = 0;
	long nz = bx.length(1); 
#endif

	warpx_push_evec( (*Efield[lev][0])[mfi].dataPtr(),
			 (*Efield[lev][1])[mfi].dataPtr(),
			 (*Efield[lev][2])[mfi].dataPtr(),
			 (*Bfield[lev][0])[mfi].dataPtr(),
			 (*Bfield[lev][1])[mfi].dataPtr(),
			 (*Bfield[lev][2])[mfi].dataPtr(), 
			 (*current[lev][0])[mfi].dataPtr(),
			 (*current[lev][1])[mfi].dataPtr(),
			 (*current[lev][2])[mfi].dataPtr(),
			 &mu_c2_dt, dtsdx_c2, dtsdx_c2+1, dtsdx_c2+2,
			 &nx, &ny, &nz,
			 &norder, &norder, &norder,
			 &nxguard, &nyguard, &nzguard,
			 &nstart, &nstart, &nstart,
			 &l_nodal );
    }
}

void
WarpX::ComputeDt ()
{
    Array<Real> dt_tmp(finest_level+1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
	const Real* dx = geom[lev].CellSize();
	dt_tmp[lev]  = cfl * 1./( std::sqrt(D_TERM(  1./(dx[0]*dx[0]),
						   + 1./(dx[1]*dx[1]),
						   + 1./(dx[2]*dx[2]))) * PhysConst::c );
    }

    // Limit dt's by the value of stop_time.
    Real dt_0 = dt_tmp[0];
    const Real eps = 1.e-3*dt_0;
    if (t_new[0] + dt_0 > stop_time - eps) {
	dt_0 = stop_time - t_new[0];
    }

    dt[0] = dt_0;
    for (int lev = 1; lev <= finest_level; ++lev) {
	dt[lev] = dt[lev-1] / nsubsteps[lev];
    }
}

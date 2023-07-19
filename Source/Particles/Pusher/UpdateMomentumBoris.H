/* Copyright 2019 David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_
#define WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_

#include <AMReX_REAL.H>
#include <iostream>

/** \brief Push the particle's positions over one timestep,
 *    given the value of its momenta `ux`, `uy`, `uz` */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumBoris(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt )
{
    using namespace amrex::literals;
    //std::cout << "+++   "
//	      << ux << " , "
//	      << uy << " , "
//	      << uz << " , "
//	      << Ex << " , "
//	      << Ey << " , "
//	      << Ez << " , "
//	      << Bx << " , "
//	      << By << " , "
//	      << Bz << " , " 
//	      << q  << " , "
//	      << m << " , "
//	      << dt << " , "
//	      << "   +++" << std::endl;

    const amrex::ParticleReal econst = 0.5_prt*q*dt/m;
    //const amrex::ParticleReal econst_2 = q*dt/m;

//    // Forward Euler Kernel
//    amrex::ParticleReal ux_temp = ux;
//    amrex::ParticleReal ux_temp_2 = ux;
//    amrex::ParticleReal uy_temp = uy;
//    amrex::ParticleReal uz_temp = uz;
//    ux_temp += econst_2*(Ex + By * uz_temp);
//    uy_temp += econst_2*(Ey);
//    uz_temp += econst_2*(Ez + By * ux_temp_2);
//    std::cout << "-- temp -   "
//              << ux_temp << " , "
//	      << uy_temp << " , "
//	      << uz_temp << " , "
//	      << Ex << " , "
//	      << Ey << " , "
//	      << Ez << " , "
//	      << Bx << " , "
//	      << By << " , "
//	      << Bz << " , " 
//	      << q  << " , "
//	      << m << " , "
//	      << dt << " , "
//	      << "   ---" << std::endl;

    // First half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;
    // Compute temporary gamma factor
    //constexpr amrex::ParticleReal inv_c2 = 1._prt/(PhysConst::c*PhysConst::c);
    //const amrex::ParticleReal inv_gamma = 1._prt/std::sqrt(1._prt + (ux*ux + uy*uy + uz*uz)*inv_c2);
    const amrex::ParticleReal inv_gamma = 1._prt;
    // Magnetic rotation
    // - Compute temporary variables
    const amrex::ParticleReal tx = econst*inv_gamma*Bx;
    const amrex::ParticleReal ty = econst*inv_gamma*By;
    const amrex::ParticleReal tz = econst*inv_gamma*Bz;
    const amrex::ParticleReal tsqi = 2._prt/(1._prt + tx*tx + ty*ty + tz*tz);
    const amrex::ParticleReal sx = tx*tsqi;
    const amrex::ParticleReal sy = ty*tsqi;
    const amrex::ParticleReal sz = tz*tsqi;
    const amrex::ParticleReal ux_p = ux + uy*tz - uz*ty;
    const amrex::ParticleReal uy_p = uy + uz*tx - ux*tz;
    const amrex::ParticleReal uz_p = uz + ux*ty - uy*tx;
    // - Update momentum
    ux += uy_p*sz - uz_p*sy;
    uy += uz_p*sx - ux_p*sz;
    uz += ux_p*sy - uy_p*sx;
    // Second half-push for E
    ux += econst*Ex;
    uy += econst*Ey;
    uz += econst*Ez;
    //std::cout << "---   "
//              << ux << " , "
//	      << uy << " , "
//	      << uz << " , "
//	      << Ex << " , "
//	      << Ey << " , "
//	      << Ez << " , "
//	      << Bx << " , "
//	      << By << " , "
//	      << Bz << " , " 
//	      << q  << " , "
//	      << m << " , "
//	      << dt << " , "
//	      << "   ---" << std::endl;
}

/** \brief Push the particle's positions over one timestep using forward euler ecdi setup,
 *    given the value of its momenta `ux`, `uy`, `uz` */
AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumForEuler(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt )
{
    using namespace amrex::literals;
//    std::cout << "+++   "
//	      << ux << " , "
//	      << uy << " , "
//	      << uz << " , "
//	      << Ex << " , "
//	      << Ey << " , "
//	      << Ez << " , "
//	      << Bx << " , "
//	      << By << " , "
//	      << Bz << " , " 
//	      << q  << " , "
//	      << m << " , "
//	      << dt << " , "
//	      << "   +++" << std::endl;

    const amrex::ParticleReal econst_2 = q*dt/m;

    // Forward Euler Kernel
    amrex::ParticleReal ux_temp = ux;
    //amrex::ParticleReal ux_temp_2 = ux;
    //amrex::ParticleReal uy_temp = uy;
    //amrex::ParticleReal uz_temp = uz;
    ux += econst_2*(Ex + By * uz);
    uy += econst_2*(Ey);
    uz += econst_2*(Ez + By * ux_temp);
    //std::cout << "---   "
//            << ux_temp << " , "
//	      << uy_temp << " , "
//	      << uz_temp << " , "
//	      << Ex << " , "
//	      << Ey << " , "
//	      << Ez << " , "
//	      << Bx << " , "
//	      << By << " , "
//	      << Bz << " , " 
//	      << q  << " , "
//	      << m << " , "
//	      << dt << " , "
//	      << "   ---" << std::endl;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void UpdateMomentumRuth(
    amrex::ParticleReal& ux, amrex::ParticleReal& uy, amrex::ParticleReal& uz,
    const amrex::ParticleReal Ex, const amrex::ParticleReal Ey, const amrex::ParticleReal Ez,
    const amrex::ParticleReal Bx, const amrex::ParticleReal By, const amrex::ParticleReal Bz,
    const amrex::ParticleReal q, const amrex::ParticleReal m, const amrex::Real dt )
{
    using namespace amrex::literals;
//    std::cout << "+++   "
//	      << ux << " , "
//	      << uy << " , "
//	      << uz << " , "
//	      << Ex << " , "
//	      << Ey << " , "
//	      << Ez << " , "
//	      << Bx << " , "
//	      << By << " , "
//	      << Bz << " , " 
//	      << q  << " , "
//	      << m << " , "
//	      << dt << " , "
//	      << "   +++" << std::endl;

    const amrex::ParticleReal econst_2 = q*dt/m;

    // Ruth First half step Kernel
    amrex::ParticleReal ux_temp = ux;
    //amrex::ParticleReal ux_temp_2 = ux;
    //amrex::ParticleReal uy_temp = uy;
    //amrex::ParticleReal uz_temp = uz;
    ux += econst_2*(Ex + By * uz);
    uy += econst_2*(Ey);
    uz += econst_2*(Ez + By * ux_temp);
    //std::cout << "---   "
//            << ux_temp << " , "
//	      << uy_temp << " , "
//	      << uz_temp << " , "
//	      << Ex << " , "
//	      << Ey << " , "
//	      << Ez << " , "
//	      << Bx << " , "
//	      << By << " , "
//	      << Bz << " , " 
//	      << q  << " , "
//	      << m << " , "
//	      << dt << " , "
//	      << "   ---" << std::endl;
}

#endif // WARPX_PARTICLES_PUSHER_UPDATEMOMENTUM_BORIS_H_

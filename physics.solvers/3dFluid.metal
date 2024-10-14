//
//  3dFluid.metal
//  physics.solvers
//
//  Created by Adellar Irankunda on 10/13/24.
//
#define timestep 1.2f   //this should be somewhat (inversely) proportional to the precision of dissipation factors

kernel void Advection(const uint3 position [[thread_position_in_threadgroup]])
{
    
}

kernel void Impulse(const uint3 position [[thread_position_in_grid]])
{
    
}

kernel void Buoyancy(const uint3 position [[thread_position_in_grid]])
{
    
}

kernel void Divergence(const uint3 position [[thread_position_in_grid]])
{
    
}

kernel void Jacobi(const uint3 position [[thread_position_in_grid]])
{
    
}





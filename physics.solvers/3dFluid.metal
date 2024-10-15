#include <metal_stdlib>
using namespace metal;
//
//  3dFluid.metal
//  physics.solvers
//
//  Created by Adellar Irankunda on 10/13/24.
//
#define timestep 1.2f   //this should be somewhat (inversely) proportional to the precision of dissipation factors
#define buoyancy 1
#define weight 0.05

kernel void Advection3(texture3d<float, access::sample> VelocityIn [[texture(0)]], texture3d<float, access::sample> QuantIn [[texture(1)]], texture3d<float, access::write> QuantOut [[texture(2)]], constant float& Decay [[buffer(1)]], constant float& Dissipation [[buffer(0)]], const uint3 position [[thread_position_in_threadgroup]])
{
    constexpr sampler texSampler(filter::linear);
    float3 dim = float3(VelocityIn.get_width(), VelocityIn.get_height(), VelocityIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    
    float3 uv = float3(coord.x / dim.x, coord.y / dim.y, coord.z / dim.z);
    float3 u = VelocityIn.sample(texSampler, uv).xyz;
    coord = coord - (u * timestep);
    
    float3 advQuant = QuantIn.sample(texSampler, coord).xyz * Dissipation - Decay;
    
    QuantOut.write(float4(advQuant, 1), position);
}

kernel void Buoyancy3(texture3d<float, access::read> VelocityIn [[texture(0)]], texture3d<float, access::read> Density [[texture(1)]], texture3d<float, access::read> Temperature [[texture(2)]], texture3d<float, access::write> VelocityOut [[texture(3)]],   const uint3 position [[thread_position_in_grid]])
{
    
    float3 dim = float3(VelocityIn.get_width(), VelocityIn.get_height(), VelocityIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    
    float3 u = VelocityIn.read(position).xyz;
    float t = Temperature.read(position).x;
    float d = Density.read(position).x;
    
    if (t > 0.f)
        u += (timestep * t * buoyancy - d * weight) * float3(0, 1, 0);
    
    VelocityOut.write(float4(u, 1), position);
}

kernel void Impulse3(texture3d<float, access::read> QuantIn [[texture(0)]], texture3d<float, access::write> QuantOut [[texture(1)]], constant float3& origin [[buffer(0)]], constant float3& size [[buffer(1)]], constant float& radius [[buffer(2)]], constant float& impulseAmount [[buffer(3)]], const uint3 position [[thread_position_in_grid]])
{
    float3 dim = float3(QuantIn.get_width(), QuantIn.get_height(), QuantIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    
    float3 pos = coord/(size.xyz - 1.f) - origin.xyz;
    float mag = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
    float radSq = radius * radius;
    
    float amount = exp(-mag/radSq) * impulseAmount * timestep;
    
    QuantOut.write(float4(QuantIn.read(position).x + amount, 0, 0, 1), position);
}

kernel void EImpulse3(texture3d<float, access::read> DensityIn [[texture(0)]], texture3d<float, access::write> DensityOut [[texture(1)]], texture3d<float, access::read> ReactionIn, constant float& extinguishment [[buffer(0)]], constant float& impulseAmount [[buffer(1)]], const uint3 position [[thread_position_in_grid]])
{
    float3 dim = float3(DensityIn.get_width(), DensityIn.get_height(), DensityIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    
    float reactLife = ReactionIn.read(position).x;
    float amount = 0.f;
    
    if (reactLife < extinguishment && reactLife > 0.f)
        amount = impulseAmount * reactLife;
    
    DensityOut.write(DensityIn.read(position).x + amount, position);
}

kernel void Divergence3(const uint3 position [[thread_position_in_grid]])
{
    
}

kernel void Jacobi3(const uint3 position [[thread_position_in_grid]])
{
    
}





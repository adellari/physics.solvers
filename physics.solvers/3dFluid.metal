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
    /*
    float3 dim = float3(VelocityIn.get_width(), VelocityIn.get_height(), VelocityIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    */
    
    float3 u = VelocityIn.read(position).xyz;
    float t = Temperature.read(position).x;
    float d = Density.read(position).x;
    
    if (t > 0.f)
        u += (timestep * t * buoyancy - d * weight) * float3(0, 1, 0);
    
    VelocityOut.write(float4(u, 1), position);
}

kernel void Impulse3(texture3d<float, access::read> QuantIn [[texture(0)]], texture3d<float, access::write> QuantOut [[texture(1)]], constant float3& origin [[buffer(0)]], constant float& radius [[buffer(2)]], constant float& impulseAmount [[buffer(3)]], const uint3 position [[thread_position_in_grid]])
{
    float3 dim = float3(QuantIn.get_width(), QuantIn.get_height(), QuantIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    
    //origin is in normalized bounding box coordinates
    float3 pos = coord/(dim.xyz - 1.f) - origin.xyz;
    float mag = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
    float radSq = radius * radius;
    
    float amount = exp(-mag/radSq) * impulseAmount * timestep;
    
    QuantOut.write(float4(QuantIn.read(position).x + amount, 0, 0, 1), position);
}

kernel void EImpulse3(texture3d<float, access::read> DensityIn [[texture(0)]], texture3d<float, access::write> DensityOut [[texture(1)]], texture3d<float, access::read> ReactionIn, constant float& extinguishment [[buffer(0)]], constant float& impulseAmount [[buffer(1)]], const uint3 position [[thread_position_in_grid]])
{
    /*
    float3 dim = float3(DensityIn.get_width(), DensityIn.get_height(), DensityIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    */
    
    float reactLife = ReactionIn.read(position).x;
    float amount = 0.f;
    
    if (reactLife < extinguishment && reactLife > 0.f)
        amount = impulseAmount * reactLife;
    
    DensityOut.write(float4(DensityIn.read(position).x + amount, 0, 0, 1), position);
}

kernel void Vorticity3(texture3d<float, access::read> VelocityIn [[texture(0)]], texture3d<float, access::write> VorticityOut [[texture(1)]], const uint3 position [[thread_position_in_grid]])
{
    /*
    float3 dim = float3(VelocityIn.get_width(), VelocityIn.get_height(), VelocityIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    */
    
    uint3 left = uint3(position.x - 1, position.yz);
    uint3 right = uint3(position.x + 1, position.yz);
    uint3 up = uint3(position.x, position.y + 1, position.z);
    uint3 down = uint3(position.x, position.y - 1, position.z);
    uint3 forward = uint3(position.xy, position.z + 1);
    uint3 back = uint3(position.xy, position.z - 1);
    
    float3 uL = VelocityIn.read(left).xyz;
    float3 uR = VelocityIn.read(right).xyz;
    float3 uU = VelocityIn.read(up).xyz;
    float3 uD = VelocityIn.read(down).xyz;
    float3 uF = VelocityIn.read(forward).xyz;
    float3 uB = VelocityIn.read(back).xyz;
    
    //check to make sure this order of op and its elements are correct
    float3 vorticity = 0.5f * float3( (uU.z - uD.z) - (uF.y - uB.y), (uF.x - uB.x) - (uR.z - uL.z), (uR.y - uL.y) - (uU.x - uD.x));
    
    VorticityOut.write(float4(vorticity, 1), position);
}

kernel void Confinement3(texture3d<float, access::read> VelocityIn [[texture(0)]], texture3d<float, access::read> VorticityIn [[texture(1)]], texture3d<float, access::write> VelocityOut [[texture(2)]], constant float& Epsilon [[buffer(0)]], const uint3 position [[thread_position_in_grid]])
{
    /*
    float3 dim = float3(VelocityIn.get_width(), VelocityIn.get_height(), VelocityIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    */
    
    uint3 left = uint3(position.x - 1, position.yz);
    uint3 right = uint3(position.x + 1, position.yz);
    uint3 up = uint3(position.x, position.y + 1, position.z);
    uint3 down = uint3(position.x, position.y - 1, position.z);
    uint3 forward = uint3(position.xy, position.z + 1);
    uint3 back = uint3(position.xy, position.z - 1);
    
    float ΩL = length(VorticityIn.read(left).xyz);
    float ΩR = length(VorticityIn.read(right).xyz);
    float ΩU = length(VorticityIn.read(up).xyz);
    float ΩD = length(VorticityIn.read(down).xyz);
    float ΩF = length(VorticityIn.read(forward).xyz);
    float ΩB = length(VorticityIn.read(back).xyz);
    
    float3 Ω = VorticityIn.read(position).xyz;
    float3 eta = 0.5 * float3(ΩR - ΩL, ΩU - ΩD, ΩF - ΩB);
    eta = normalize(eta + float3(0.001, 0.001, 0.001));
    
    float3 F = timestep * Epsilon * float3(eta.y * Ω.z - eta.z * Ω.y, eta.z * Ω.x - eta.x * Ω.z, eta.x * Ω.y - eta.y * Ω.y);
    
    VelocityOut.write(float4(VelocityIn.read(position).xyz + F, 1), position);
}

kernel void Divergence3(texture3d<float, access::read> VelocityIn [[texture(0)]], texture3d<float, access::write> DivergenceOut [[texture(1)]], const uint3 position [[thread_position_in_grid]])
{ 
    /*
    float3 dim = float3(VelocityIn.get_width(), VelocityIn.get_height(), VelocityIn.get_depth());
    float3 coord = float3(position.x, position.y, position.z);
    */
    
    uint3 left = uint3(position.x - 1, position.yz);
    uint3 right = uint3(position.x + 1, position.yz);
    uint3 up = uint3(position.x, position.y + 1, position.z);
    uint3 down = uint3(position.x, position.y - 1, position.z);
    uint3 forward = uint3(position.xy, position.z + 1);
    uint3 back = uint3(position.xy, position.z - 1);
    
    float3 uL = VelocityIn.read(left).xyz;
    float3 uR = VelocityIn.read(right).xyz;
    float3 uU = VelocityIn.read(up).xyz;
    float3 uD = VelocityIn.read(down).xyz;
    float3 uF = VelocityIn.read(forward).xyz;
    float3 uB = VelocityIn.read(back).xyz;
    
    float divergence = 0.5 * ((uR.x - uL.x) + (uU.y - uD.y ) + (uF.z - uB.z));
    
    DivergenceOut.write(float4(divergence, 0, 0, 1), position);
}

kernel void Jacobi3(texture3d<float, access::read> PressureIn [[texture(0)]], texture3d<float, access::write> PressureOut [[texture(1)]], texture3d<float, access::read> DivergenceIn [[texture(2)]], const uint3 position [[thread_position_in_grid]])
{
    uint3 left = uint3(position.x - 1, position.yz);
    uint3 right = uint3(position.x + 1, position.yz);
    uint3 up = uint3(position.x, position.y + 1, position.z);
    uint3 down = uint3(position.x, position.y - 1, position.z);
    uint3 forward = uint3(position.xy, position.z + 1);
    uint3 back = uint3(position.xy, position.z - 1);
    
    float pL = PressureIn.read(left).x;
    float pR = PressureIn.read(right).x;
    float pU = PressureIn.read(up).x;
    float pD = PressureIn.read(down).x;
    float pF = PressureIn.read(forward).x;
    float pB = PressureIn.read(back).x;
    
    float divergence = DivergenceIn.read(position).x;
    float pressure = (pL + pR + pU + pD + pF + pB - divergence) / 6.0;
    
    PressureOut.write(float4(pressure, 0, 0, 1), position);
}

kernel void PoissonCorrection3(texture3d<float, access::read> VelocityIn [[texture(0)]], texture3d<float, access::write> VelocityOut [[texture(1)]], texture3d<float, access::read> PressureIn [[texture(2)]], const uint3 position [[thread_position_in_grid]])
{
    uint3 left = uint3(position.x - 1, position.yz);
    uint3 right = uint3(position.x + 1, position.yz);
    uint3 up = uint3(position.x, position.y + 1, position.z);
    uint3 down = uint3(position.x, position.y - 1, position.z);
    uint3 forward = uint3(position.xy, position.z + 1);
    uint3 back = uint3(position.xy, position.z - 1);
    
    float pL = PressureIn.read(left).x;
    float pR = PressureIn.read(right).x;
    float pU = PressureIn.read(up).x;
    float pD = PressureIn.read(down).x;
    float pF = PressureIn.read(forward).x;
    float pB = PressureIn.read(back).x;
    
    float3 u = VelocityIn.read(position).xyz;
    float3 pGrad = float3((pR - pL), (pU - pD), (pF - pB));
    u = u - pGrad;
    
    VelocityOut.write(float4(u, 1), position);
}




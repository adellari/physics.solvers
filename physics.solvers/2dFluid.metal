//
//  2dFluid.metal
//  physics.solvers
//
//  Created by Adellar Irankunda on 9/22/24.
//

#include <metal_stdlib>
using namespace metal;
#define timestep 0.125f
#define td 0.01f //1 / 2000 pixels (texel delta)
//#define DISSIPATION 0.99f
#define JACOBI_ITERATIONS 50

constexpr sampler textureSampler(filter::linear, address::repeat);

//textures needed
/*
 velocity(float2)
 pressure(float)
 temperature(float)
 density(float)
 divergence(float)
 */


float4 gatherX(device texture2d<float, access::sample>* tex, float2 uv)
{
    float4 left = tex->sample(textureSampler, float2(uv.x - td, uv.y));
    float4 right = tex->sample(textureSampler, float2(uv.x + td, uv.y));
    
    float4 up = tex->sample(textureSampler, float2(uv.x, uv.y + td));
    float4 down = tex->sample(textureSampler, float2(uv.x, uv.y - td));
    
    return float4(left.x, right.x, up.x, down.x);
}

float4x4 gather(device texture2d<float, access::sample>* tex, float2 uv, float2 ts)
{
    float4 left = tex->sample(textureSampler, float2(uv.x - ts.x, uv.y));
    float4 right = tex->sample(textureSampler, float2(uv.x + ts.x, uv.y));
    
    float4 up = tex->sample(textureSampler, float2(uv.x, uv.y + ts.y));
    float4 down = tex->sample(textureSampler, float2(uv.x, uv.y - ts.y));
    
    return float4x4(left, right, up, down);
}

//advect velocity to velocity
//advect temperature to velocity
//advect dnesity to velocity

kernel void Advection(texture2d<float, access::sample> velocity [[texture(0)]], texture2d<float, access::sample> output [[texture(1)]], constant float& dissipation [[buffer(0)]], uint2 position [[thread_position_in_grid]])
{
    const ushort2 textureSize = ushort2(velocity.get_width(), velocity.get_height());
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    float4 val = velocity.sample(textureSampler, uv);
}

//impulse (back force from advection) temperature
//impulse density

kernel void Impulse(texture2d<float, access::read_write> input [[texture(0)]], texture2d<float, access::read_write> output [[texture(1)]], uint2 position [[thread_position_in_grid]])
{
    
}

//calculate divergence of velocity

//jacobi iterations
kernel void Jacobi(texture2d<float, access::read_write> velocity [[texture(0)]], texture2d<float, access::read_write> output [[texture(1)]], uint2 position [[thread_position_in_grid]])
{
    
}

//subtract gradient of pressure from the velocity
kernel void LaplaceCorrection(texture2d<float, access::read_write> velocity [[texture(0)]], texture2d<float, access::read_write> output [[texture(1)]], uint2 position [[thread_position_in_grid]])
{
    
}



//kernel void Forces(texture2d<float,

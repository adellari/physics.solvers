//
//  2dFluid.metal
//  physics.solvers
//
//  Created by Adellar Irankunda on 9/22/24.
//

#include <metal_stdlib>
using namespace metal;
#define timestep 0.125f
#define td 0.01f                //1 / 2000 pixels (texel delta)
//#define DISSIPATION 0.99f
#define JACOBI_ITERATIONS 50
#define _Sigma 1.f               //smoke buoyancy
#define _Kappa 0.05f            //smoke weight

constexpr sampler textureSampler(filter::linear, address::repeat);

//textures needed
/*
 velocity(float2)
 pressure(float)
 temperature(float)
 density(float)
 divergence(float)
 */

float4 Sample(device texture2d<float, access::read_write>* tex, float2 uv)
{
    float4 val = float4();
    uint2 dim = uint2(tex->get_width(), tex->get_height());
    
    float2 puv = float2(dim.x * uv.x, dim.y * uv.y);
    uint2 coord = uint2(round(puv));
    
    float2 fractional = fract(puv);
    
    return val;
}

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

//velocity: rg (old) | ba (new)
//output: (r) pressure | (g) temperature | (b) density | (a) divergence
kernel void Advection(texture2d<float, access::read_write> velocity [[texture(0)]], texture2d<float, access::sample> output [[texture(1)]], constant float& dissipation [[buffer(0)]], uint2 position [[thread_position_in_grid]])
{
    const ushort2 textureSize = ushort2(velocity.get_width(), velocity.get_height());
    const float2 texelSize = float2(1.f / textureSize.x, 1.f / textureSize.y);
    
    //velocity advection
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    float2 cVelocity = velocity.read(position).rg;
    //float2 cVelocity = velocity.sample(textureSampler, uv).rg;
    
    float2 newUv = uv - (cVelocity * timestep * texelSize);
    //float2 nVelocity = velocity.sample(textureSampler, newUv).ba * dissipation;
    float2 nVelocity = velocity.read(newUv).ba * dissipation;
    
    float4 composite = output.sample(textureSampler, newUv);
    float nTemperature = composite.g * dissipation;
    float nDensity = composite.b * dissipation;
    
    //Apply Buoyancy in the y direction
    nVelocity += (timestep * (nTemperature * _Sigma - nDensity * _Kappa)) * float2(0.f, 1.f);
    
    //output.write(composite, uv);
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

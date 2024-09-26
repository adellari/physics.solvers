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

//read the current texture's velocity (u) at pixel `position`
//convert (u) into a heading
//read at `position` - `heading` = uDelta
//uDelta = u * timestep

float2 Sample(texture2d<float, access::read_write> tex, uint2 pix, float2 u)
{
    float4 val = float4();
    uint2 dim = uint2(tex.get_width(), tex.get_height());
    uint2 signs = uint2(sign(u));       //get the sign of the velocity components
    
    float2 nu = normalize(u);           //get the direction of the velocity vector
    
    uint2 ru = uint2(ceil(abs(nu)));    //ceil the absolute value of the velocity
    ru = signs * ru;                    //re-apply velocity signs
    
    uint2 pixSample = pix - ru;
    
    float2 uDelta = tex.read(pixSample).rg;   //we have our new vector
    
    //interpolate our new vector
    uDelta = u *  (1 - length(u)) + (uDelta * length(u));
    return uDelta;
}

float2 BilinearSample(texture2d<float, access::read_write> tex, uint2 pix)
{
    float2 texSize = float2(tex.get_width(), tex.get_height());
    uint2 TLCoord = clamp(pix, uint2(0), uint2(texSize) - 1);
    uint2 BRCoord = clamp(pix, uint2(0), uint2(texSize) - 1);
}

float2 BilinAdvection(texture2d<float, access::read_write> tex, uint2 pix, float2 uv)
{
    float2 texSize = float2(tex.get_width(), tex.get_height());
    float2 texCoord = (float2(pix) - uv - 0.5) / texSize;
    
    uint2 pixLow = uint2(floor(texCoord * texSize - 0.5));
    uint2 pixHigh = pixLow + 1;
    
    pixLow = clamp(pixLow, uint2(0), uint2(texSize) - 1);
    pixHigh = clamp(pixHigh, uint2(0), uint2(texSize) - 1);
    
    float2 t = fract(texCoord * texSize + 0.5);
    
    float2 LL = tex.read(pixLow).rg;
    float2 LR = tex.read(uint2(pixHigh.x, pixLow.y)).rg;
    float2 UL = tex.read(uint2(pixLow.x, pixHigh.y)).rg;
    float2 UR = tex.read(pixHigh).rg;
    
    float2 LowerSample = mix(LL, LR, t.x);
    float2 UpperSample = mix(UL, UR, t.x);
    
    return mix(LowerSample, UpperSample, t.y);

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
kernel void Advection(texture2d<float, access::sample> velocityIn [[texture(0)]], texture2d<float, access::write> velocityOut [[texture(1)]], texture2d<float, access::sample> compositeIn [[texture(2)]], texture2d<float, access::write> compositeOut [[texture(3)]] , constant float& dissipation [[buffer(0)]], uint2 position [[thread_position_in_grid]])
{
    const ushort2 textureSize = ushort2(512, 512);
    const float2 texelSize = float2(1.f / textureSize.x, 1.f / textureSize.y);
    
    //velocity advection
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    float2 cVelocity = velocityIn.sample(textureSampler, uv).rg;
    
    float2 newUv = uv - (cVelocity * timestep * texelSize);
    float2 nVelocity = velocityIn.sample(textureSampler, newUv).rg;
    
    velocityOut.write(float4(nVelocity, 0.f, 0.f) * dissipation, position);
    
    //float2 nVelocity = velocity.read(newUv).ba * dissipation;
    
    float4 compositeC = compositeIn.sample(textureSampler, newUv);
    float nTemperature = compositeC.g * dissipation;
    float nDensity = compositeC.b * dissipation;
    
    compositeOut.write(float4(compositeC.r, nTemperature, nDensity, compositeC.a), position);
    
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

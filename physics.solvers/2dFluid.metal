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

struct AdvectionParams{
    float uDissipation; //velocity
    float tDissipation; //temperature
    float dDissipation; //density
};

struct ImpulseParams{
    float2 origin;
    float radius;
    float iTemperature;
    float iDensity;
    float iAuxillary;
};

struct JacobiParams{
    float Alpha;
    float InvBeta;
};

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
    
    return float2();
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
kernel void Advection(texture2d<float, access::sample> velocityIn [[texture(0)]], texture2d<float, access::write> velocityOut [[texture(1)]], texture2d<float, access::sample> compositeIn [[texture(2)]], texture2d<float, access::write> compositeOut [[texture(3)]] , constant AdvectionParams& params [[buffer(0)]], uint2 position [[thread_position_in_grid]])
{
    const ushort2 textureSize = ushort2(velocityIn.get_width(), velocityIn.get_height());
    const float2 texelSize = float2(1.f / textureSize.x, 1.f / textureSize.y);
    
    //velocity advection
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    float2 cVelocity = velocityIn.sample(textureSampler, uv).rg;
    
    float2 newUv = uv - (cVelocity * timestep * texelSize);
    float2 nVelocity = velocityIn.sample(textureSampler, newUv).rg;
    
    
    //float2 nVelocity = velocity.read(newUv).ba * dissipation;
    
    float4 compositeC = compositeIn.sample(textureSampler, newUv);
    float nTemperature = compositeC.g * params.tDissipation;
    float nDensity = compositeC.b * params.dDissipation;
    
    compositeOut.write(float4(compositeC.r, nTemperature, nDensity, compositeC.a), position);
    
    //Apply Buoyancy in the y direction
    nVelocity += (timestep * (nTemperature * _Sigma - nDensity * _Kappa)) * float2(0.f, 1.f);
    
    velocityOut.write(float4(nVelocity, 0.f, 0.f) * params.uDissipation, position);
    //output.write(composite, uv);
}

//          [Need to Blit velocity AND composite here] before Impulse


//impulse (back force from advection) temperature
//impulse density

kernel void Impulse(texture2d<float, access::sample> compositeIn [[texture(0)]], texture2d<float, access::write> compositeOut [[texture(1)]], constant ImpulseParams* params [[buffer(0)]], uint2 position [[thread_position_in_grid]])
{
    float2 textureSize = float2(compositeIn.get_width(), compositeOut.get_height());
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    
    float d = distance(uv, params->origin);
    float impulse = 0.f;
    
    if (d < params->radius)
    {
        float a = (params->radius - d) * 0.5f;   //
        impulse = min(a, 1.f);
    }
    
    float4 comp = compositeIn.sample(textureSampler, uv);
    float temp = comp.g;
    float dens = comp.b;
    
    temp = max(0.f, mix(temp, params->iTemperature, impulse));
    dens = max(0.f, mix(dens, params->iDensity, impulse));
    
    compositeOut.write(float4(comp.r, temp, dens, comp.a), position);
}

//calculate divergence of velocity
kernel void Divergence(texture2d<float, access::sample> velocity [[texture(0)]], texture2d<float, access::sample> compositeIn [[texture(1)]], texture2d<float, access::write> compositeOut [[texture(2)]], const uint2 position [[thread_position_in_grid]])
{
    float2 textureSize = float2(velocity.get_width(), velocity.get_height());
    float2 texelSize = 1.f/textureSize;
    
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    float4 composite = compositeIn.sample(textureSampler, uv);
    
    float2 uN = velocity.sample(textureSampler, float2(uv + float2(0, -texelSize.y))).rg;
    float2 uS = velocity.sample(textureSampler, float2(uv + float2(0, texelSize.y))).rg;
    float2 uE = velocity.sample(textureSampler, float2(uv + float2(texelSize.x , 0))).rg;
    float2 uW = velocity.sample(textureSampler, float2(uv + float2(-texelSize.x , 0))).rg;
    
    float divergence = (uN.y - uS.y + uE.x - uW.x) * 0.5 / 1.f; //multiply by the inverse cell size
    compositeOut.write(float4(composite.rgb, divergence), position);
    
}

//jacobi iterations
kernel void Jacobi(texture2d<float, access::sample> compositeIn [[texture(0)]], texture2d<float, access::write> compositeOut [[texture(1)]], constant JacobiParams* jacobiParams [[buffer(0)]], uint2 position [[thread_position_in_grid]])
{
    //high level ( we're solving for a pressure field that satisfies the Pressure Poisson Eqation ∇²P(x) = 0 )
    //to this effect we start with initializing p to 0
    //we use Equation 16 (GPU Gems) to which accounts for the divergence of our current field
    //use the pressure as our new P(x) field
    //continue for ITER num of jacobian steps
    //getting closer to convergence
    
    //runs ITER number of times
    //sample the pressure texture, perform calcualtions on this value
    //use the divergence in calculation
    //write to the pressure texture
    
    float2 textureSize = float2(compositeIn.get_width(), compositeIn.get_height());
    float2 texelSize = 1.f / textureSize;
    float alpha = jacobiParams->Alpha;
    float invB = jacobiParams->InvBeta;
    
    float2 uv = float2(position.x * texelSize.x, position.y * texelSize.y);
    
    float4 compositeRead = compositeIn.sample(textureSampler, uv);
    float div = compositeRead.a;
    
    float pN = compositeIn.sample(textureSampler, uv + float2(0, -texelSize.y)).r;
    float pS = compositeIn.sample(textureSampler, uv + float2(0, texelSize.y)).r;
    float pE = compositeIn.sample(textureSampler, uv + float2(texelSize.x, 0)).r;
    float pW = compositeIn.sample(textureSampler, uv + float2(-texelSize.x, 0)).r;
    
    float prime = (pN + pS + pE + pW * alpha * div) * invB;
    compositeOut.write(float4(prime, compositeRead.gba), position);
    
}

//subtract gradient of pressure from the velocity
kernel void LaplaceCorrection(texture2d<float, access::read_write> velocity [[texture(0)]], texture2d<float, access::read_write> output [[texture(1)]], uint2 position [[thread_position_in_grid]])
{
    
}



//kernel void Forces(texture2d<float,

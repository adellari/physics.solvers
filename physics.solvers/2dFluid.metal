//
//  2dFluid.metal
//  physics.solvers
//
//  Created by Adellar Irankunda on 9/22/24.
//

#include <metal_stdlib>
using namespace metal;
#define timestep 0.125f
//#define DISSIPATION 0.99f
//#define JACOBI_ITERATIONS 50
#define _Sigma 1.0f               //smoke buoyancy
#define _Kappa 0.05f            //smoke weight


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


//advect velocity to velocity
//advect temperature to velocity
//advect dnesity to velocity

kernel void Advection(texture2d<float, access::sample> velocitySample [[texture(0)]], texture2d<float, access::sample> sourceSample [[texture(1)]], texture2d<float, access::write> sink [[texture(2)]], constant float* dissipation, const uint2 position [[thread_position_in_grid]])
{
    constexpr sampler textureSampler(filter::linear, address::clamp_to_zero);
    //position = uint2(position.x, 512 - position.y);
    const float2 textureSize = float2(velocitySample.get_width(), velocitySample.get_height());
    const float2 texelSize = float2(1.f / textureSize.x, 1.f / textureSize.y);
    float2 uv = float2(position.x * texelSize.x, position.y * texelSize.y);
    
    float2 currentVelocity = velocitySample.sample(textureSampler, uv).xy;
    float2 newPosition = float2(position.x - timestep * currentVelocity.x, position.y - timestep * currentVelocity.y);
    float2 newValue = 0.9999 * sourceSample.sample(textureSampler, texelSize * newPosition).xy;
    sink.write(float4(newValue, newValue), position);
}

kernel void Buoyancy(texture2d<float, access::read> velocityIn [[texture(0)]], texture2d<float, access::read> tempDensityIn [[texture(1)]], texture2d<float, access::write> velocityOut [[texture(2)]], const uint2 position [[thread_position_in_grid]])
{
    float2 currentVelocity = velocityIn.read(position).xy;
    float2 TemperatureDensity = tempDensityIn.read(position).xy;
    float Temperature = TemperatureDensity.x;
    float Density = TemperatureDensity.y;
    
    if (Temperature > 0.f)
    {
        float2 buoy = float2(0.f, timestep * Temperature * _Sigma - Density * _Kappa);
        currentVelocity += buoy;
    }
    
    velocityOut.write(float4(currentVelocity, currentVelocity), position);
}

//          [Need to Blit velocity AND composite here] before Impulse


//impulse (back force from advection) temperature
//impulse density

kernel void Impulse(texture2d<float, access::read> tempDensityIn [[texture(0)]], texture2d<float, access::write> tempDensityOut [[texture(1)]], constant ImpulseParams* params [[buffer(0)]], const uint2 position [[thread_position_in_grid]])
{
    //constexpr sampler textureSampler(filter::nearest, address::clamp_to_edge);
    //position = uint2(position.x, 512 - position.y);
    float2 textureSize = float2(tempDensityIn.get_width(), tempDensityIn.get_height());
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    
    float d = distance(float2(0.5, 0.5), uv);
    float impulse = 0.f;
    
    if (d < 0.1)
    {
        float a = (0.1f - d) * 0.5f;   //
        impulse = min(a, 1.f);
    }
    
    //float4 comp = tempDensityIn.sample(textureSampler, uv);
    float4 comp = tempDensityIn.read(position);
    float temp = comp.r;
    float dens = comp.g;
    
    temp = 10.f * impulse;
    dens = 1.f * impulse;
    
    tempDensityOut.write(float4(temp, dens, temp, dens), position);
}

//calculate divergence of velocity
//output: (r) pressure | (g) temperature | (b) density | (a) divergence
kernel void Divergence(texture2d<float, access::read> velocity [[texture(0)]], texture2d<float, access::write> divergenceOut [[texture(1)]], texture2d<float, access::write> pressureOut [[texture(2)]], const uint2 position [[thread_position_in_grid]])
{
    //constexpr sampler textureSampler(filter::nearest, address::clamp_to_zero);
    //position = uint2(position.x, 512 - position.y);
    float2 textureSize = float2(velocity.get_width(), velocity.get_height());
    //float2 texelSize = 1.f/textureSize;
    
    //float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    
    /*
    float2 N = uv + float2(0, -texelSize.y);
    float2 S = uv + float2(0, texelSize.y);
    float2 W = uv + float2(-texelSize.x, 0);
    float2 E = uv + float2(texelSize.x, 0);
    */
    
    uint2 N = position + uint2(0, -1);
    uint2 S = position + uint2(0, 1);
    uint2 W = position + uint2(-1, 0);
    uint2 E = position + uint2(1, 0);

    float2 uN = velocity.read(N).xy;
    float2 uS = velocity.read(S).xy;
    float2 uE = velocity.read(E).xy;
    float2 uW = velocity.read(W).xy;
    
    /*
    if (position.x >= 500) uE = 0.f;
    if (position.x <= 12) uW = 0.f;
    
    if (position.y >= 500) uS = 0.f;
    if (position.y <= 12) uN = 0.f;
    */
    
    float divergence = (0.5f/1.f) * (uE.x - uW.x + uN.y - uS.y); //multiply by the inverse cell size
    divergenceOut.write(float4(divergence, divergence, divergence, divergence), position);
    pressureOut.write(float4(0.f, 0.f, 0.f, 0.f), position);
}

//jacobi iterations
//output: (r) pressure | (g) temperature | (b) density | (a) divergence
kernel void Jacobi(texture2d<float, access::read> pressureIn [[texture(0)]], texture2d<float, access::write> pressureOut [[texture(1)]], texture2d<float, access::read> divergenceIn [[texture(2)]], /*constant JacobiParams* jacobiParams [[buffer(0)]], */const uint2 position [[thread_position_in_grid]])
{
    //high level ( we're solving for a pressure field that satisfies the Pressure Poisson Eqation ∇²P(x) = 0 )
    //to this effect we start with initializing p to 0
    //we use Equation 16 (GPU Gems) to which accounts for the divergence of our current field
    //use the pressure as our new P(x) field
    //continue for ITER num of jacobian steps
    //getting closer to convergence
    //constexpr sampler jacobiSampler(filter::nearest, address::clamp_to_edge);
    //constexpr sampler divergenceSampler(filter::nearest, address::clamp_to_edge);
    
    //runs ITER number of times
    //sample the pressure texture, perform calcualtions on this value
    //use the divergence in calculation
    //write to the pressure texture
    //position = uint2(position.x, 512 - position.y);
    float2 textureSize = float2(pressureIn.get_width(), pressureIn.get_height());
    float2 texelSize = 1.f / textureSize;
    //float alpha = jacobiParams->Alpha;
    //float invB = jacobiParams->InvBeta;
    
    //float2 uv = float2(position.x * texelSize.x, position.y * texelSize.y);
    float div = divergenceIn.read(position).r;
    //float div = divergenceIn.sample(divergenceSampler,  uv).r;
    /*
    float2 N = uv + float2(0, -texelSize.y);
    float2 S = uv + float2(0, texelSize.y);
    float2 W = uv + float2(-texelSize.x, 0);
    float2 E = uv + float2(texelSize.x, 0);
    */
    uint2 N = position + uint2(0, -1);
    uint2 S = position + uint2(0, 1);
    uint2 W = position + uint2(-1, 0);
    uint2 E = position + uint2(1, 0);

    float pN = pressureIn.read(N).r;
    float pS = pressureIn.read(S).r;
    float pE = pressureIn.read(E).r;
    float pW = pressureIn.read(W).r;
    float pC = pressureIn.read(position).r;
    
    /*
    if (position.x >= 500) pE = pC;
    if (position.x <= 12) pW = pC;
    
    if (position.y >= 500) pS = pC;
    if (position.y <= 12) pN = pC;
    */

    float prime = (pW + pE + pS + pN + (-1) * div) * 0.25f;
    pressureOut.write(float4(prime, prime, prime, prime), position);
    
}

//subtract gradient of pressure from the velocity
//output: (r) pressure | (g) temperature | (b) density | (a) divergence
kernel void PoissonCorrection(texture2d<float, access::read> velocityIn [[texture(0)]], texture2d<float, access::write> velocityOut [[texture(1)]], texture2d<float, access::read> pressureIn [[texture(2)]], const uint2 position [[thread_position_in_grid]])
{
    //constexpr sampler textureSampler(filter::nearest, address::clamp_to_edge);
    //position = uint2(position.x, 512 - position.y);
    float2 textureSize = float2(velocityIn.get_width(), velocityIn.get_height());
    float2 texelSize = 1.f / textureSize;
    
    float2 uv = float2(position.x * texelSize.x, position.y * texelSize.y);
    
    uint2 N = position + uint2(0, -1);
    uint2 S = position + uint2(0, 1);
    uint2 W = position + uint2(-1, 0);
    uint2 E = position + uint2(1, 0);

    float pN = pressureIn.read(N).r;
    float pS = pressureIn.read(S).r;
    float pE = pressureIn.read(E).r;
    float pW = pressureIn.read(W).r;
    float pC = pressureIn.read(position).r;
    
    /*
    if (position.x >= 500) pE = pC;
    if (position.x <= 12) pW = pC;
    
    if (position.y >= 500) pS = pC;
    if (position.y <= 12) pN = pC;
    */
    
    //float2 oldVelocity = velocityIn.sample(textureSampler, uv).rg;
    float2 oldVelocity = velocityIn.read(position).xy;
    float2 pGradient = float2(pE - pW, pN - pS) * 1.f;
    float2 velocity = oldVelocity - pGradient;
    //velocity = normalize(velocity) * texelSize;
    //velocityOut.write(float4(-1, -1, 0, 1), position);           //testing uv thread position drift
    velocityOut.write(float4(velocity, velocity), position);
}

kernel void Constitution(texture2d<float, access::sample> tempDensityIn [[texture(0)]], texture2d<float, access::read_write> chain [[texture(1)]], const uint2 position [[thread_position_in_grid]])
{
    constexpr sampler textureSampler(filter::linear, address::clamp_to_zero);
    //position = uint2(position.x, 512 - position.y);
    float2 textureSize = float2(tempDensityIn.get_width(), tempDensityIn.get_height());
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    float4 col = tempDensityIn.sample(textureSampler, uv);
    chain.write(float4(0, col.g, col.r, 1), position);
    
}



//kernel void Forces(texture2d<float,

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

//velocity: rg (old) | ba (new)
//output: (r) pressure | (g) temperature | (b) density | (a) divergence
kernel void Advection(texture2d<float, access::read> velocityIn [[texture(0)]], texture2d<float, access::write> velocityOut [[texture(1)]], texture2d<float, access::read> tempDensityIn [[texture(2)]], texture2d<float, access::write> tempDensityOut [[texture(3)]], texture2d<float, access::sample> velocityInSample, constant AdvectionParams* params [[buffer(0)]], const uint2 position [[thread_position_in_grid]])
{
    constexpr sampler textureSampler(filter::linear, address::clamp_to_edge);
    //position = uint2(position.x, 512 - position.y);
    const float2 textureSize = float2(velocityIn.get_width(), velocityIn.get_height());
    const float2 texelSize = float2(1.f / textureSize.x, 1.f / textureSize.y);
    
    //velocity advection
    float2 uv = float2(position.x * texelSize.x, position.y * texelSize.y);
    //float2 u = velocityIn.sample(textureSampler, uv).rg;
    float2 u = velocityIn.read(position).rg;
    float boundaryFactor = 1.f;
    if (uv.x >= 1.f - texelSize.x || uv.x <= texelSize.x || uv.y >= 1.f - texelSize.y || uv.y <= texelSize.y) boundaryFactor = 0.f;
    
    float2 coord = uv - (u * texelSize * timestep);
    //coord = clamp(float2(0), float2(1), coord);
    //velocity advection
    float2 newVelocity = velocityInSample.sample(textureSampler, coord).rg * 0.9999 * boundaryFactor;
    //float2 newVelocity = velocityIn.read(coord).rg * 0.9999 * boundaryFactor;
    
    
    //temperature and density advection
    //float4 tempDensity = tempDensityIn.sample(textureSampler, uv) * boundaryFactor;
    float4 tempDensity = tempDensityIn.read(position) * boundaryFactor;
    float nTemperature = tempDensity.r * 0.99;
    float nDensity = tempDensity.g * 0.99999;
    
    //compositeOut.write(float4(compositeC.r, nTemperature, nDensity, 0), position);
    tempDensityOut.write(float4(nTemperature, nDensity, 0.f, 1.f), position);
    
    
    //Apply Buoyancy and weight in the y direction
    if (nTemperature > 0.f )
    {
        newVelocity += (timestep * (nTemperature * _Sigma - nDensity * _Kappa) * float2(0.f, 1.f));
    }
    
    
    //if (length(nVelocity) < 0.05f)
        //nVelocity = uv * length(float2(0.5f, 0.5f) - uv);
    velocityOut.write(float4(newVelocity, 0.f, 0.f), position);
    //output.write(composite, uv);
}

//          [Need to Blit velocity AND composite here] before Impulse


//impulse (back force from advection) temperature
//impulse density

kernel void Impulse(texture2d<float, access::read> tempDensityIn [[texture(0)]], texture2d<float, access::write> tempDensityOut [[texture(1)]], constant ImpulseParams* params [[buffer(0)]], const uint2 position [[thread_position_in_grid]])
{
    constexpr sampler textureSampler(filter::nearest, address::clamp_to_edge);
    //position = uint2(position.x, 512 - position.y);
    float2 textureSize = float2(tempDensityIn.get_width(), tempDensityIn.get_height());
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    
    float d = distance(float2(0.5, 0.5), uv);
    float impulse = 0.f;
    
    if (d < params->radius)
    {
        float a = (params->radius - d) * 0.5f;   //
        impulse = min(a, 1.f);
    }
    
    //float4 comp = tempDensityIn.sample(textureSampler, uv);
    float4 comp = tempDensityIn.read(position);
    float temp = comp.r;
    float dens = comp.g;
    
    temp =  max(0.f, mix(temp, params->iTemperature, impulse));
    dens =  max(0.f, mix(dens, params->iDensity, impulse));
    
    tempDensityOut.write(float4(temp, dens, 0.f, 0.f), position);
}

//calculate divergence of velocity
//output: (r) pressure | (g) temperature | (b) density | (a) divergence
kernel void Divergence(texture2d<float, access::read> velocity [[texture(0)]], texture2d<float, access::write> divergenceOut [[texture(1)]], texture2d<float, access::write> pressureOut [[texture(2)]], const uint2 position [[thread_position_in_grid]])
{
    constexpr sampler textureSampler(filter::nearest, address::clamp_to_edge);
    //position = uint2(position.x, 512 - position.y);
    float2 textureSize = float2(velocity.get_width(), velocity.get_height());
    float2 texelSize = 1.f/textureSize;
    
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    
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

    float2 uN = velocity.read(N).rg;
    float2 uS = velocity.read(S).rg;
    float2 uE = velocity.read(E).rg;
    float2 uW = velocity.read(W).rg;
    
    if (uv.x >= 1.f - texelSize.x) uE = 0.f;
    if (uv.x <= texelSize.x) uW = 0.f;
    
    if (uv.y >= 1.f - texelSize.y) uS = 0.f;
    if (uv.y <= texelSize.y) uN = 0.f;
    
    float divergence = 0.5f * (uW.x - uE.x + uN.y - uS.y); //multiply by the inverse cell size
    divergenceOut.write(float4(divergence, 0.f, 0.f, 0.f), position);
    pressureOut.write(float4(0.f, 0.f, 0.f, 0.f), position);
}

//jacobi iterations
//output: (r) pressure | (g) temperature | (b) density | (a) divergence
kernel void Jacobi(texture2d<float, access::read> pressureIn [[texture(0)]], texture2d<float, access::write> pressureOut [[texture(1)]], texture2d<float, access::read> divergenceIn [[texture(2)]], constant JacobiParams* jacobiParams [[buffer(0)]], const uint2 position [[thread_position_in_grid]])
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
    float alpha = jacobiParams->Alpha;
    float invB = jacobiParams->InvBeta;
    
    float2 uv = float2(position.x * texelSize.x, position.y * texelSize.y);
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
    
    if (uv.x >= 1.f - texelSize.x) pE = pC;
    if (uv.x <= texelSize.x) pW = pC;
    
    if (uv.y >= 1.f - texelSize.y) pS = pC;
    if (uv.y <= texelSize.y) pN = pC;
    

    float prime = (pN + pS + pE + pW + -1 * div)/4.f;
    pressureOut.write(float4(prime, 0.f, 0.f, 0.f), position);
    
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
    
    if (uv.x >= 1.f - texelSize.x) pE = pC;
    if (uv.x <= texelSize.x) pW = pC;
    
    if (uv.y >= 1.f - texelSize.y) pS = pC;
    if (uv.y <= texelSize.y) pN = pC;
    
    //float2 oldVelocity = velocityIn.sample(textureSampler, uv).rg;
    float2 oldVelocity = velocityIn.read(position).rg;
    float2 pGradient = float2(pW - pE, pN - pS);
    float2 velocity = oldVelocity - pGradient;
    //velocity = normalize(velocity) * texelSize;
    //velocityOut.write(float4(-1, -1, 0, 1), position);           //testing uv thread position drift
    velocityOut.write(float4(velocity, 0.f, 0.f), position);
}

kernel void Constitution(texture2d<float, access::sample> tempDensityIn [[texture(0)]], texture2d<float, access::read_write> chain [[texture(1)]], const uint2 position [[thread_position_in_grid]])
{
    constexpr sampler textureSampler(filter::linear, address::clamp_to_edge);
    //position = uint2(position.x, 512 - position.y);
    float2 textureSize = float2(tempDensityIn.get_width(), tempDensityIn.get_height());
    float2 uv = float2(position.x / textureSize.x, position.y / textureSize.y);
    float4 col = tempDensityIn.sample(textureSampler, uv);
    chain.write(float4(col.g * 255, 0, 0, 1), position);
    
}



//kernel void Forces(texture2d<float,

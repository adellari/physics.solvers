//
//  Viewport.metal
//  physics.solvers
//
//  Created by Adellar Irankunda on 11/1/24.
//

#include <metal_stdlib>
using namespace metal;

struct Camera
{
    float4x4 worldToCamera;
    float4x4 projection;
    float3 position;
};

struct Ray
{
    float3 origin;
    float3 direction;
};

struct RayHit
{
    float3 position;
    float3 normal;
    float distance;
};

Ray CreateRay(float3 origin, float3 direction)
{
    Ray r = Ray();
    r.direction = normalize(direction);
    r.origin = origin;
    
    return r;
}

Ray CreateCameraRay()
{
    Ray r = Ray();
    
    return r;
}


kernel void Renderer(texture2d<float, access::write> output [[texture(0)]], constant Camera& camera [[buffer(0)]], const uint3 position [[thread_position_in_grid]])
{
    
}

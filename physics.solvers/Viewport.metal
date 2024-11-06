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

RayHit CreateRayhit()
{
    RayHit rh = RayHit();
    rh.distance = INFINITY;
    rh.normal = float3();
    rh.position = float3();
    
    return rh;
}

Ray CreateRay(float3 origin, float3 direction)
{
    Ray r = Ray();
    r.direction = normalize(direction);
    r.origin = origin;
    
    return r;
}

Ray CreateCameraRay(float2 uv, constant Camera& cam)
{
    Ray r = Ray();
    float3 origin = (float4(cam.position, 1) * cam.worldToCamera).xyz;
    float3 dir = (float4(uv, 0, 1) * cam.projection).xyz;
    
    dir = normalize((float4(dir, 0) * cam.worldToCamera).xyz);
    r = CreateRay(origin, dir);
    
    return r;
}

void IntersectGroundPlane(Ray ray, thread RayHit* hit)
{
    float t = -ray.origin.y / ray.direction.y;
    
    if (t < hit->distance && t > 0.f)
    {
        hit->normal = float3(0.f, 1.f, 0.f);
        hit->distance = t;
    }
}


kernel void Renderer(texture2d<float, access::write> output [[texture(0)]], texture2d<float, access::write> chain [[texture(1)]], constant Camera& camera [[buffer(0)]], const uint2 position [[thread_position_in_grid]])
{
    const ushort2 textureSize = ushort2(output.get_width(), output.get_height());
    const float2 coord = float2((float)position.x / (float)textureSize.x, (float)position.y / (float)textureSize.y);
    const float2 uv = coord * 2.f - 1.f;
    
    float4 col = float4();
    
    Ray ray = CreateCameraRay(uv, camera);
    RayHit hit = CreateRayhit();

    col = float4(ray.direction, 1.f);
    IntersectGroundPlane(ray, &hit);
    //if (hit.distance < INFINITY)
        //col = float4(hit.normal, 1.f);
    chain.write(abs(col), position);
    output.write(col, position);
}

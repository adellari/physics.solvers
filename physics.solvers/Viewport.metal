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
    float t = 0.01f -ray.origin.y / ray.direction.y;
    
    if (t < hit->distance && t > 0.f)
    {
        hit->normal = float3(0.f, 1.f, 0.f);
        hit->distance = t;
        hit->position = ray.direction * t + ray.origin;
    }
}


kernel void Renderer(texture2d<float, access::write> output [[texture(0)]], texture2d<float, access::write> chain [[texture(1)]], constant Camera& camera [[buffer(0)]], const uint2 position [[thread_position_in_grid]])
{
    const ushort2 textureSize = ushort2(output.get_width(), output.get_height());
    const float2 texel = float2(1.f/(float)textureSize.x, 1.f/(float)textureSize.y);
    const float2 coord = float2((float)position.x / (float)textureSize.x, (float)position.y / (float)textureSize.y);
    const float2 uv = coord * 2.f - 1.f;
    
    float4 col = float4();
    
    Ray ray = CreateCameraRay(uv, camera);
    Ray raydx = CreateCameraRay(uv + float2(texel.x, 0), camera);
    Ray raydy = CreateCameraRay(uv + float2(0, texel.y), camera);
    RayHit hit = CreateRayhit();
    ///we have dfdx and dfdy in metal
    
    col = float4(ray.direction, 1.f);
    IntersectGroundPlane(ray, &hit);
    
    
    //unfiltered checkerboard
    /*
    if (hit.distance < INFINITY){
        float2 floored = floor(hit.position.xz * 12.f);
        float pat = fmod(floored.x + floored.y, 2.f);
        col = float4(pat, pat, pat, 1.f);
    }
     */
    
    //filtered checkerboard
    if (hit.distance < INFINITY)
    {
        float3 ddx_pos = (raydx.origin - raydx.direction) * dot(raydx.origin - hit.position, hit.normal) / dot(raydx.direction, hit.normal);
        float3 ddy_pos = (raydy.origin - raydy.direction) * dot(raydy.origin - hit.position, hit.normal) / dot(raydy.direction, hit.normal);
        
        float2 guv = hit.position.xz * 12.f;
        float2 guvdx = ddx_pos.xz * 12.f - guv;
        float2 guvdy = ddy_pos.xz * 12.f - guv;
        
        float2 w = max(abs(guvdx), abs(guvdy)) + 0.01;
        float2 i = 2.f * (abs(fract( (guv-0.5*w)/2.f ) - 0.5) -abs(fract( (guv+0.5*w)/2.f ) - 0.5f)) / w;
        float pat = 0.5 - 0.5 * i.x * i.y;
        col = float4(pat, pat, pat, 1.f);
        /*
         vec2 w = max(abs(ddx), abs(ddy)) + 0.01;
             // analytical integral (box filter)
             vec2 i = 2.0*(abs(fract((p-0.5*w)/2.0)-0.5)-abs(fract((p+0.5*w)/2.0)-0.5))/w;
             // xor pattern
             return 0.5 - 0.5*i.x*i.y;
         */
    }
    
    
    chain.write(abs(col), position);
    output.write(col, position);
}

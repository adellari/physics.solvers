//
//  Renderer.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 10/22/24.
//
import Metal
import simd

class Renderer
{
    var device : MTLDevice
    var commandQueue : MTLCommandQueue
    var tracer : MTLComputePipelineState
    var viewportTexture : MTLTexture
    var chain : (Ping : MTLTexture, Pong : MTLTexture)?
    var fluidTexture : MTLTexture?
    var camera : CameraParams
    var azimuth : Double = 0.0
    var elevation : Double = 0.0
    
    struct CameraParams {
        var position : simd_float3;
        var cameraMatrix : simd_float4x4;
        var projectionMatrix : simd_float4x4;
    }
    
    init (queue : MTLCommandQueue) throws
    {
        self.device = queue.device
        commandQueue = queue
        
        let lib = try device.makeDefaultLibrary(bundle: .main)
        let traceFunc = lib.makeFunction(name: "Renderer")!
        tracer = try device.makeComputePipelineState(function: traceFunc)
        let viewportDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 1024, height: 512, mipmapped: false)
        viewportDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        
        self.camera = CameraParams(position: simd_float3(0, 1, 0), cameraMatrix: simd_float4x4(), projectionMatrix: simd_float4x4())
        self.viewportTexture = device.makeTexture(descriptor: viewportDesc)!
        //phi works properly
        //theta spins around the forward vector
        self.camera.cameraMatrix = self.Camera(eye: simd_float3(0, 0, 0), theta: 0, phi: 0)
        self.camera.projectionMatrix = self.Projection(fov: 60, aspect: 2.0, near: 0.1, far: 1000)
    }
    
    public func CreateChain(format : MTLPixelFormat)
    {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: self.viewportTexture.width, height: self.viewportTexture.height, mipmapped: false)
        descriptor.usage = .shaderWrite
        self.chain = (device.makeTexture(descriptor: descriptor)!, device.makeTexture(descriptor: descriptor)!)
    }
    
    public func Draw() -> MTLTexture?
    {
        //draw the scene from the resultant 3d velocity texture
        let renderBuffer = commandQueue.makeCommandBuffer()!
        let renderEncoder = renderBuffer.makeComputeCommandEncoder()!
        renderEncoder.setComputePipelineState(tracer)
        self.camera.cameraMatrix = self.Camera(eye: simd_float3(0, 0, 0), theta: 0, phi: .pi/2)
        renderEncoder.setBytes(&camera, length: MemoryLayout<CameraParams>.size, index: 0)
        renderEncoder.setTexture(self.viewportTexture, index: 0)
        if (self.chain?.Ping != nil) {renderEncoder.setTexture(self.chain!.Ping, index: 1)}
        renderEncoder.dispatchThreadgroups(MTLSize(width: 32, height: 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 16, depth: 1))
        //dispatch
        renderEncoder.endEncoding()
        renderBuffer.commit()
        
        
        return self.chain?.Ping
    }
    
    //theta = inclination with range [-π/2, π/2] where 0 is the equator
    //phi = azimuth with range [0, 2π]
    //this is a view matrix
    ///Typically this is constructed from origin and forward vectors provided
    ///since we're working in spherical coordinates, we must calculate the forward vector using
    ///spherical to cartesian conversion.
    ///
    ///x=rsin(θ)cos(ϕ)
    ///y=rsin(θ)sin(ϕ)
    ///z=rcos(θ)
    func Camera(eye: simd_float3, theta: Float, phi: Float) -> simd_float4x4
    {
        
    
        let forward = simd_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
        
        var helper = simd_float3(0, 1, 0) //check to see if this isn't aligned with forward
        var right = simd_float3(0)
        var up = simd_float3(0)
        if (abs(forward.y) > 0.99)
        {
            helper = simd_float3(1, 0, 0)
            up = normalize(cross(helper, forward))
            right = normalize(cross(forward, up))
        }
        else{
            right = normalize(cross(helper, forward))
            up = normalize(cross(forward, right))
        }
            
        
        
        
        let camMatrix = simd_float4x4(simd_float4(right.x, up.x, forward.x, 0),
                                      simd_float4(right.y, up.y, forward.y, 0),
                                      simd_float4(right.z, up.z, forward.z, 0),
                                      simd_float4(-dot(right, eye), -dot(up, eye), -dot(forward, eye), 1));
        return camMatrix;
    }
    
    func Projection(fov: Float, aspect: Float, near: Float, far: Float) -> float4x4{
        let rFov = fov * (.pi / 180.0)
        let y = 1.0 / tan(rFov / 2.0)
        let x = y / aspect
        
        let zRange = far - near
        let zNear = near
        
        let m00 = x
        let m11 = y
        let m22 = (far + near) / zRange
        let m32 = 2 * zNear * far / zRange

        let rotationMatrix = simd_float4x4(
                simd_float4(1, 0, 0, 0),
                simd_float4(0, 0, 1, 0),
                simd_float4(0, 1, 0, 1),
                simd_float4(1, 1, 1, 1)
            )
        
        
        return float4x4([simd_float4(m00, 0, 0, 0),
                        simd_float4(0, m11, 0, 0),
                        simd_float4(0, 0, m22, -1),
                        simd_float4(0, 0, m32, 0)])
        
        
    }
    
}

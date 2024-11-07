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
        var cameraMatrix : simd_float4x4;
        var projectionMatrix : simd_float4x4;
        var position : simd_float3;
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
        
        self.camera = CameraParams(cameraMatrix: simd_float4x4(), projectionMatrix: simd_float4x4(), position: simd_float3(0, 1, 0))
        self.viewportTexture = device.makeTexture(descriptor: viewportDesc)!
        //phi works properly
        //theta spins around the forward vector
        self.camera.cameraMatrix = self.Camera(eye: simd_float3(0, 0, 0), theta: 0, phi: Float(self.azimuth) + .pi/2)
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
        //we always add π to phi, in addition to negating x, and y axis to swap to left handed basis
        self.camera.cameraMatrix = self.Camera(eye: simd_float3(0, 0, 0), theta: Float(self.elevation), phi: Float(self.azimuth) + .pi)
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
    //couldn't get this basis creation with lin algebra working, now using trigonometry:
    ///This is a right-handed system, we shift phi by π and negate x and y
    ///in the camera projection matrix, to get to a left handed system
    ///where phi is rotation around basis y, from [0, 2π]
    ///and theta is rotation around basis x, from [-π/2, π/2]
    ///theta = 0, phi = 0 (+π conversion) is left-hand system forward vector
    ///theta = π/2, phi = 0 (+π conversion) is the left-hand system up vector
    ///theta = 0, phi = π/2 (+π conversion) is the left-hand system right vector
    ///https://www.3dgep.com/understanding-the-view-matrix/#:~:text=model%20might%20look-,like%20this%3A,-FPS%20camera%2C%20right
    func Camera(eye: simd_float3, theta: Float, phi: Float) -> simd_float4x4
    {
        
        let sinTheta = sin(theta)
        let sinPhi = sin(phi)
        let cosTheta = cos(theta)
        let cosPhi = cos(phi)
        
        let xAxis = simd_float3(cosPhi, 0, -sinPhi)
        let yAxis = simd_float3(sinPhi * sinTheta, cosTheta, cosPhi * sinTheta)
        let zAxis = simd_float3(sinPhi * cosTheta, -sinTheta, cosTheta * cosPhi)
    
        let camMatrix = simd_float4x4(simd_float4(xAxis.x, yAxis.x, zAxis.x, 0),
                                      simd_float4(xAxis.y, yAxis.y, zAxis.y, 0),
                                      simd_float4(xAxis.z, yAxis.z, zAxis.z, 0),
                                      simd_float4(-dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1));
        return camMatrix;
    }
    
    func Projection(fov: Float, aspect: Float, near: Float, far: Float) -> float4x4{
        let rFov = fov * (.pi / 180.0)
        let y = 1.0 / tan(rFov / 2.0)
        let x = y / aspect
        
        let zRange = far - near
        let zNear = near
        
        let m00 = -x
        let m11 = -y
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

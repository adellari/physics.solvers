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
    var chain : (MTLTexture, MTLTexture)?
    var fluidTexture : MTLTexture?
    var cameraMatrix : simd_float4x4 = float4x4()
    var projectionMatrix : simd_float4x4 = float4x4()
    
    init (queue : MTLCommandQueue) throws
    {
        self.device = queue.device
        commandQueue = queue
        
        let lib = try device.makeDefaultLibrary(bundle: .main)
        let traceFunc = lib.makeFunction(name: "Tracer")!
        tracer = try device.makeComputePipelineState(function: traceFunc)
        self.projectionMatrix = Projection(fov: 60, aspect: 1, near: 0.1, far: 1000)
        self.cameraMatrix = Camera(eye: simd_float3(repeating: 0), theta: 0, phi: 0)
    }
    
    public func Draw()
    {
        //draw the scene from the resultant 3d velocity texture
    }
    
    func Camera(eye: simd_float3, theta: Float, phi: Float) -> simd_float4x4
    {
        let sinTheta = sin(theta)
        let cosTheta = cos(theta)
        let sinPhi = sin(phi)
        let cosPhi = cos(phi)
        
        let xAxis = simd_float3(cosPhi, 0, -sinPhi)
        let yAxis = simd_float3(cosPhi * sinTheta, cosTheta, cosPhi * sinTheta)
        let zAxis = simd_float3(sinPhi * cosTheta, -sinTheta, cosTheta * cosPhi);
        
        let camMatrix = simd_float4x4(simd_float4(xAxis.x, yAxis.x, zAxis.x, 0),
                                      simd_float4(xAxis.y, yAxis.y, zAxis.y, 0),
                                      simd_float4(xAxis.z, yAxis.z, zAxis.z, 0),
                                      simd_float4(-dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1));
        return camMatrix;
    }
    
    func Projection(fov: Float, aspect: Float, near: Float, far: Float) -> float4x4{
        var rFov = fov * (.pi / 180.0)
        let y = 1.0 / tan(rFov / 2.0)
        let x = y / aspect
        
        let zRange = far - near
        let zNear = near
        
        var m00 = -x
        var m11 = -y
        var m22 = (far + near) / zRange
        var m23 = -1
        var m32 = 2 * zNear * far / zRange

        let rotationMatrix = simd_float4x4(
                simd_float4(0, -1, 0, 0),
                simd_float4(1, 0, 0, 0),
                simd_float4(0, 0, 1, 0),
                simd_float4(0, 0, 0, 1)
            )
        
        
        return float4x4([simd_float4(m00, 0, 0, 0),
                        simd_float4(0, m11, 0, 0),
                        simd_float4(0, 0, m22, -1),
                        simd_float4(0, 0, m32, 0)])
        
        
    }
    
}

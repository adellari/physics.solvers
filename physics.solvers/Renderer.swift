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
    var cameraMatrix : simd_float4x4
    
    init (queue : MTLCommandQueue) throws
    {
        self.device = queue.device
        commandQueue = queue
        
        let lib = try device.makeDefaultLibrary(bundle: .main)
        let traceFunc = lib.makeFunction(name: "Tracer")!
        tracer = try device.makeComputePipelineState(function: traceFunc)
        
        cameraMatrix = Camera(eye: SIMD3<Float>(.zero), )
    }
    
    public func Draw()
    {
        //draw the scene from the resultant 3d velocity texture
    }
    
    func Camera(eye: simd_float3, theta: Float, phi: Float) -> simd_float4x4
    {
        let sinTheta = sin(theta)
        let cosTheta = cos(theta)
    }
    
}

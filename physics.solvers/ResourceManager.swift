//
//  ResourceManager.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 9/22/24.
//

import Foundation
import Metal

class Fluid 
{
    var velocity : MTLTexture? // in : rg, out: ba
    var compositeIn : MTLTexture?
    var compositeOut : MTLTexture?
    //pressure, temperature, density, divergence
    
    init(device: MTLDevice)
    {
        let velocityDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 500, height: 500, mipmapped: false)
        let compositeDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 500, height: 500, mipmapped: false)
        
        velocityDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        compositeDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        
        self.velocity = device.makeTexture(descriptor: velocityDesc)
        self.compositeIn = device.makeTexture(descriptor: compositeDesc)
        self.compositeOut = device.makeTexture(descriptor: compositeDesc)
    }
}

class ResourceManager
{
    var device : MTLDevice?
    var commandQueue : MTLCommandQueue?
    var advectionPipeline : MTLComputePipelineState?
    var gradientPipeline : MTLComputePipelineState?
    var jacobiPipeline : MTLComputePipelineState?
    
    init(_device: MTLDevice) throws
    {
        let library = try _device.makeDefaultLibrary(bundle: .main)
        let advection = library.makeFunction(name: "Advection")
        let jacobi = library.makeFunction(name: "Jacobi")
        let gradient = library.makeFunction(name: "Gradient")
        
        self.jacobiPipeline = try library.device.makeComputePipelineState(function: jacobi!)
        self.advectionPipeline = try library.device.makeComputePipelineState(function: advection!)
        self.gradientPipeline = try library.device.makeComputePipelineState(function: gradient!)
        
        self.commandQueue = device!.makeCommandQueue()
        self.device = _device
    }
}

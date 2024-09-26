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
    var velocityIn : MTLTexture? // in : rg, out: ba
    var velocityOut : MTLTexture?
    var compositeIn : MTLTexture?
    var compositeOut : MTLTexture?
    //pressure, temperature, density, divergence
    
    init(device: MTLDevice)
    {
        let velocityRDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: 512, height: 512, mipmapped: false)
        let velocityWDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: 512, height: 512, mipmapped: false)
        let compositeRDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 512, height: 512, mipmapped: false)
        let compositeWDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 512, height: 512, mipmapped: false)
        
        velocityRDesc.usage = MTLTextureUsage([.shaderRead])
        velocityWDesc.usage = MTLTextureUsage([.shaderWrite])
        compositeRDesc.usage = MTLTextureUsage([.shaderRead])
        compositeWDesc.usage = MTLTextureUsage([.shaderWrite])
        
        self.velocityIn = device.makeTexture(descriptor: velocityRDesc)
        self.velocityOut = device.makeTexture(descriptor: velocityWDesc)
        self.compositeIn = device.makeTexture(descriptor: compositeRDesc)
        self.compositeOut = device.makeTexture(descriptor: compositeWDesc)
    }
    
    func Blit(source : MTLTexture, usage: MTLTextureUsage = [.shaderRead] ) throws -> MTLTexture {
        let device = source.device
        
        var descriptor = MTLTextureDescriptor()
        descriptor.width = source.width
        descriptor.height = source.height
        descriptor.pixelFormat = source.pixelFormat
        descriptor.usage = usage
        
        let destination = device.makeTexture(descriptor: descriptor)
        
        let commandQueue = device.makeCommandQueue()
        let commandBuffer = commandQueue?.makeCommandBuffer()
    
        
        let blitEncoder = commandBuffer?.makeBlitCommandEncoder()
        blitEncoder?.copy(from: source, to: destination!)
        blitEncoder?.endEncoding()
        
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        
        return destination!
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

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
    struct AdvectionParams{
        var uDissipation: Float
        var tDissipation: Float
        var dDissipation: Float
    }
    
    struct ImpulseParams{
        var origin: SIMD2<Float>
        var radius: Float
        var iTemperature: Float
        var iDensity: Float
        var iAuxillary: Float
    }
    
    var velocityIn : MTLTexture? // in : rg, out: ba
    var velocityOut : MTLTexture?
    var compositeIn : MTLTexture?
    var compositeOut : MTLTexture?
    
    var advectionParams : AdvectionParams
    var impulseParams : ImpulseParams
    
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
        
        self.advectionParams = AdvectionParams(uDissipation: 0.999999, tDissipation: 0.99, dDissipation: 0.999999)
        self.impulseParams = ImpulseParams(origin: SIMD2<Float>(0.5, 0.0), radius: 0.1, iTemperature: 10, iDensity: 1, iAuxillary: 0)
    }
    
    func Blit(source : MTLTexture, usage: MTLTextureUsage = [.shaderRead] ) -> MTLTexture {
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
    
    func Blit(commandBuffer: MTLCommandBuffer, source : MTLTexture, usage: MTLTextureUsage = [.shaderRead] ) throws -> MTLTexture {
        let device = source.device
        
        var descriptor = MTLTextureDescriptor()
        descriptor.width = source.width
        descriptor.height = source.height
        descriptor.pixelFormat = source.pixelFormat
        descriptor.usage = usage
        
        let destination = device.makeTexture(descriptor: descriptor)
    
        
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        blitEncoder?.copy(from: source, to: destination!)
        blitEncoder?.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return destination!
    }
}

class ResourceManager
{
    var device : MTLDevice?
    var commandQueue : MTLCommandQueue?
    var advectionPipeline : MTLComputePipelineState
    var gradientPipeline : MTLComputePipelineState
    var jacobiPipeline : MTLComputePipelineState
    var impulsePipeline : MTLComputePipelineState
    var divergencePipeline : MTLComputePipelineState
    var fluid : Fluid
    
    let groupSize : MTLSize = MTLSize(width: 32, height: 32, depth: 1)
    let threadsPerGroup : MTLSize = MTLSize(width: 512 / 32, height: 512 / 32, depth: 1)
    
    init(_device: MTLDevice) throws
    {
        let library = try _device.makeDefaultLibrary(bundle: .main)
        let advection = library.makeFunction(name: "Advection")
        let impulse = library.makeFunction(name: "Impulse")
        let jacobi = library.makeFunction(name: "Jacobi")
        let gradient = library.makeFunction(name: "Gradient")
        let divergence = library.makeFunction(name: "Divergence")
        
        self.jacobiPipeline = try library.device.makeComputePipelineState(function: jacobi!)
        self.advectionPipeline = try library.device.makeComputePipelineState(function: advection!)
        self.gradientPipeline = try library.device.makeComputePipelineState(function: gradient!)
        self.impulsePipeline = try library.device.makeComputePipelineState(function: impulse!)
        self.divergencePipeline = try library.device.makeComputePipelineState(function: divergence!)
        
        self.commandQueue = device!.makeCommandQueue()
        self.device = _device
        
        self.fluid = .init(device: _device)
    }
    
    func Draw() throws
    {
        let commandBuffer = commandQueue!.makeCommandBuffer()
        let encoder = commandBuffer!.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(self.advectionPipeline)
        encoder.setTexture(fluid.velocityIn!, index: 0)
        encoder.setTexture(fluid.velocityOut!, index: 1)
        encoder.setTexture(fluid.compositeIn!, index: 2)
        encoder.setTexture(fluid.compositeOut!, index: 3)
        encoder.setBytes(&fluid.advectionParams, length: MemoryLayout<Fluid.AdvectionParams>.stride, index: 0)
        encoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        let blitEncoder = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder.copy(from: fluid.velocityOut!, to: fluid.velocityIn!)
        blitEncoder.copy(from: fluid.compositeOut!, to: fluid.compositeIn!)
        blitEncoder.endEncoding()
        
        encoder.setComputePipelineState(self.impulsePipeline)
        encoder.setTexture(fluid.compositeIn, index: 0)
        encoder.setTexture(fluid.compositeOut, index: 1)
        encoder.setBytes(&fluid.impulseParams, length:MemoryLayout<Fluid.ImpulseParams>.stride, index: 0)
        encoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        
        //confirm we don't need to blit the composite texture here
        
        encoder.setComputePipelineState(self.divergencePipeline)
        encoder.setTexture(fluid.velocityIn, index: 0)
        encoder.setTexture(fluid.compositeIn, index: 1)
        encoder.setTexture(fluid.compositeOut, index: 2)
        encoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        
        let blitEncoder2 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder2.copy(from: fluid.compositeOut!, to: fluid.compositeIn!)
        blitEncoder2.endEncoding()
        //need to set the pressure to 0 at this step, before doing jacobi iteration
        
        //encoder!.setComputePipelineState(self.impulsePipeline)
        
        //advect
        //blit composite & velocity textures
        //apply impulse
        //blit composite texture
        //
        
        encoder.setComputePipelineState(self.divergencePipeline)
    }
    
}

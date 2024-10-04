//
//  ResourceManager.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 9/22/24.
//

import MetalKit
import Foundation
import Metal
import SwiftUI

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
    
    struct JacobiParams{
        var Alpha: Float
        var InvBeta: Float
    }
    
    var velocityIn : MTLTexture? // in : rg, out: ba
    var velocityOut : MTLTexture?
    var compositeIn : MTLTexture?
    var compositeOut : MTLTexture?
    var tempDensityIn : MTLTexture
    var tempDensityOut : MTLTexture
    var divergenceIn : MTLTexture
    var divergenceOut : MTLTexture
    var pressureIn : MTLTexture
    var pressureOut : MTLTexture
    var chain : MTLTexture?
    
    var advectionParams : AdvectionParams
    var impulseParams : ImpulseParams
    var jacobiParams : JacobiParams
    
    //pressure, temperature, density, divergence
    
    init(device: MTLDevice)
    {
        let velocityRDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: 512, height: 512, mipmapped: false)
        let velocityWDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: 512, height: 512, mipmapped: false)
        let compositeRDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 512, height: 512, mipmapped: false)
        let compositeWDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 512, height: 512, mipmapped: false)
        let swapDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 512, height: 512, mipmapped: false)
        let singleRDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 512, height: 512, mipmapped: false)
        let singleWDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 512, height: 512, mipmapped: false)
        
        velocityRDesc.usage = MTLTextureUsage([.shaderRead])
        velocityWDesc.usage = MTLTextureUsage([.shaderWrite])
        compositeRDesc.usage = MTLTextureUsage([.shaderRead])
        compositeWDesc.usage = MTLTextureUsage([.shaderWrite])
        swapDesc.usage = MTLTextureUsage([.shaderWrite, .shaderRead])
        singleRDesc.usage = MTLTextureUsage([.shaderRead])
        singleWDesc.usage = MTLTextureUsage([.shaderWrite])
        
        self.velocityIn = device.makeTexture(descriptor: velocityRDesc)
        self.velocityOut = device.makeTexture(descriptor: velocityWDesc)
        self.compositeIn = device.makeTexture(descriptor: compositeRDesc)
        self.compositeOut = device.makeTexture(descriptor: compositeWDesc)
        self.pressureIn = device.makeTexture(descriptor: singleRDesc)!
        self.pressureOut = device.makeTexture(descriptor: singleWDesc)!
        self.divergenceIn = device.makeTexture(descriptor: singleRDesc)!
        self.divergenceOut = device.makeTexture(descriptor: singleWDesc)!
        self.tempDensityIn = device.makeTexture(descriptor: velocityRDesc)!
        self.tempDensityOut = device.makeTexture(descriptor: velocityWDesc)!
        self.chain = device.makeTexture(descriptor: swapDesc)
        
        self.advectionParams = AdvectionParams(uDissipation: 0.99999, tDissipation: 0.99, dDissipation: 0.9999)
        self.impulseParams = ImpulseParams(origin: SIMD2<Float>(0.5, 0), radius: 0.1, iTemperature: 10, iDensity: 1, iAuxillary: 0)
        self.jacobiParams = JacobiParams(Alpha: -1, InvBeta: 0.25)
        
    }
    
    func Blit(source : MTLTexture, usage: MTLTextureUsage = [.shaderRead] ) -> MTLTexture {
        let device = source.device
        
        let descriptor = MTLTextureDescriptor()
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

class ResourceManager : NSObject
{
    var device : MTLDevice?
    var commandQueue : MTLCommandQueue?
    var advectionPipeline : MTLComputePipelineState
    var poissonPipeline : MTLComputePipelineState
    var jacobiPipeline : MTLComputePipelineState
    var impulsePipeline : MTLComputePipelineState
    var divergencePipeline : MTLComputePipelineState
    var constitutionPipeline : MTLComputePipelineState
    var fluid : Fluid
    var metalView : MTKView
    
    let groupSize : MTLSize = MTLSize(width: 32, height: 32, depth: 1)
    let threadsPerGroup : MTLSize = MTLSize(width: 512 / 32, height: 512 / 32, depth: 1)
    
    init(_device: MTLDevice) throws
    {
        let library = try _device.makeDefaultLibrary(bundle: .main)
        let advection = library.makeFunction(name: "Advection")
        let impulse = library.makeFunction(name: "Impulse")
        let jacobi = library.makeFunction(name: "Jacobi")
        let poisson = library.makeFunction(name: "PoissonCorrection")
        let divergence = library.makeFunction(name: "Divergence")
        let constitution = library.makeFunction(name: "Constitution")
        
        self.jacobiPipeline = try library.device.makeComputePipelineState(function: jacobi!)
        self.advectionPipeline = try library.device.makeComputePipelineState(function: advection!)
        self.poissonPipeline = try library.device.makeComputePipelineState(function: poisson!)
        self.impulsePipeline = try library.device.makeComputePipelineState(function: impulse!)
        self.divergencePipeline = try library.device.makeComputePipelineState(function: divergence!)
        self.constitutionPipeline = try library.device.makeComputePipelineState(function: constitution!)
        
        self.commandQueue = _device.makeCommandQueue()
        self.device = _device
        
        self.fluid = .init(device: _device)
        self.metalView = MTKView(frame: CGRect(x:0, y:0, width:512, height:512), device: _device)
        super.init()
        
        self.metalView.delegate = self
        self.metalView.framebufferOnly = false
        self.metalView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha:1)
        //self.metalView.colorPixelFormat = .rgba8Uint
        //self.metalView.colorPixelFormat = .rgba32Float
    }
    
    func Draw()
    {
        let commandBuffer = commandQueue!.makeCommandBuffer()
        let encoder1 = commandBuffer!.makeComputeCommandEncoder()!
        
        encoder1.setComputePipelineState(self.advectionPipeline)
        encoder1.setTexture(fluid.velocityIn!, index: 0)
        encoder1.setTexture(fluid.velocityOut!, index: 1)
        encoder1.setTexture(fluid.tempDensityIn, index: 2)
        encoder1.setTexture(fluid.tempDensityOut, index: 3)
        encoder1.setBytes(&fluid.advectionParams, length: MemoryLayout<Fluid.AdvectionParams>.stride, index: 0)
        encoder1.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder1.endEncoding()
        
        let blitEncoder1 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder1.copy(from: fluid.velocityOut!, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(512, 512, 1), to: fluid.velocityIn!, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOriginMake(0,0,0))
        blitEncoder1.copy(from: fluid.tempDensityOut, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(512, 512, 1), to: fluid.tempDensityIn, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOriginMake(0,0,0))
        blitEncoder1.endEncoding()
        
        
        
        let encoder2 = commandBuffer!.makeComputeCommandEncoder()!
        encoder2.setComputePipelineState(self.impulsePipeline)
        encoder2.setTexture(fluid.tempDensityIn, index: 0)
        encoder2.setTexture(fluid.tempDensityOut, index: 1)
        encoder2.setBytes(&fluid.impulseParams, length:MemoryLayout<Fluid.ImpulseParams>.stride, index: 0)
        encoder2.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder2.endEncoding()
        
        //confirm we don't need to blit the composite texture here
        let blitImpulse = commandBuffer!.makeBlitCommandEncoder()!
        blitImpulse.copy(from: fluid.tempDensityOut, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(512, 512, 1),to: fluid.tempDensityIn, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOriginMake(0,0,0))
        blitImpulse.endEncoding()
        
        let divEncoder = commandBuffer!.makeComputeCommandEncoder()!
        divEncoder.setComputePipelineState(self.divergencePipeline)
        divEncoder.setTexture(fluid.velocityIn!, index: 0)
        divEncoder.setTexture(fluid.divergenceOut, index: 1)
        divEncoder.setTexture(fluid.pressureOut, index: 2)
        divEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        divEncoder.endEncoding()
        
        let blitEncoder3 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder3.copy(from: fluid.divergenceOut, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(512, 512, 1), to: fluid.divergenceIn, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOriginMake(0,0,0))
        blitEncoder3.copy(from: fluid.pressureOut, to: fluid.pressureIn)
        blitEncoder3.endEncoding()
        //blitEncoder.endEncoding()
        //need to set the pressure to 0 at this step, before doing jacobi iteration
        
        
        for _ in 0..<50
        {
            let _c = commandBuffer!.makeComputeCommandEncoder()!
            _c.setComputePipelineState(self.jacobiPipeline)
            _c.setBytes(&fluid.jacobiParams, length: MemoryLayout<Fluid.JacobiParams>.stride, index: 0)
            _c.setTexture(fluid.pressureIn, index: 0)
            _c.setTexture(fluid.pressureOut, index: 1)
            _c.setTexture(fluid.divergenceIn, index: 2)
            _c.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
            _c.endEncoding()
            let _b = commandBuffer!.makeBlitCommandEncoder()!
            _b.copy(from: fluid.pressureOut, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(512, 512, 1), to: fluid.pressureIn, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOriginMake(0,0,0))
            _b.endEncoding()
        }
        
        let encoder3 = commandBuffer!.makeComputeCommandEncoder()!
        encoder3.setComputePipelineState(self.poissonPipeline)
        encoder3.setTexture(fluid.velocityIn!, index: 0)
        encoder3.setTexture(fluid.velocityOut!, index: 1)
        encoder3.setTexture(fluid.pressureIn, index: 2)
        encoder3.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder3.endEncoding()
        
        let blitEncoder4 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder4.copy(from: fluid.velocityOut!, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOriginMake(0, 0, 0), sourceSize: MTLSizeMake(512, 512, 1), to: fluid.velocityIn!, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOriginMake(0,0,0))
        blitEncoder4.endEncoding()
        
        /*
        let chainEncoder = commandBuffer!.makeComputeCommandEncoder()!
        chainEncoder.setComputePipelineState(self.constitutionPipeline)
        chainEncoder.setTexture(fluid.tempDensityIn, index: 0)
        chainEncoder.setTexture(fluid.chain!, index: 1)
        chainEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        chainEncoder.endEncoding()
        */
        //advect
        //blit composite & velocity textures
        //apply impulse
        //blit composite texture
        //
        
        //encoder.endEncoding()
        commandBuffer!.commit()
    }
    
}



 extension ResourceManager: MTKViewDelegate
 {
 
 func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize)
 {
 
 }
 
 func draw(in view: MTKView)
 {
 guard let drawable = view.currentDrawable else { return }
 Draw()
 
 let commandBuffer = commandQueue!.makeCommandBuffer()!
 let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
 
 blitEncoder.copy(from: fluid.chain!, to: drawable.texture)
 blitEncoder.endEncoding()
 commandBuffer.present(drawable)
 commandBuffer.commit()
 }
 
 }
 
 
 struct MetalViewRepresentable: NSViewRepresentable {
 
 var metalView: MTKView
 
 func makeNSView(context: Context) -> MTKView{
 return metalView
 }
 
 func updateNSView(_ nsView: MTKView, context: Context)
 {
 
 }
 }
 

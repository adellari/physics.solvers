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
    
    struct Surface {
        var Ping : MTLTexture
        var Pong : MTLTexture
    }
    
    /*
    var velocityIn : MTLTexture // in : rg, out: ba
    var velocityOut : MTLTexture
    var compositeIn : MTLTexture?
    var compositeOut : MTLTexture?
    var tempDensityIn : MTLTexture
    var tempDensityOut : MTLTexture
    var divergenceIn : MTLTexture
    var divergenceOut : MTLTexture
    var pressureIn : MTLTexture
    var pressureOut : MTLTexture
    */
    var chain : MTLTexture?
    
    var Velocity : Surface
    var Temperature : Surface
    var Density : Surface
    var Divergence : Surface
    var Pressure : Surface
    
    var advectionParams : AdvectionParams
    var impulseParams : ImpulseParams
    var jacobiParams : JacobiParams
    
    //pressure, temperature, density, divergence
    
    init(device: MTLDevice)
    {
        let velocityRDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: 512, height: 512, mipmapped: false)
        let velocityWDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: 512, height: 512, mipmapped: false)
        let swapDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 1024, height: 1024, mipmapped: true)
        let singleRDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 512, height: 512, mipmapped: false)
        let singleWDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 512, height: 512, mipmapped: false)
        
        velocityRDesc.usage = MTLTextureUsage([.shaderRead])
        //velocityRDesc.allowGPUOptimizedContents = true
        velocityWDesc.usage = MTLTextureUsage([.shaderWrite])
        //velocityWDesc.allowGPUOptimizedContents = true

        swapDesc.usage = MTLTextureUsage([.shaderWrite, .shaderRead])
        swapDesc.mipmapLevelCount = 2
        singleRDesc.usage = MTLTextureUsage([.shaderRead])
        //singleRDesc.allowGPUOptimizedContents = true
        singleWDesc.usage = MTLTextureUsage([.shaderWrite])
        //singleWDesc.allowGPUOptimizedContents = true
        
        self.Velocity = Surface(Ping: device.makeTexture(descriptor: velocityRDesc)!, Pong: device.makeTexture(descriptor: velocityWDesc)!)
        self.velocityOut =
        self.pressureIn = device.makeTexture(descriptor: singleRDesc)!
        self.pressureOut = device.makeTexture(descriptor: singleWDesc)!
        self.divergenceIn = device.makeTexture(descriptor: singleRDesc)!
        self.divergenceOut = device.makeTexture(descriptor: singleWDesc)!
        self.tempDensityIn = device.makeTexture(descriptor: velocityRDesc)!
        self.tempDensityOut = device.makeTexture(descriptor: velocityWDesc)!
        self.chain = device.makeTexture(descriptor: swapDesc)
        
        self.advectionParams = AdvectionParams(uDissipation: 0.99999, tDissipation: 0.99, dDissipation: 0.9999)
        self.impulseParams = ImpulseParams(origin: SIMD2<Float>(0.5, 0), radius: 0.1, iTemperature: 10, iDensity: 1, iAuxillary: 0)
        self.jacobiParams = JacobiParams(Alpha: -1.0, InvBeta: 0.25)
        
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
    var buoyancyPipeline : MTLComputePipelineState
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
        let buoyancy = library.makeFunction(name: "Buoyancy")
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
        self.buoyancyPipeline = try library.device.makeComputePipelineState(function: buoyancy!)
        
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
        
        
        let advectVelocity = commandBuffer!.makeComputeCommandEncoder()!
        var diffuse = 0.99;
        advectVelocity.setComputePipelineState(self.advectionPipeline)
        advectVelocity.setTexture(fluid.velocityIn, index: 0)
        advectVelocity.setTexture(fluid.velocityIn, index: 1)
        advectVelocity.setTexture(fluid.velocityOut, index: 2)
        advectVelocity.setBytes(&diffuse, length: MemoryLayout<Float>.stride, index: 0)
        advectVelocity.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        advectVelocity.label = "advectVelocity"
        advectVelocity.endEncoding()
        
        
        
        let blitEncoder1 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder1.copy(from: fluid.velocityOut, to: fluid.velocityIn)
        //blitEncoder1.copy(from: fluid.tempDensityOut, to: fluid.tempDensityIn)
        blitEncoder1.endEncoding()
        
        let advectTemperatureDensity = commandBuffer!.makeComputeCommandEncoder()!
        advectTemperatureDensity.setComputePipelineState(self.advectionPipeline)
        diffuse = 0.9999
        advectTemperatureDensity.setTexture(fluid.velocityIn, index: 0)
        advectTemperatureDensity.setTexture(fluid.tempDensityIn, index: 1)
        advectTemperatureDensity.setTexture(fluid.tempDensityOut, index: 2)
        advectTemperatureDensity.setBytes(&diffuse, length: MemoryLayout<Float>.stride, index: 0)
        advectTemperatureDensity.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        advectTemperatureDensity.label = "advectTemperatureAndDensity"
        advectTemperatureDensity.endEncoding()
        
        let blitTemperatureDensity = commandBuffer!.makeBlitCommandEncoder()!
        blitTemperatureDensity.copy(from: fluid.tempDensityOut, to: fluid.tempDensityIn)
        blitTemperatureDensity.endEncoding()
        
        let buoyancyEncoder = commandBuffer!.makeComputeCommandEncoder()!
        buoyancyEncoder.setComputePipelineState(self.buoyancyPipeline)
        buoyancyEncoder.setTexture(fluid.velocityIn, index: 0)
        buoyancyEncoder.setTexture(fluid.tempDensityIn, index: 1)
        buoyancyEncoder.setTexture(fluid.velocityOut, index: 2)
        buoyancyEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        buoyancyEncoder.label = "Buoyancy"
        buoyancyEncoder.endEncoding()
        
        let blitVelocity = commandBuffer!.makeBlitCommandEncoder()!
        blitVelocity.copy(from: fluid.velocityOut, to: fluid.velocityIn)
        blitVelocity.endEncoding()
        
        let encoder2 = commandBuffer!.makeComputeCommandEncoder()!
        encoder2.setComputePipelineState(self.impulsePipeline)
        encoder2.setTexture(fluid.tempDensityIn, index: 0)
        encoder2.setTexture(fluid.tempDensityOut, index: 1)
        encoder2.setBytes(&fluid.impulseParams, length:MemoryLayout<Fluid.ImpulseParams>.stride, index: 0)
        encoder2.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder2.label = "Impulse"
        encoder2.endEncoding()
        
        //confirm we don't need to blit the composite texture here
        let blitImpulse = commandBuffer!.makeBlitCommandEncoder()!
        blitImpulse.copy(from: fluid.tempDensityOut, to: fluid.tempDensityIn)
        blitImpulse.endEncoding()
        
        let divEncoder = commandBuffer!.makeComputeCommandEncoder()!
        divEncoder.setComputePipelineState(self.divergencePipeline)
        divEncoder.setTexture(fluid.velocityIn, index: 0)
        divEncoder.setTexture(fluid.divergenceOut, index: 1)
        divEncoder.setTexture(fluid.pressureOut, index: 2)
        divEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        divEncoder.label = "Divergence"
        divEncoder.endEncoding()
        
        let blitEncoder3 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder3.copy(from: fluid.divergenceOut, to: fluid.divergenceIn)
        blitEncoder3.copy(from: fluid.pressureOut, to: fluid.pressureIn)
        blitEncoder3.endEncoding()
        //blitEncoder.endEncoding()
        //need to set the pressure to 0 at this step, before doing jacobi iteration
        
        
        for _ in 0..<50
        {
            let _c = commandBuffer!.makeComputeCommandEncoder()!
            _c.setComputePipelineState(self.jacobiPipeline)
            //_c.setBytes(&fluid.jacobiParams, length: MemoryLayout<Fluid.JacobiParams>.stride, index: 0)
            _c.setTexture(fluid.pressureIn, index: 0)
            _c.setTexture(fluid.pressureOut, index: 1)
            _c.setTexture(fluid.divergenceIn, index: 2)
            _c.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
            _c.endEncoding()
            let _b = commandBuffer!.makeBlitCommandEncoder()!
            _b.copy(from: fluid.pressureOut, to: fluid.pressureIn)
            _b.label = "Jacobi"
            _b.endEncoding()
        }
        
        let encoder3 = commandBuffer!.makeComputeCommandEncoder()!
        encoder3.setComputePipelineState(self.poissonPipeline)
        encoder3.setTexture(fluid.velocityIn, index: 0)
        encoder3.setTexture(fluid.velocityOut, index: 1)
        encoder3.setTexture(fluid.pressureIn, index: 2)
        encoder3.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder3.label = "Poisson Correction"
        encoder3.endEncoding()
        
        let blitEncoder4 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder4.copy(from: fluid.velocityOut, to: fluid.velocityIn)
        blitEncoder4.endEncoding()
        
        
        let chainEncoder = commandBuffer!.makeComputeCommandEncoder()!
        chainEncoder.setComputePipelineState(self.constitutionPipeline)
        chainEncoder.setTexture(fluid.velocityIn, index: 0)
        chainEncoder.setTexture(fluid.chain, index: 1)
        chainEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        chainEncoder.endEncoding()
        
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
     let blitSwapchain = commandBuffer.makeBlitCommandEncoder()!
     let nextTexture = drawable.texture
     let chain = fluid.chain!
     //print(nextTexture.width, nextTexture.height)
     blitSwapchain.copy(from: chain, to: nextTexture)
     blitSwapchain.endEncoding()
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
 

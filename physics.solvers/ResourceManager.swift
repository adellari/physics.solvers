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
        //velocityRDesc.textureType = .type2DMultisample
        //velocityRDesc.sampleCount = 4
        velocityRDesc.allowGPUOptimizedContents = true
        velocityWDesc.usage = MTLTextureUsage([.shaderWrite])
        velocityWDesc.allowGPUOptimizedContents = true
        //velocityWDesc.textureType = .type2DMultisample
        //velocityWDesc.sampleCount = 4

        swapDesc.usage = MTLTextureUsage([.shaderWrite, .shaderRead])
        swapDesc.mipmapLevelCount = 2
        
        singleRDesc.usage = MTLTextureUsage([.shaderRead])
        singleRDesc.allowGPUOptimizedContents = true
        //singleRDesc.swizzle = .init(red: .red, green: .green, blue: .blue, alpha: .alpha)
        singleRDesc.compressionType = .lossless
        
        singleWDesc.usage = MTLTextureUsage([.shaderWrite])
        singleWDesc.compressionType = .lossless
        singleWDesc.allowGPUOptimizedContents = true
        
        self.Velocity = Surface(Ping: device.makeTexture(descriptor: velocityRDesc)!, Pong: device.makeTexture(descriptor: velocityWDesc)!)
        self.Pressure = Surface(Ping: device.makeTexture(descriptor: singleRDesc)!, Pong: device.makeTexture(descriptor: singleWDesc)!)
        self.Divergence = Surface(Ping: device.makeTexture(descriptor: singleRDesc)!, Pong: device.makeTexture(descriptor: singleWDesc)!)
        self.Temperature = Surface(Ping: device.makeTexture(descriptor: singleRDesc)!, Pong: device.makeTexture(descriptor: singleWDesc)!)
        self.Density = Surface(Ping: device.makeTexture(descriptor: singleRDesc)!, Pong: device.makeTexture(descriptor: singleWDesc)!)
        self.chain = device.makeTexture(descriptor: swapDesc)
        
        self.Velocity.Ping.label = "Velocity Read"
        self.Velocity.Pong.label = "Velocity Write"
        
        self.Temperature.Ping.label = "Temperature Read"
        self.Temperature.Pong.label = "Temperature Write"
        
        self.Density.Ping.label = "Density Read"
        self.Density.Pong.label = "Density Write"
        
        self.Pressure.Ping.label = "Pressure Read"
        self.Pressure.Pong.label = "Pressure Write"
        
        self.Divergence.Ping.label = "Divergence Read"
        self.Divergence.Pong.label = "Divergence Write"
        
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
    var frames = 0
    
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
        advectVelocity.setTexture(fluid.Velocity.Ping, index: 0)
        advectVelocity.setTexture(fluid.Velocity.Ping, index: 1)
        advectVelocity.setTexture(fluid.Velocity.Pong, index: 2)
        advectVelocity.setBytes(&diffuse, length: MemoryLayout<Float>.stride, index: 0)
        advectVelocity.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        advectVelocity.label = "Advect Velocity"
        advectVelocity.endEncoding()
        
        
        
        let blitEncoder1 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder1.copy(from: fluid.Velocity.Pong, to: fluid.Velocity.Ping)
        blitEncoder1.label = "Swap Velocity"
        //blitEncoder1.copy(from: fluid.tempDensityOut, to: fluid.tempDensityIn)
        blitEncoder1.endEncoding()
        
        let advectTemperature = commandBuffer!.makeComputeCommandEncoder()!
        advectTemperature.setComputePipelineState(self.advectionPipeline)
        advectTemperature.setTexture(fluid.Velocity.Ping, index: 0)
        advectTemperature.setTexture(fluid.Temperature.Ping, index: 1)
        advectTemperature.setTexture(fluid.Temperature.Pong, index: 2)
        advectTemperature.label = "Advect Temperature"
        advectTemperature.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        advectTemperature.endEncoding()
        
        let blitTempAdvection = commandBuffer!.makeBlitCommandEncoder()!
        blitTempAdvection.copy(from: fluid.Temperature.Pong, to: fluid.Temperature.Ping)
        blitTempAdvection.label = "Swap Temperature"
        blitTempAdvection.endEncoding()
        
        let advectDensity = commandBuffer!.makeComputeCommandEncoder()!
        advectDensity.setComputePipelineState(self.advectionPipeline)
        advectDensity.setTexture(fluid.Velocity.Ping, index: 0)
        advectDensity.setTexture(fluid.Density.Ping, index: 1)
        advectDensity.setTexture(fluid.Density.Pong, index: 2)
        advectDensity.label = "Advect Density"
        advectDensity.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        advectDensity.endEncoding()
        
        let blitDensityAdvection = commandBuffer!.makeBlitCommandEncoder()!
        blitDensityAdvection.copy(from: fluid.Density.Pong, to: fluid.Density.Ping)
        blitDensityAdvection.label = "Swap Density"
        blitDensityAdvection.endEncoding()
        //diffuse = 0.9999
        
        let buoyancyEncoder = commandBuffer!.makeComputeCommandEncoder()!
        buoyancyEncoder.setComputePipelineState(self.buoyancyPipeline)
        buoyancyEncoder.setTexture(fluid.Velocity.Ping, index: 0)
        buoyancyEncoder.setTexture(fluid.Density.Ping, index: 1)
        buoyancyEncoder.setTexture(fluid.Temperature.Ping, index: 2)
        buoyancyEncoder.setTexture(fluid.Velocity.Pong, index: 3)
        buoyancyEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        buoyancyEncoder.label = "Buoyancy"
        buoyancyEncoder.endEncoding()
        
        let blitVelocity = commandBuffer!.makeBlitCommandEncoder()!
        blitVelocity.copy(from: fluid.Velocity.Pong, to: fluid.Velocity.Ping)
        blitVelocity.label = "Swap Velocity 2"
        blitVelocity.endEncoding()
        
        let encoder2 = commandBuffer!.makeComputeCommandEncoder()!
        encoder2.setComputePipelineState(self.impulsePipeline)
        encoder2.setTexture(fluid.Temperature.Pong, index: 0)
        encoder2.setTexture(fluid.Density.Pong, index: 1)
        encoder2.setBytes(&fluid.impulseParams, length:MemoryLayout<Fluid.ImpulseParams>.stride, index: 0)
        encoder2.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder2.label = "Impulse"
        encoder2.endEncoding()
        
        //confirm we don't need to blit the composite texture here
        let blitImpulse = commandBuffer!.makeBlitCommandEncoder()!
        blitImpulse.copy(from: fluid.Temperature.Pong, to: fluid.Temperature.Ping)
        blitImpulse.copy(from: fluid.Density.Pong, to: fluid.Density.Ping)
        blitImpulse.label = "Swap Temperature and Density from Impulses"
        blitImpulse.endEncoding()
        
        let divEncoder = commandBuffer!.makeComputeCommandEncoder()!
        divEncoder.setComputePipelineState(self.divergencePipeline)
        divEncoder.setTexture(fluid.Velocity.Ping, index: 0)
        divEncoder.setTexture(fluid.Divergence.Pong, index: 1)
        divEncoder.setTexture(fluid.Pressure.Pong, index: 2)
        divEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        divEncoder.label = "Divergence"
        divEncoder.endEncoding()
        
        let blitEncoder3 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder3.copy(from: fluid.Divergence.Pong, to: fluid.Divergence.Ping)
        blitEncoder3.copy(from: fluid.Pressure.Pong, to: fluid.Pressure.Ping)
        blitEncoder3.label = "Swap Divergence"
        blitEncoder3.endEncoding()
        //blitEncoder.endEncoding()
        //need to set the pressure to 0 at this step, before doing jacobi iteration
        
        
        for _ in 0..<50
        {
            let _c = commandBuffer!.makeComputeCommandEncoder()!
            _c.setComputePipelineState(self.jacobiPipeline)
            //_c.setBytes(&fluid.jacobiParams, length: MemoryLayout<Fluid.JacobiParams>.stride, index: 0)
            _c.setTexture(fluid.Pressure.Ping, index: 0)
            _c.setTexture(fluid.Pressure.Pong, index: 1)
            _c.setTexture(fluid.Divergence.Ping, index: 2)
            _c.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
            _c.label = "Jacobi"
            _c.endEncoding()
            
            let _b = commandBuffer!.makeBlitCommandEncoder()!
            _b.copy(from: fluid.Pressure.Pong, to: fluid.Pressure.Ping)
            _b.label = "Swap Pressure"
            _b.endEncoding()
        }
        
        let encoder3 = commandBuffer!.makeComputeCommandEncoder()!
        encoder3.setComputePipelineState(self.poissonPipeline)
        encoder3.setTexture(fluid.Velocity.Ping, index: 0)
        encoder3.setTexture(fluid.Velocity.Pong, index: 1)
        encoder3.setTexture(fluid.Pressure.Ping, index: 2)
        encoder3.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder3.label = "Poisson Correction"
        encoder3.endEncoding()
        
        let blitEncoder4 = commandBuffer!.makeBlitCommandEncoder()!
        blitEncoder4.copy(from: fluid.Velocity.Pong, to: fluid.Velocity.Ping)
        blitEncoder4.label = "Swap Velocity"
        blitEncoder4.endEncoding()
        
        
        let chainEncoder = commandBuffer!.makeComputeCommandEncoder()!
        chainEncoder.setComputePipelineState(self.constitutionPipeline)
        chainEncoder.setTexture(fluid.Velocity.Ping, index: 0)
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
     
     if (frames == 0)
     {
         do {
             let captureManager = MTLCaptureManager.shared()
             let descriptor = MTLCaptureDescriptor()
             descriptor.captureObject = self.device
             //try captureManager.startCapture(with: descriptor)
         }
         catch {
             print("failed to make capture device")
         }
     }
     
     
     if frames == 6 {
         //MTLCaptureManager.shared().stopCapture()
     }
     frames += 1
     
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
 

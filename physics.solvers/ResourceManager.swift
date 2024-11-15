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
        let velocityDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: 512, height: 512, mipmapped: false)
        let swapDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 1024, height: 1024, mipmapped: true)
        let singleCDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 512, height: 512, mipmapped: false)
        
        velocityDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        velocityDesc.allowGPUOptimizedContents = true
        velocityDesc.compressionType = .lossless

        swapDesc.usage = MTLTextureUsage([.shaderWrite, .shaderRead])
        swapDesc.mipmapLevelCount = 2
        
        singleCDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        singleCDesc.allowGPUOptimizedContents = true
        singleCDesc.compressionType = .lossless
        
        self.Velocity = Surface(Ping: device.makeTexture(descriptor: velocityDesc)!, Pong: device.makeTexture(descriptor: velocityDesc)!)
        self.Pressure = Surface(Ping: device.makeTexture(descriptor: singleCDesc)!, Pong: device.makeTexture(descriptor: singleCDesc)!)
        self.Divergence = Surface(Ping: device.makeTexture(descriptor: singleCDesc)!, Pong: device.makeTexture(descriptor: singleCDesc)!)
        self.Temperature = Surface(Ping: device.makeTexture(descriptor: singleCDesc)!, Pong: device.makeTexture(descriptor: singleCDesc)!)
        self.Density = Surface(Ping: device.makeTexture(descriptor: singleCDesc)!, Pong: device.makeTexture(descriptor: singleCDesc)!)
        self.chain = device.makeTexture(descriptor: swapDesc)
        
        self.Velocity.Ping.label = "Velocity Ping"
        self.Velocity.Pong.label = "Velocity Pong"
        
        self.Temperature.Ping.label = "Temperature Ping"
        self.Temperature.Pong.label = "Temperature Pong"
        
        self.Density.Ping.label = "Density Ping"
        self.Density.Pong.label = "Density Pong"
        
        self.Pressure.Ping.label = "Pressure Ping"
        self.Pressure.Pong.label = "Pressure Pong"
        
        self.Divergence.Ping.label = "Divergence Ping"
        self.Divergence.Pong.label = "Divergence Pong"
        
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

class ResourceManager2D : NSObject
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
    var constituteObstaclePipeline : MTLComputePipelineState
    var fluid : Fluid
    //var metalView : MTKView
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
        let constituteObstacle = library.makeFunction(name: "ConstituteObstacle")
        
        self.jacobiPipeline = try library.device.makeComputePipelineState(function: jacobi!)
        self.advectionPipeline = try library.device.makeComputePipelineState(function: advection!)
        self.poissonPipeline = try library.device.makeComputePipelineState(function: poisson!)
        self.impulsePipeline = try library.device.makeComputePipelineState(function: impulse!)
        self.divergencePipeline = try library.device.makeComputePipelineState(function: divergence!)
        self.constitutionPipeline = try library.device.makeComputePipelineState(function: constitution!)
        self.buoyancyPipeline = try library.device.makeComputePipelineState(function: buoyancy!)
        self.constituteObstaclePipeline = try library.device.makeComputePipelineState(function: constituteObstacle!)
        
        self.commandQueue = _device.makeCommandQueue()
        self.device = _device
        
        self.fluid = .init(device: _device)
    }
    
    func Swap(surface : inout Fluid.Surface)
    {
        let temp = surface.Ping
        surface.Ping = surface.Pong
        surface.Pong = temp
    }
    
    func Simulate(obstacleTex : MTLTexture? = nil, chainOutput : MTLTexture? = nil) -> MTLTexture?
    {
        let commandBuffer = commandQueue!.makeCommandBuffer()
        
        
        let advectVelocity = commandBuffer!.makeComputeCommandEncoder()!
        var diffuse : Float32 = 0.99
        advectVelocity.setComputePipelineState(self.advectionPipeline)
        advectVelocity.setTexture(fluid.Velocity.Ping, index: 0)
        advectVelocity.setTexture(fluid.Velocity.Ping, index: 1)
        advectVelocity.setTexture(fluid.Velocity.Pong, index: 2)
        advectVelocity.setTexture(obstacleTex!, index: 3)
        advectVelocity.setBytes(&diffuse, length: MemoryLayout<Float>.size, index: 3)
        advectVelocity.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        advectVelocity.label = "Advect Velocity"
        advectVelocity.endEncoding()
        
        Swap(surface: &fluid.Velocity)

        let advectTemperature = commandBuffer!.makeComputeCommandEncoder()!
        diffuse = 0.90
        advectTemperature.setComputePipelineState(self.advectionPipeline)
        advectTemperature.setTexture(fluid.Velocity.Ping, index: 0)
        advectTemperature.setTexture(fluid.Temperature.Ping, index: 1)
        advectTemperature.setTexture(fluid.Temperature.Pong, index: 2)
        advectTemperature.setTexture(obstacleTex!, index: 3)
        advectTemperature.setBytes(&diffuse, length: MemoryLayout<Float>.size, index: 3)
        advectTemperature.label = "Advect Temperature"
        advectTemperature.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        advectTemperature.endEncoding()
        
        Swap(surface: &fluid.Temperature)

        let advectDensity = commandBuffer!.makeComputeCommandEncoder()!
        diffuse = 0.9
        advectDensity.setComputePipelineState(self.advectionPipeline)
        advectDensity.setTexture(fluid.Velocity.Ping, index: 0)
        advectDensity.setTexture(fluid.Density.Ping, index: 1)
        advectDensity.setTexture(fluid.Density.Pong, index: 2)
        advectDensity.setTexture(obstacleTex!, index: 3)
        advectDensity.setBytes(&diffuse, length: MemoryLayout<Float>.size, index: 3)
        advectDensity.label = "Advect Density"
        advectDensity.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        advectDensity.endEncoding()
        
        Swap(surface: &fluid.Density)
        
        let buoyancyEncoder = commandBuffer!.makeComputeCommandEncoder()!
        buoyancyEncoder.setComputePipelineState(self.buoyancyPipeline)
        buoyancyEncoder.setTexture(fluid.Velocity.Ping, index: 0)
        buoyancyEncoder.setTexture(fluid.Density.Ping, index: 1)
        buoyancyEncoder.setTexture(fluid.Temperature.Ping, index: 2)
        buoyancyEncoder.setTexture(fluid.Velocity.Pong, index: 3)
        buoyancyEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        buoyancyEncoder.label = "Buoyancy"
        buoyancyEncoder.endEncoding()
        
        Swap(surface: &fluid.Velocity)
        
        let encoder2 = commandBuffer!.makeComputeCommandEncoder()!
        encoder2.setComputePipelineState(self.impulsePipeline)
        encoder2.setTexture(fluid.Temperature.Pong, index: 0)
        encoder2.setTexture(fluid.Density.Pong, index: 1)
        encoder2.setBytes(&fluid.impulseParams, length:MemoryLayout<Fluid.ImpulseParams>.stride, index: 0)
        encoder2.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder2.label = "Impulse"
        encoder2.endEncoding()
        
        Swap(surface: &fluid.Temperature)
        Swap(surface: &fluid.Density)
        
        let divEncoder = commandBuffer!.makeComputeCommandEncoder()!
        divEncoder.setComputePipelineState(self.divergencePipeline)
        divEncoder.setTexture(fluid.Velocity.Ping, index: 0)
        divEncoder.setTexture(fluid.Divergence.Pong, index: 1)
        divEncoder.setTexture(fluid.Pressure.Pong, index: 2)    //we don't actually zero out pressure anymore
        divEncoder.setTexture(obstacleTex!, index: 3)
        divEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        divEncoder.label = "Divergence"
        divEncoder.endEncoding()
        
        Swap(surface: &fluid.Divergence)
        
        for _ in 0..<80
        {
            let _c = commandBuffer!.makeComputeCommandEncoder()!
            _c.setComputePipelineState(self.jacobiPipeline)
            //_c.setBytes(&fluid.jacobiParams, length: MemoryLayout<Fluid.JacobiParams>.stride, index: 0)
            _c.setTexture(fluid.Pressure.Ping, index: 0)
            _c.setTexture(fluid.Pressure.Pong, index: 1)
            _c.setTexture(fluid.Divergence.Ping, index: 2)
            _c.setTexture(obstacleTex!, index: 3)
            _c.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
            _c.label = "Jacobi"
            _c.endEncoding()
            Swap(surface: &fluid.Pressure)
            
        }
        
        let encoder3 = commandBuffer!.makeComputeCommandEncoder()!
        encoder3.setComputePipelineState(self.poissonPipeline)
        encoder3.setTexture(fluid.Velocity.Ping, index: 0)
        encoder3.setTexture(fluid.Velocity.Pong, index: 1)
        encoder3.setTexture(fluid.Pressure.Ping, index: 2)
        encoder3.setTexture(obstacleTex!, index: 3)
        encoder3.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        encoder3.label = "Poisson Correction"
        encoder3.endEncoding()
        
        Swap(surface: &fluid.Velocity)
        
        let chainEncoder = commandBuffer!.makeComputeCommandEncoder()!
        chainEncoder.setComputePipelineState(self.constituteObstaclePipeline)
        chainEncoder.setTexture(fluid.Velocity.Ping, index: 0)
        chainEncoder.setTexture(obstacleTex!, index: 1)
        chainEncoder.setTexture(chainOutput!, index: 2)
        chainEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: MTLSize(width: threadsPerGroup.width * 2, height: threadsPerGroup.height, depth: threadsPerGroup.depth))
        chainEncoder.endEncoding()

        //advect
        //blit composite & velocity textures
        //apply impulse
        //blit composite texture
        //
        
        //encoder.endEncoding()
        commandBuffer!.commit()
        return chainOutput
    }
    
}


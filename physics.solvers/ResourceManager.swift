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
    
    struct GridSurface {
        var Full : Surface
        var Half : Surface
        var Quarter : Surface
        var Eighth : Surface
        var Sixteenth : Surface
    }
    
    var chain : MTLTexture?
    
    var Velocity : Surface
    var Temperature : Surface
    var Density : Surface
    var Divergence : Surface
    var Pressure : Surface
    var ResidualGrid : GridSurface
    var Residual : MTLTexture
    
    
    var advectionParams : AdvectionParams
    var impulseParams : ImpulseParams
    var jacobiParams : JacobiParams
    
    //pressure, temperature, density, divergence
    
    func LabelSurfaces()
    {
        self.ResidualGrid.Full.Ping.label = "Full Res Residual Ping"
        self.ResidualGrid.Half.Ping.label = "1/2 Res Residual Ping"
        self.ResidualGrid.Quarter.Ping.label = "1/4 Res Residual Ping"
        self.ResidualGrid.Eighth.Ping.label = "1/8 Res Residual Ping"
        self.ResidualGrid.Sixteenth.Ping.label = "1/16 Res Residual Ping"
        self.ResidualGrid.Full.Pong.label = "Full Res Residual Pong"
        self.ResidualGrid.Half.Pong.label = "1/2 Res Residual Pong"
        self.ResidualGrid.Quarter.Pong.label = "1/4 Res Residual Pong"
        self.ResidualGrid.Eighth.Pong.label = "1/8 Res Residual Pong"
        self.ResidualGrid.Sixteenth.Pong.label = "1/16 Res Residual Pong"
        self.Residual.label = "Residual"
        
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
    }
    
    init(device: MTLDevice)
    {
        let velocityDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float, width: 512, height: 512, mipmapped: false)
        let swapDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 1024, height: 1024, mipmapped: true)
        let singleCDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 512, height: 512, mipmapped: false)
        
        let gridHalfDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 256, height: 256, mipmapped: false)
        let gridQuarterDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 128, height: 128, mipmapped: false)
        let gridEighthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 64, height: 64, mipmapped: false)
        let gridSixteenthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 32, height: 32, mipmapped: false)
        
        gridHalfDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        gridQuarterDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        gridEighthDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        gridSixteenthDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        
        velocityDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        velocityDesc.allowGPUOptimizedContents = true
        velocityDesc.compressionType = .lossless

        swapDesc.usage = MTLTextureUsage([.shaderWrite, .shaderRead])
        swapDesc.mipmapLevelCount = 2
        
        singleCDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        singleCDesc.allowGPUOptimizedContents = true
        singleCDesc.compressionType = .lossless
        
        let fullSurface = Surface(Ping: device.makeTexture(descriptor: singleCDesc)!, Pong: device.makeTexture(descriptor : singleCDesc)!)
        let halfSurface = Surface(Ping: device.makeTexture(descriptor: gridHalfDesc)!, Pong: device.makeTexture(descriptor: gridHalfDesc)!)
        let quarterSurface = Surface(Ping: device.makeTexture(descriptor: gridQuarterDesc)!, Pong: device.makeTexture(descriptor: gridQuarterDesc)!)
        let eighthSurface = Surface(Ping: device.makeTexture(descriptor: gridEighthDesc)!, Pong: device.makeTexture(descriptor: gridEighthDesc)!)
        let sixteenthSurface = Surface(Ping: device.makeTexture(descriptor: gridSixteenthDesc)!, Pong: device.makeTexture(descriptor: gridSixteenthDesc)!)
        
        self.ResidualGrid = GridSurface(Full: fullSurface, Half: halfSurface, Quarter: quarterSurface, Eighth: eighthSurface, Sixteenth: sixteenthSurface)
        self.Velocity = Surface(Ping: device.makeTexture(descriptor: velocityDesc)!, Pong: device.makeTexture(descriptor: velocityDesc)!)
        self.Pressure = Surface(Ping: device.makeTexture(descriptor: singleCDesc)!, Pong: device.makeTexture(descriptor: singleCDesc)!)
        self.Divergence = Surface(Ping: device.makeTexture(descriptor: singleCDesc)!, Pong: device.makeTexture(descriptor: singleCDesc)!)
        self.Temperature = Surface(Ping: device.makeTexture(descriptor: singleCDesc)!, Pong: device.makeTexture(descriptor: singleCDesc)!)
        self.Density = Surface(Ping: device.makeTexture(descriptor: singleCDesc)!, Pong: device.makeTexture(descriptor: singleCDesc)!)
        self.Residual = device.makeTexture(descriptor: singleCDesc)!
        self.chain = device.makeTexture(descriptor: swapDesc)
    
        
        self.advectionParams = AdvectionParams(uDissipation: 0.99999, tDissipation: 0.99, dDissipation: 0.9999)
        self.impulseParams = ImpulseParams(origin: SIMD2<Float>(0.5, 0), radius: 0.1, iTemperature: 10, iDensity: 1, iAuxillary: 0)
        self.jacobiParams = JacobiParams(Alpha: -1.0, InvBeta: 0.25)
        
        LabelSurfaces()
        
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
    var gsPipeline : MTLComputePipelineState
    var impulsePipeline : MTLComputePipelineState
    var divergencePipeline : MTLComputePipelineState
    var residualPipeline : MTLComputePipelineState
    var restrictionPipeline : MTLComputePipelineState
    var prolongationPipeline : MTLComputePipelineState
    var constitutionPipeline : MTLComputePipelineState
    var constituteObstaclePipeline : MTLComputePipelineState
    var fluid : Fluid
    var obstacles : MTLTexture?
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
        let gs = library.makeFunction(name: "GaussSeidel")
        let poisson = library.makeFunction(name: "PoissonCorrection")
        let divergence = library.makeFunction(name: "Divergence")
        let residual = library.makeFunction(name: "Residual")
        let constitution = library.makeFunction(name: "Constitution")
        let constituteObstacle = library.makeFunction(name: "ConstituteObstacle")
        let restrict = library.makeFunction(name: "Restrict")
        let prolongate = library.makeFunction(name: "Prolongate")
        
        self.jacobiPipeline = try library.device.makeComputePipelineState(function: jacobi!)
        self.gsPipeline = try library.device.makeComputePipelineState(function: gs!)
        self.advectionPipeline = try library.device.makeComputePipelineState(function: advection!)
        self.poissonPipeline = try library.device.makeComputePipelineState(function: poisson!)
        self.impulsePipeline = try library.device.makeComputePipelineState(function: impulse!)
        self.divergencePipeline = try library.device.makeComputePipelineState(function: divergence!)
        self.residualPipeline = try library.device.makeComputePipelineState(function: residual!)
        self.constitutionPipeline = try library.device.makeComputePipelineState(function: constitution!)
        self.buoyancyPipeline = try library.device.makeComputePipelineState(function: buoyancy!)
        self.constituteObstaclePipeline = try library.device.makeComputePipelineState(function: constituteObstacle!)
        self.restrictionPipeline = try library.device.makeComputePipelineState(function: restrict!)
        self.prolongationPipeline = try library.device.makeComputePipelineState(function: prolongate!)
        
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
    
    ///[Full, Half, Quarter, Eighth, Sixteenth]
    func MGRestrictLevel(cmdBuffer : MTLCommandBuffer, inSurface : inout Fluid.Surface, outSurface : inout Fluid.Surface)
    {
        //apply weighted jacobi or red black gauss seidel
        //calculate residual
        //restrict residual
        //apply (several iterations) jacobi or rgbs
        //(calculate the residual?) at some point are we calculating the residual of residual?? - yes!
        var threadsSize = MTLSize(width: inSurface.Ping.width/32, height: inSurface.Ping.height/32, depth: 1)
        //smooth
        for i in 0...4
        {
            let jacobiEncoder = cmdBuffer.makeComputeCommandEncoder()!
            jacobiEncoder.setComputePipelineState(self.jacobiPipeline)
            jacobiEncoder.label = "Weighted Jacobi \(i)"
            jacobiEncoder.setTexture(inSurface.Ping, index: 0)
            jacobiEncoder.setTexture(inSurface.Pong, index: 1)
            jacobiEncoder.setTexture(fluid.Divergence.Ping, index: 2)
            jacobiEncoder.setTexture(obstacles!, index: 3)
            jacobiEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsSize)
            jacobiEncoder.endEncoding()
            Swap(surface: &inSurface)
        }
        
        //compute residual
        //x is stored in Ping
        //residual is stored in Pong
        let residualEncoder = cmdBuffer.makeComputeCommandEncoder()!
        residualEncoder.setComputePipelineState(self.residualPipeline)
        residualEncoder.label = "Compute Residual"
        residualEncoder.setTexture(inSurface.Ping, index: 0)
        residualEncoder.setTexture(fluid.Divergence.Ping, index: 1)
        residualEncoder.setTexture(inSurface.Pong, index: 2)
        residualEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsSize)
        residualEncoder.endEncoding()
        
        threadsSize = MTLSize(width: outSurface.Ping.width/32, height: outSurface.Ping.height/32, depth: 1)
        
        let restrictEncoder = cmdBuffer.makeComputeCommandEncoder()!
        restrictEncoder.setComputePipelineState(self.restrictionPipeline)
        restrictEncoder.label = "Restrict"
        restrictEncoder.setTexture(inSurface.Pong, index: 0)
        restrictEncoder.setTexture(outSurface.Ping, index: 1)
        restrictEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsSize)
        restrictEncoder.endEncoding()
        //the result of all iterations is our error
    }
    
    //apply weighted jacobi on restriction
    //prolong the residual to a higher level
    //add the prolonged residual to the
    //new level's existing solution
    //repeat
    
    //Ping has the solution
    //Pong has the solution's residual
    func MGProlongateLevel(cmdBuffer : MTLCommandBuffer, inSurface : inout Fluid.Surface, outSurface : inout Fluid.Surface)
    {
        var threadsSize = MTLSize(width: inSurface.Ping.width/32, height: inSurface.Ping.height/32, depth: 1)
        //iterate on prior level's residual
        let smoothEncoder = cmdBuffer.makeComputeCommandEncoder()!
        smoothEncoder.setComputePipelineState(self.jacobiPipeline)
        smoothEncoder.label = "Prolongation Smoothing"
        //we prolong the previous level's restricted residual
        smoothEncoder.setTexture(inSurface.Ping, index: 0)
        smoothEncoder.setTexture(inSurface.Pong, index: 1)
        smoothEncoder.setTexture(fluid.Divergence.Ping, index: 2)
        smoothEncoder.setTexture(obstacles!, index: 3)
        smoothEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsSize)
        smoothEncoder.endEncoding()
        
        threadsSize = MTLSize(width: outSurface.Ping.width/32, height: outSurface.Ping.height/32, depth: 1)
        let prolongateEncoder = cmdBuffer.makeComputeCommandEncoder()!
        prolongateEncoder.setComputePipelineState(self.prolongationPipeline)
        prolongateEncoder.label = "Prolongation"
        prolongateEncoder.setTexture(inSurface.Pong, index: 0)
        prolongateEncoder.setTexture(outSurface.Ping, index: 1) //level up current solution
        prolongateEncoder.setTexture(outSurface.Pong, index: 2) //the output of this is
        prolongateEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsSize)
        prolongateEncoder.endEncoding()
    }
    
    func Simulate(obstacleTex : MTLTexture? = nil, chainOutput : MTLTexture? = nil) -> MTLTexture?
    {
        let commandBuffer = commandQueue!.makeCommandBuffer()
        obstacles = obstacleTex
        
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
        
        /*
         Pre smooth - 1-3 jacobi
         Residual
         Project
         Restrict
         Repeat for each restriction
         try to approximate the coarsest level (maybe 10+ iterations)
         
        
         
         
         */
         
        for _ in 0..<40
        {
            let _c = commandBuffer!.makeComputeCommandEncoder()!
            _c.setComputePipelineState(self.jacobiPipeline)
            //_c.setBytes(&fluid.jacobiParams, length: MemoryLayout<Fluid.JacobiParams>.stride, index: 0)
            _c.setTexture(fluid.Pressure.Ping, index: 0)
            _c.setTexture(fluid.Pressure.Pong, index: 1)
            _c.setTexture(fluid.Divergence.Ping, index: 2)
            _c.setTexture(fluid.Residual, index: 4)
            _c.setTexture(obstacleTex!, index: 3)
            _c.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
            _c.label = "Pre-Smoothing"
            _c.endEncoding()
            Swap(surface: &fluid.Pressure)
            
        }
        
        let residualEncoder = commandBuffer!.makeComputeCommandEncoder()!
        residualEncoder.setComputePipelineState(self.residualPipeline)
        residualEncoder.label = "Compute Pressure Residual"
        residualEncoder.setTexture(fluid.ResidualGrid.Full.Ping, index: 2)
        residualEncoder.setTexture(fluid.Divergence.Ping, index: 1)
        residualEncoder.setTexture(fluid.Pressure.Ping, index: 0)
        residualEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
        residualEncoder.endEncoding()
        
        var gridLevels = [fluid.ResidualGrid.Full, fluid.ResidualGrid.Half, fluid.ResidualGrid.Quarter, fluid.ResidualGrid.Eighth, fluid.ResidualGrid.Sixteenth]
        let levelsSize = gridLevels.count - 1
        
        for i in 0..<levelsSize
        {
            var inSurf = gridLevels[i]
            var outSurf = gridLevels[i+1]
            MGRestrictLevel(cmdBuffer: commandBuffer!, inSurface: &inSurf, outSurface: &outSurf)
            gridLevels[i] = inSurf
            gridLevels[i+1] = outSurf
        }
        
        for i in 0..<levelsSize
        {
            var inSurf = gridLevels[levelsSize - i]
            var outSurf = gridLevels[levelsSize - i-1]
            MGProlongateLevel(cmdBuffer: commandBuffer!, inSurface: &inSurf, outSurface: &outSurf)
            gridLevels[levelsSize - i] = inSurf
            gridLevels[levelsSize - i-1] = outSurf
        }
        /*
        var redBlack : Int = 0
        for _ in 0..<40
        {
            let _c = commandBuffer!.makeComputeCommandEncoder()!
            _c.setComputePipelineState(self.gsPipeline)
            _c.label = "Red Gauss Seidel"
            redBlack = 0
            _c.setBytes(&redBlack, length: MemoryLayout<Int>.size, index: 0)
            _c.setTexture(fluid.Pressure.Ping, index: 0)
            _c.setTexture(fluid.Pressure.Pong, index: 1)
            _c.setTexture(fluid.Divergence.Ping, index: 2)
            _c.setTexture(obstacleTex!, index: 3)
            _c.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
            Swap(surface: &fluid.Pressure)
            ///this is properly swapping the textures
            ///check to make sure this is being done serially - it does!

            
            _c.label = "Black Gauss Seidel"
            redBlack = 1
            _c.setBytes(&redBlack, length: MemoryLayout<Int>.size, index: 0)
            _c.setTexture(fluid.Pressure.Ping, index: 0)
            _c.setTexture(fluid.Pressure.Pong, index: 1)
            _c.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadsPerGroup)
            Swap(surface: &fluid.Pressure)
            _c.endEncoding()
            
        }
        */
        
        
        /*
        let restrictions = [fluid.Pressure.Pong, fluid.ResidualGrid.Half, fluid.ResidualGrid.Quarter, fluid.ResidualGrid.Eighth, fluid.ResidualGrid.Sixteenth]
        
        for i in 1..<restrictions.count
        {
            let threads = 512 / (2 * i);
            let threadSize = MTLSize(width: threads/32, height: threads/32, depth: 1)
            let restrictEncoder = commandBuffer!.makeComputeCommandEncoder()!
            MultigridCycle(cmdBuffer: commandBuffer, )
            restrictEncoder.setComputePipelineState(self.restrictionPipeline)
            restrictEncoder.setTexture(restrictions[i - 1], index: 0)
            restrictEncoder.setTexture(restrictions[i], index: 1)
            restrictEncoder.dispatchThreadgroups(groupSize, threadsPerThreadgroup: threadSize)
            restrictEncoder.endEncoding()
        }
        
        */
        let encoder3 = commandBuffer!.makeComputeCommandEncoder()!
        encoder3.setComputePipelineState(self.poissonPipeline)
        encoder3.setTexture(fluid.Velocity.Ping, index: 0)
        encoder3.setTexture(fluid.Velocity.Pong, index: 1)
        encoder3.setTexture(fluid.ResidualGrid.Full.Pong, index: 2)
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


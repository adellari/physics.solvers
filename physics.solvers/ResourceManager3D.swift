//
//  ReousrceManager3D.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 10/13/24.
//

import Metal
import MetalKit
import simd

class Fluid3D {
    
    struct Surface {
        var Ping : MTLTexture
        var Pong : MTLTexture
    }
    
    struct Params {
        let tempImpulse : Float = 10
        let reactImpulse : Float = 1
        let densityImpulse : Float = 1
        let reactDecay : Float = 0.001
        let vorticityStrength : Float = 1
        let origin : SIMD3<Float> = SIMD3<Float>(0, 0.1, 0)
        let radius : Float = 0.5
    }
    
    var Velocity : Surface
    var Temperature : Surface
    var Pressure : Surface
    //var Divergence : Surface
    var Density : Surface
    var Reaction : Surface  //keep track of fire reaction lifetime 
    var Temporary : MTLTexture
    var Obstacles : MTLTexture?
    
    
    init(device : MTLDevice, size : MTLSize)
    {
        let singleChannel = MTLTextureDescriptor()
        singleChannel.pixelFormat = .r16Float
        singleChannel.textureType = .type3D
        singleChannel.width = size.width; singleChannel.height = size.height; singleChannel.depth = size.depth
        singleChannel.usage = MTLTextureUsage([.shaderWrite, .shaderRead])
        
        let fourChannel = MTLTextureDescriptor()
        fourChannel.pixelFormat = .rgba16Float;
        fourChannel.textureType = .type3D
        fourChannel.width = size.width; singleChannel.height = size.height; singleChannel.depth = size.depth
        fourChannel.usage = MTLTextureUsage([.shaderWrite, .shaderRead])
        
        
        Velocity = Surface(Ping: device.makeTexture(descriptor: fourChannel)!, Pong: device.makeTexture(descriptor: fourChannel)!)
        //Divergence = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        Temperature = Surface(Ping: device.makeTexture(descriptor: singleChannel)!, Pong: device.makeTexture(descriptor: singleChannel)!)
        Density = Surface(Ping: device.makeTexture(descriptor: singleChannel)!, Pong: device.makeTexture(descriptor: singleChannel)!)
        Pressure = Surface(Ping: device.makeTexture(descriptor: singleChannel)!, Pong: device.makeTexture(descriptor: singleChannel)!)
        Reaction = Surface(Ping: device.makeTexture(descriptor: singleChannel)!, Pong: device.makeTexture(descriptor: singleChannel)!)
        Temporary = device.makeTexture(descriptor: fourChannel)!
        //Obstacles = device.makeTexture(descriptor: singleChannelW)!
        Velocity.Ping.label = "Velocity Ping"
        self.Velocity.Pong.label = "Velocity Pong"
        
        self.Temperature.Ping.label = "Temperature Ping"
        self.Temperature.Pong.label = "Temperature Pong"
        
        self.Density.Ping.label = "Density Ping"
        self.Density.Pong.label = "Density Pong"
        
        self.Pressure.Ping.label = "Pressure Ping"
        self.Pressure.Pong.label = "Pressure Pong"
        
        //self.Divergence.Ping.label = "Divergence Read"
        //self.Divergence.Pong.label = "Divergence Write"
        
        self.Reaction.Ping.label = "Reaction Ping"
        self.Reaction.Pong.label = "Reaction Pong"
        
        self.Temporary.label = "Temporary 4-Channel"
    }
}

struct Kernels
{
    var Advection : MTLComputePipelineState
    var Buoyancy : MTLComputePipelineState
    var Impulse : MTLComputePipelineState
    var eImpulse : MTLComputePipelineState
    var Vorticity : MTLComputePipelineState
    var Confinement : MTLComputePipelineState
    var Divergence : MTLComputePipelineState
    var Jacobi : MTLComputePipelineState
    var Poisson : MTLComputePipelineState
}

class ResourceManager3D
{
    
    var device : MTLDevice
    var commandQueue : MTLCommandQueue
    
    var kernels : Kernels
    var fluid : Fluid3D
    
    init (_device: MTLDevice, dims: MTLSize) throws
    {
        self.device = _device
        self.commandQueue = device.makeCommandQueue()!
        
        var lib = try device.makeDefaultLibrary(bundle: .main)
        var adv3 = lib.makeFunction(name: "Advection3")!
        var imp3 = lib.makeFunction(name: "Impulse3")!
        var buoy3 = lib.makeFunction(name: "Buoyancy3")!
        var eimp3 = lib.makeFunction(name: "EImpulse3")!
        var vort3 = lib.makeFunction(name: "Vorticity3")!
        var conf3 = lib.makeFunction(name: "Confinement3")!
        var div3 = lib.makeFunction(name: "Divergence3")!
        var jac3 = lib.makeFunction(name: "Jacobi3")!
        var pois3 = lib.makeFunction(name: "Poisson3")!
        
        var Adv3 = try device.makeComputePipelineState(function: adv3)
        var Imp3 = try device.makeComputePipelineState(function: imp3)
        var Buoy3 = try device.makeComputePipelineState(function: buoy3)
        var eImp3 = try device.makeComputePipelineState(function: eimp3)
        var Vort3 = try device.makeComputePipelineState(function: vort3)
        var Conf3 = try device.makeComputePipelineState(function: conf3)
        var Div3 = try device.makeComputePipelineState(function: div3)
        var Jac3 = try device.makeComputePipelineState(function: jac3)
        var Pois3 = try device.makeComputePipelineState(function: pois3)
        
        self.kernels = Kernels(Advection: Adv3, Buoyancy: Buoy3, Impulse: Imp3, eImpulse: eImp3, Vorticity: Vort3, Confinement: Conf3, Divergence: Div3, Jacobi: Jac3, Poisson: Pois3)
        
        self.fluid = Fluid3D(device: self.device, size: dims)
    }
    
    public func Simulate()
    {
        
    }
    
    //temperature advection
    
    //density advection
    
    //reaction advection
    
    //velocity advection
    
    //buoyancy
    
    //reaction impulse
    
    //temperature impulse
    
    //extinguishment density impulse
    
    //vorticity confinement
    
    //divergence
    
    //[shouldn't we zero out the pressure to respect the poisson eq]
    //[We in fact should NOT zero out pressure]
    
    //jacobi iteration relaxation on pressure
    
}

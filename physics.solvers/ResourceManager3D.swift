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
    var Temporary3f : MTLTexture
    var Obstacles : MTLTexture?
    
    
    init(device : MTLDevice, size : MTLSize)
    {
        let singleChannelW = MTLTextureDescriptor()
        singleChannelW.pixelFormat = .r16Float
        singleChannelW.textureType = .type3D
        singleChannelW.width = size.width; singleChannelW.height = size.height; singleChannelW.depth = size.depth
        singleChannelW.usage = MTLTextureUsage([.shaderWrite])
        
        let fourChannelW = MTLTextureDescriptor()
        fourChannelW.pixelFormat = .rgba16Float;
        fourChannelW.textureType = .type3D
        fourChannelW.width = size.width; singleChannelW.height = size.height; singleChannelW.depth = size.depth
        fourChannelW.usage = MTLTextureUsage([.shaderWrite])
        
        let singleChannelR = MTLTextureDescriptor()
        singleChannelR.pixelFormat = .r16Float
        singleChannelR.textureType = .type3D
        singleChannelR.width = size.width; singleChannelW.height = size.height; singleChannelW.depth = size.depth
        singleChannelR.usage = MTLTextureUsage([.shaderRead])
        
        let fourChannelR = MTLTextureDescriptor()
        fourChannelR.pixelFormat = .rg16Float
        fourChannelR.textureType = .type3D
        fourChannelR.width = size.width; singleChannelW.height = size.height; singleChannelW.depth = size.depth
        fourChannelR.usage = MTLTextureUsage([.shaderWrite])
        
        let fourChannelRW = fourChannelW
        fourChannelRW.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        
        Velocity = Surface(Ping: device.makeTexture(descriptor: fourChannelR)!, Pong: device.makeTexture(descriptor: fourChannelW)!)
        //Divergence = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        Temperature = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        Density = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        Pressure = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        Reaction = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        Temporary3f = device.makeTexture(descriptor: fourChannelRW)!
        //Obstacles = device.makeTexture(descriptor: singleChannelW)!
        Velocity.Ping.label = "Velocity Read"
        self.Velocity.Pong.label = "Velocity Write"
        
        self.Temperature.Ping.label = "Temperature Read"
        self.Temperature.Pong.label = "Temperature Write"
        
        self.Density.Ping.label = "Density Read"
        self.Density.Pong.label = "Density Write"
        
        self.Pressure.Ping.label = "Pressure Read"
        self.Pressure.Pong.label = "Pressure Write"
        
        //self.Divergence.Ping.label = "Divergence Read"
        //self.Divergence.Pong.label = "Divergence Write"
        
        self.Reaction.Ping.label = "Reaction Read"
        self.Reaction.Pong.label = "Reaction Write"
        
        self.Temporary3f.label = "Temporary Velocity"
    }
}

extension ResourceManager {
    
    
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

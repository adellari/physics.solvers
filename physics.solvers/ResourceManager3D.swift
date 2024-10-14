//
//  ReousrceManager3D.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 10/13/24.
//

import Metal
import MetalKit

class Fluid3D {
    
    struct Surface {
        var Ping : MTLTexture
        var Pong : MTLTexture
    }
    
    var Velocity : Surface
    var Temperature : Surface
    var Pressure : Surface
    var Divergence : Surface
    var Density : Surface
    var Obstacles : MTLTexture?
    
    init(device : MTLDevice, size : MTLSize)
    {
        let singleChannelW = MTLTextureDescriptor()
        singleChannelW.pixelFormat = .r16Float
        singleChannelW.textureType = .type3D
        singleChannelW.width = size.width; singleChannelW.height = size.height; singleChannelW.depth = size.depth
        singleChannelW.usage = MTLTextureUsage([.shaderWrite])
        
        let doubleChannelW = MTLTextureDescriptor()
        doubleChannelW.pixelFormat = .rg16Float
        doubleChannelW.textureType = .type3D
        doubleChannelW.width = size.width; singleChannelW.height = size.height; singleChannelW.depth = size.depth
        doubleChannelW.usage = MTLTextureUsage([.shaderWrite])
        
        let singleChannelR = MTLTextureDescriptor()
        singleChannelR.pixelFormat = .r16Float
        singleChannelR.textureType = .type3D
        singleChannelR.width = size.width; singleChannelW.height = size.height; singleChannelW.depth = size.depth
        singleChannelR.usage = MTLTextureUsage([.shaderRead])
        
        let doubleChannelR = MTLTextureDescriptor()
        doubleChannelR.pixelFormat = .rg16Float
        doubleChannelR.textureType = .type3D
        doubleChannelR.width = size.width; singleChannelW.height = size.height; singleChannelW.depth = size.depth
        doubleChannelR.usage = MTLTextureUsage([.shaderWrite])
        
        Velocity = Surface(Ping: device.makeTexture(descriptor: doubleChannelR)!, Pong: device.makeTexture(descriptor: doubleChannelW)!)
        Divergence = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        Temperature = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        Density = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        Pressure = Surface(Ping: device.makeTexture(descriptor: singleChannelR)!, Pong: device.makeTexture(descriptor: singleChannelW)!)
        
        //Obstacles = device.makeTexture(descriptor: singleChannelW)!
        Velocity.Ping.label = "Velocity Read"
        self.Velocity.Pong.label = "Velocity Write"
        
        self.Temperature.Ping.label = "Temperature Read"
        self.Temperature.Pong.label = "Temperature Write"
        
        self.Density.Ping.label = "Density Read"
        self.Density.Pong.label = "Density Write"
        
        self.Pressure.Ping.label = "Pressure Read"
        self.Pressure.Pong.label = "Pressure Write"
        
        self.Divergence.Ping.label = "Divergence Read"
        self.Divergence.Pong.label = "Divergence Write"
    }
}

extension ResourceManager {
    
    
    
}

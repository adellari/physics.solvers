//
//  Presenter.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 10/18/24.
//

import MetalKit
import SwiftUI
import Metal

struct Viewport : NSViewRepresentable {
    var renderer : Renderer?
    var sdf : MeshSDF?
    var simulator : ResourceManager3D?
    var simulator2d : ResourceManager2D
    
    class Coordinator : NSObject, MTKViewDelegate {
        var renderer : Renderer?
        var sdf : MeshSDF?
        var simulator : ResourceManager3D?
        var simulator2d : ResourceManager2D
        var viewport : Viewport
        var device : MTLDevice
        var commandQueue : MTLCommandQueue
        var chain : (Ping : MTLTexture, Pong : MTLTexture)?
        var framesGenerated = 0
        
        init(viewport : Viewport, simulator : ResourceManager3D?, simulator2d : ResourceManager2D, renderer : Renderer?, sdf: MeshSDF?)
        {
            self.viewport = viewport
            self.simulator = simulator
            self.simulator2d = simulator2d
            self.device = simulator2d.device!
            self.renderer = renderer
            self.commandQueue = simulator2d.commandQueue!
            self.sdf = sdf
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            
        }
        
        func CreateChain(format : MTLPixelFormat, size: MTLSize)
        {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: size.width, height: size.height, mipmapped: false)
            descriptor.usage = .shaderWrite
            self.chain = (device.makeTexture(descriptor: descriptor)!, device.makeTexture(descriptor: descriptor)!)
        }
        
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable else { return }
            sdf?.Voxelize()
            let simulation = simulator2d.Simulate(obstacleTex : sdf?.sliceTex, chainOutput: chain?.Ping)
            //let render = renderer?.Draw(chain: self.chain)
            //print("swapchain size: \(drawable.texture.width) x \(drawable.texture.height)")
            
            if (simulation != nil)
            {
                let cmd = commandQueue.makeCommandBuffer()!
                let blitEncoder = cmd.makeBlitCommandEncoder()!
                blitEncoder.copy(from: simulation!, to: drawable.texture)
                blitEncoder.endEncoding()
                cmd.commit()
            }
            /*
            if ( framesGenerated == 5)
            {
                let captureManager = MTLCaptureManager.shared()
                let capDesc = MTLCaptureDescriptor()
                capDesc.captureObject = simulator2d.device
                
                
                do {
                    try captureManager.startCapture(with: capDesc)
                }
                catch {
                    fatalError("Failed to start Metal capture: \(error)")
                }
            }
            */
             
            drawable.present()
            /*
            if (framesGenerated == 6)
            {
                MTLCaptureManager.shared().stopCapture()
            }
           */
            //blit the result of the renderer to chainTex
            framesGenerated += 1
        }
    }
    
    func makeNSView(context: Context) -> MTKView
    {
        let view = MTKView()
        let dim = MTLSize(width: 1024, height: 512, depth: 1)
        view.device = simulator2d.device
        view.delegate = context.coordinator
        view.framebufferOnly = false
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.frame = CGRect(x: 0, y: 0, width: dim.width, height: dim.height)
        context.coordinator.CreateChain(format: view.colorPixelFormat, size: dim)
        
        
        
        return view
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(viewport: self, simulator: simulator, simulator2d: simulator2d, renderer: renderer, sdf: sdf)
    }
}

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
        
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable else { return }
            sdf?.Voxelize(sharedBuffer: nil)
            simulator2d.Simulate()
            let render = renderer?.Draw()
            //print("swapchain size: \(drawable.texture.width) x \(drawable.texture.height)")
            
            if (render != nil)
            {
                let cmd = commandQueue.makeCommandBuffer()!
                let blitEncoder = cmd.makeBlitCommandEncoder()!
                blitEncoder.copy(from: render!, to: drawable.texture)
                blitEncoder.endEncoding()
                cmd.commit()
            }
            
             
            drawable.present()
            
            //MTLCaptureManager.shared().stopCapture()
            //blit the result of the renderer to chainTex
        }
    }
    
    func makeNSView(context: Context) -> MTKView
    {
        let view = MTKView()
        view.device = simulator2d.device
        view.delegate = context.coordinator
        view.framebufferOnly = false
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.frame = CGRect(x: 0, y: 0, width: 1024, height: 512)
        renderer!.CreateChain(format: view.colorPixelFormat)
        /*
        let captureManager = MTLCaptureManager.shared()
        let capDesc = MTLCaptureDescriptor()
        capDesc.captureObject = simulator2d.device
        
        do {
            try captureManager.startCapture(with: capDesc)
        }
        catch {
            fatalError("Failed to start Metal capture: \(error)")
        }
        */
        return view
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(viewport: self, simulator: simulator, simulator2d: simulator2d, renderer: renderer, sdf: sdf)
    }
}

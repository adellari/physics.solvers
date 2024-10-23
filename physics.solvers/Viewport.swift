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
    var simulator : ResourceManager3D
    
    class Coordinator : NSObject, MTKViewDelegate {
        var renderer : Renderer?
        var simulator : ResourceManager3D
        var viewport : Viewport
        var device : MTLDevice
        var commandQueue : MTLCommandQueue
        
        init(viewport : Viewport, simulator : ResourceManager3D, renderer : Renderer?)
        {
            self.viewport = viewport
            self.simulator = simulator
            self.device = simulator.device
            self.renderer = renderer
            self.commandQueue = simulator.commandQueue
        }
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            
        }
        
        func draw(in view: MTKView) {
            guard let drawable = view.currentDrawable else { return }
            
            simulator.Simulate()
            renderer?.Draw()
            
            var chainTex = drawable.texture
            //blit the result of the renderer to chainTex
        }
    }
    
    func makeNSView(context: Context) -> MTKView
    {
        var view = MTKView()
        view.device = simulator.device
        view.delegate = context.coordinator
        view.framebufferOnly = false
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.frame = CGRect(x: 0, y: 0, width: 512, height: 512)
        
        return view
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(viewport: self, simulator: simulator, renderer: renderer)
    }
}

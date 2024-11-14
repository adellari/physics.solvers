//
//  ContentView.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 9/22/24.
//

import SwiftUI

struct ContentView: View {
    var Simulation : ResourceManager2D?
    var counter : Int = 0
    var viewport : Viewport
    @State var horizontalValue : Double = 0.0
    @State var verticalValue : Double = 2.0
    
    var body: some View {
        VStack {
            Text("Hello, world!")
            //guess the frame has to be 1/2 of the actual drawable
            //at least on the laptop screen
            viewport
                .frame(width: 512, height: 256)
            //Swift frames and Metal textures are 0,0 in the top left
            //so scrolling down, to the right gives positive velocity
            //as a result, we negate y velocity due to the nature of our
            //view matrix elevation, theta basis
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            viewport.renderer!.azimuth += value.velocity.width * 0.001
                            //print("\(value.velocity.height * 0.01)")
                            viewport.renderer!.elevation += value.velocity.height * -0.001
                        }
                )
            
            Slider(value: $horizontalValue, in: 0...64, onEditingChanged: { _ in
                viewport.sdf!.sliceIdx = Int(horizontalValue)
            })
        }
        .onAppear()
        {
            print(" we've served the view")
        }
        .padding()
    }
    
    init()
    {
        do
        {
            guard let device = MTLCreateSystemDefaultDevice() else{
                fatalError("failed to create metal device")
            }
            Simulation = try ResourceManager2D(_device: device)
            let sdf = try MeshSDF(_device: device, sharedQueue: Simulation!.commandQueue!)
            let renderer = try Renderer(queue: Simulation!.commandQueue!)
            let viewport = Viewport(renderer: renderer, sdf: sdf, simulator2d: Simulation!)
            self.viewport = viewport
            //Simulation!.Draw()
            
        }
        catch 
        {
            fatalError("failed to create viewport components \(error)")
        }
    }
}

#Preview {
    ContentView()
}

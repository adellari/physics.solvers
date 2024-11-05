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
    
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
            //guess the frame has to be 1/2 of the actual drawable
            //at least on the laptop screen
            viewport
                .frame(width: 512, height: 256)
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
            let renderer = try Renderer(queue: Simulation!.commandQueue!)
            let viewport = Viewport(renderer: renderer, simulator2d: Simulation!)
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

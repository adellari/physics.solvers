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
            viewport
            //MetalViewRepresentable(metalView: Simulation!.metalView)
                .frame(width: 512, height: 512)
            /*
            Button(action:
            {
                //Simulation!.Draw()
                
                Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { _ in
                    Simulation!.Draw()
                    //counter = counter.advanced(by: 1)
                    
                }
                
            })
            {
                Image(systemName: "eye")
            }
            */
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

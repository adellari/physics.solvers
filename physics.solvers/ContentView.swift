//
//  ContentView.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 9/22/24.
//

import SwiftUI

struct ContentView: View {
    var Simulation : ResourceManager?
    var counter : Int = 0
    
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
            MetalViewRepresentable(metalView: Simulation!.metalView)
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
            Simulation = try ResourceManager(_device: device)
            //Simulation!.Draw()
            
        }
        catch 
        {
            
        }
    }
}

#Preview {
    ContentView()
}

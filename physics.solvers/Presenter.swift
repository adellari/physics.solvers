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
    var renderer : Renderer
    var simulator : ResourceManager3D
}

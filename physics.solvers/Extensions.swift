//
//  Extensions.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 9/30/24.
//

import Metal
import AppKit
import SwiftUI

extension NSImage {
    convenience init?(mtlTexture: MTLTexture) {
        guard let ciImage = CIImage(mtlTexture: mtlTexture, options: nil) else { return nil }
        let rep = NSCIImageRep(ciImage: ciImage)
        self.init(size: rep.size)
        self.addRepresentation(rep)
    }
}

struct ImageViewWrapper: NSViewRepresentable {
    let nsImage: NSImage

    func makeNSView(context: Context) -> NSImageView {
        let imageView = NSImageView()
        imageView.image = nsImage
        return imageView
    }

    func updateNSView(_ nsView: NSImageView, context: Context) {
        nsView.image = nsImage
    }
}

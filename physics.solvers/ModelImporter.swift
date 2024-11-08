//
//  ModelImporter.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 10/20/24.
//
import Foundation
import simd
import GLTFKit2


struct Triangle {
    var v1 : SIMD3<Float> = .zero
    var v2 : SIMD3<Float> = .zero
    var v3 : SIMD3<Float> = .zero
    var c : SIMD3<Float> = .zero
}

class ModelImporter
{
    var asset : GLTFAsset?
    var triangles : [Triangle]?
    
    init(_ name: String, completion: @escaping (Bool) -> Void)
    {
        let gltfName = "scene"
        guard let assetURL = Bundle.main.url(forResource: gltfName, withExtension: "gltf")
        else {
            print("Failed to find specified gltf file \(gltfName)")
            completion(false)
            return
        }
        
        GLTFAsset.load(with: assetURL, options: [:]) { (progress, status, maybeAsset, maybeError, _)  in
            DispatchQueue.main.async {
                
                if status == .complete {
                    self.asset = maybeAsset!
                    print("Loaded asset")
                    self.loadTriangles()
                    print("Loaded triangles buffer")
                    completion(true)
                    return
                    
                }
                
                else if let error = maybeError {
                    print("Failed to load asset: \(error)")
                }
                completion(false)
                return
            }
        }
    }
    
    private func loadTriangles()
    {
        var tris = [Triangle]()
        
        var pos : [SIMD3<Float>] = []
        var ids : [Int] = []
        let primitive = asset!.meshes[0].primitives[0]
        
        print("loading mesh with \(primitive.indices!.count) vertices")
        
        var minimums = simd_float3(10000, 10000, 10000);
        var maximums = simd_float3.zero;
        if let _positions = primitive.copyPackedVertexPositions() {
            let posPtr = _positions.withUnsafeBytes { bytes in
                return UnsafeRawBufferPointer(start: bytes.baseAddress, count: bytes.count)
            }
            
            for i in 0..<_positions.count / (MemoryLayout<Float>.stride * 3)
            {
                let position = posPtr.baseAddress!.advanced(by: MemoryLayout<Float>.stride * 3 * i).assumingMemoryBound(to: Float.self)
                
                let xp = position[0]
                let yp = position[1]
                let zp = position[2]
                
                if (xp) > maximums.x { maximums.x = xp }
                if (yp) > maximums.y { maximums.y = yp }
                if (zp) > maximums.z { maximums.z = zp }
                if (xp) < minimums.x { minimums.x = xp }
                if (yp) < minimums.y { minimums.y = yp }
                if (zp) < minimums.z { minimums.z = zp }
                
                pos.append(SIMD3<Float>(x: xp, y: yp, z: zp))
            }
        }
        
        //trying to center the model
        let span = maximums - minimums
        let fitFactor = min(64 / span.x, 64 / span.y, 64 / span.z)
        var offset = -minimums * fitFactor
        //let fitFactor : Float = 1
        print("model minimum: \(minimums), maximums: \(maximums), span: \(maximums - minimums), scaleFactor: \(fitFactor)")
        print("offset to 0: \(offset)")
        offset += simd_float3(32 - (fitFactor * span.x/2), 32 - (fitFactor * span.y/2), 32 - (fitFactor * span.z/2))
        print("offset to the center: \(offset)")
        if let indices = primitive.indices {
            
            let uint16Data = indices.bufferView!.buffer.data!.withUnsafeBytes { $0.bindMemory(to: UInt16.self)}
            for i in stride(from: 0, to: indices.count * 2, by: MemoryLayout<UInt16>.stride)
            {
                let index = Int(uint16Data[i])
                ids.append(index)
                
            }
            
            for i in 0..<indices.count / 3 {
                let a = ids[i * 3]
                let b = ids[i * 3 + 1]
                let c = ids[i * 3 + 2]
                
                let triangle = Triangle(v1: (pos[a] * fitFactor) + offset, v2: (pos[b] * fitFactor) + offset, v3: (pos[c] * fitFactor) + offset)
                tris.append(triangle)
            }
        }
        
        self.triangles = tris
    }
}

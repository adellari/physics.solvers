//
//  MeshSDF.swift
//  physics.solvers
//
//  Created by Adellar Irankunda on 11/8/24.
//
import Metal

class MeshSDF
{
    var device : MTLDevice
    var voxelizer : MTLComputePipelineState
    var jfer : MTLComputePipelineState
    var sdfer : MTLComputePipelineState
    var slicer : MTLComputePipelineState
    var voxelTex : MTLTexture
    var sdfTex : MTLTexture
    var sliceTex : MTLTexture
    var commandQueue : MTLCommandQueue
    var triangles : [Triangle]?
    var voxelGroups : MTLSize?
    var trisCount : Int?
    
    init(_device : MTLDevice, sharedQueue : MTLCommandQueue?) throws
    {
        self.device = _device
        let library = try device.makeDefaultLibrary(bundle: .main)
        let voxelFunc = library.makeFunction(name: "MeshToVoxel")!
        let jfaFunc = library.makeFunction(name: "JFAIteration")!
        let sdfFunc = library.makeFunction(name: "JFAPost")!
        let sliceFunc = library.makeFunction(name: "ExtractSlice")!
        voxelizer = try device.makeComputePipelineState(function: voxelFunc)
        jfer = try device.makeComputePipelineState(function: jfaFunc)
        sdfer = try device.makeComputePipelineState(function: sdfFunc)
        slicer = try device.makeComputePipelineState(function: sliceFunc)
        self.commandQueue = sharedQueue ?? device.makeCommandQueue()!
        
        let volumeDesc = MTLTextureDescriptor()
        volumeDesc.pixelFormat = .rgba16Float
        volumeDesc.textureType = .type3D
        volumeDesc.width = 64; volumeDesc.height = 64; volumeDesc.depth = 64;
        volumeDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        
        let sliceDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: 512, height: 512, mipmapped: false)
        sliceDesc.usage = MTLTextureUsage([.shaderRead, .shaderWrite])
        
        voxelTex = device.makeTexture(descriptor: volumeDesc)!
        sdfTex = device.makeTexture(descriptor: volumeDesc)!
        sliceTex = device.makeTexture(descriptor: sliceDesc)!
        
        LoadMesh()
    }
    
    func LoadMesh()
    {

        var model : ModelImporter?
        model = ModelImporter("vr_hand_simple") { success in
            
            if success{
                self.triangles = model!.triangles!
                print("loaded \(self.triangles!.count) triangles")
                self.trisCount = self.triangles!.count
                let root = sqrt(Double(self.trisCount!))
                let groupSize = ceil(root / 16)
                self.voxelGroups = MTLSize(width: Int(groupSize), height: Int(groupSize), depth: 1)
                print("voxel groups: \(self.voxelGroups!.width) x \(self.voxelGroups!.height) x \(self.voxelGroups!.depth)")
            }
            else {
                fatalError("failed to load model and triangles")
            }
            
        }
        
    }
    
    func Voxelize(sharedBuffer : MTLCommandBuffer? = nil, outputSlice : Int?)
    {
        
        let commandBuffer = sharedBuffer ?? commandQueue.makeCommandBuffer()!
        let trisBuffer = device.makeBuffer(bytes: self.triangles!, length: self.triangles!.count * MemoryLayout<Triangle>.stride, options: [])
        
        let voxelEncoder = commandBuffer.makeComputeCommandEncoder()!
        voxelEncoder.setComputePipelineState(voxelizer)
        voxelEncoder.label = "Mesh to Voxels"
        voxelEncoder.setBuffer(trisBuffer, offset: 0, index: 1)
        voxelEncoder.setBytes(&self.trisCount!, length: MemoryLayout<Int>.size, index: 0)
        voxelEncoder.setTexture(voxelTex, index: 0)
        voxelEncoder.dispatchThreadgroups(voxelGroups!, threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        voxelEncoder.endEncoding()
        
        var iteration = 32
        
        while iteration >= 1
        {
            var iter = Int16(iteration)
            let jfaEncoder = commandBuffer.makeComputeCommandEncoder()!
            jfaEncoder.setComputePipelineState(jfer)
            jfaEncoder.setTexture(voxelTex, index: 0)
            jfaEncoder.setBytes(&iter, length:MemoryLayout<Int16>.size, index: 0)
            jfaEncoder.dispatchThreadgroups(MTLSize(width: 8, height: 8, depth: 8), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 8))
            jfaEncoder.endEncoding()
            
            iteration /= 2
        }
        
        let sdfEncoder = commandBuffer.makeComputeCommandEncoder()!
        sdfEncoder.setComputePipelineState(sdfer)
        sdfEncoder.setTexture(voxelTex, index: 0)
        sdfEncoder.dispatchThreadgroups(MTLSize(width: 8, height: 8, depth: 8), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 8))
        sdfEncoder.endEncoding()
        
        var sliceIndex = outputSlice ?? 31
        let sliceEncoder = commandBuffer.makeComputeCommandEncoder()!
        sliceEncoder.setComputePipelineState(slicer)
        sliceEncoder.setTexture(voxelTex, index: 0)
        sliceEncoder.setTexture(sliceTex, index: 1)
        sliceEncoder.setBytes(&sliceIndex, length:MemoryLayout<Int>.size, index: 0)
        sliceEncoder.dispatchThreadgroups(MTLSize(width: 32, height: 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 512/32, height: 512/32, depth: 1))
        sliceEncoder.endEncoding()
        
        commandBuffer.commit();
        
        
    }
    
}

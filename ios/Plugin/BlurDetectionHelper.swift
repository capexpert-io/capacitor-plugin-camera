import Foundation
import UIKit
import TensorFlowLite

/**
 * TensorFlow Lite Blur Detection Helper for iOS
 * Based on MobileNetV2 model trained for blur detection
 */
class BlurDetectionHelper {
    
    private static let TAG = "BlurDetectionHelper"
    private static let INPUT_WIDTH = 600 // Model's expected input width
    private static let INPUT_HEIGHT = 600 // Model's expected input height
    private static let BATCH_SIZE = 1 // Model expects a batch size of 1
    private static let NUM_CHANNELS = 3 // RGB
    private static let NUM_CLASSES = 2 // blur, sharp
    
    private var interpreter: Interpreter?
    private var isInitialized = false
    
    // Text recognition blur detection helper
    private var textBlurHelper: TextRecognitionBlurHelper?
    private var useTextRecognition = true
    
    init() {
        // Initialize text recognition blur helper
        textBlurHelper = TextRecognitionBlurHelper()
    }
    
    /**
     * Initialize the TFLite model
     * @return true if initialization successful
     */
    func initialize() -> Bool {
        do {
            // Load model from framework bundle
            let frameworkBundle = Bundle(for: type(of: self))
            guard let modelPath = frameworkBundle.path(forResource: "blur_detection_model", ofType: "tflite") else {
                return false
            }
            
            // Configure interpreter options for better performance
            var options = Interpreter.Options()
            options.threadCount = 4 // Use multiple threads for better performance
            
            // Create interpreter
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            
            // Allocate memory for input and output tensors
            try interpreter?.allocateTensors()
            
            // Initialize text recognition helper
            textBlurHelper?.initialize()
            
            isInitialized = true
            return true
            
        } catch {
            isInitialized = false
            return false
        }
    }
    
    /**
     * Detect blur in image using hybrid approach (Text Recognition + TFLite)
     * @param image Input UIImage
     * @return Blur confidence score (0.0 = sharp, 1.0 = very blurry)
     */
    func detectBlur(image: UIImage) -> Double {
        
        // First try text recognition if enabled
        if useTextRecognition, let textHelper = textBlurHelper, textHelper.getIsInitialized() {
            
            let textResult = textHelper.detectBlurWithConfidence(image: image)
            let hasText = textResult["hasText"] as? Bool ?? false
            
            if hasText {
                // Image contains text, use text recognition result
                let isBlur = textResult["isBlur"] as? Bool
                let textConfidence = textResult["textConfidence"] as? Double
                let wordCount = textResult["wordCount"] as? Int
                let readableWords = textResult["readableWords"] as? Int
                
                
                if let isBlur = isBlur {
                    let blurConfidence = isBlur ? 1.0 : 0.0
                    return blurConfidence
                } else {
                }
            } else {
            }
        }
        
        // Fallback to TFLite model
        return detectBlurWithTFLite(image: image)
    }
    
    /**
     * Detect blur in image using TFLite model only
     * @param image Input UIImage
     * @return Blur confidence score (0.0 = sharp, 1.0 = very blurry)
     */
    func detectBlurWithTFLite(image: UIImage) -> Double {
        guard isInitialized, let interpreter = interpreter else {
            let laplacianScore = calculateLaplacianBlurScore(image: image)
            let isBlur = laplacianScore < 150
            return isBlur ? 1.0 : 0.0
        }
        
        
        do {
            // Preprocess image for model input
            guard let inputData = preprocessImage(image) else {
                let laplacianScore = calculateLaplacianBlurScore(image: image)
                let isBlur = laplacianScore < 150
                return isBlur ? 1.0 : 0.0
            }
            
            // Copy input data to interpreter
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference
            try interpreter.invoke()
            
            // Get output data
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            
            // Parse output probabilities (assuming float32 output)
            let probabilities = outputData.withUnsafeBytes { bytes in
                Array(bytes.bindMemory(to: Float32.self))
            }
            
            // probabilities[0] = blur probability, probabilities[1] = sharp probability
            let blurConfidence = probabilities.count > 0 ? Double(probabilities[0]) : 0.0
            let sharpConfidence = probabilities.count > 1 ? Double(probabilities[1]) : 0.0
            
            // Determine if image is blurry using TFLite confidence
            let isBlur = (blurConfidence >= 0.99 || sharpConfidence < 0.1)

            
            // Return 1.0 for blur, 0.0 for sharp (to maintain double return type)
            return isBlur ? 1.0 : 0.0
            
        } catch {
            // Fallback to Laplacian algorithm
            let laplacianScore = calculateLaplacianBlurScore(image: image)
            let isBlur = laplacianScore < 150
            return isBlur ? 1.0 : 0.0
        }
    }
    
    /**
     * Preprocess image for MobileNetV2 input (224x224 RGB, normalized)
     */
    private func preprocessImage(_ image: UIImage) -> Data? {
        // Resize image to model's expected dimensions
        guard let resizedImage = resizeImage(image, to: CGSize(width: Self.INPUT_WIDTH, height: Self.INPUT_HEIGHT)) else {
            return nil
        }
        
        // Convert to pixel data
        guard let cgImage = resizedImage.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        
        let context = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )
        
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        // Convert to float array and normalize to [0, 1]
        var normalizedPixels = [Float32]()
        let totalPixels = Self.BATCH_SIZE * width * height * Self.NUM_CHANNELS
        normalizedPixels.reserveCapacity(totalPixels)
        
        // Add batch dimension by repeating the image data BATCH_SIZE times
        for _ in 0..<Self.BATCH_SIZE {
            for i in stride(from: 0, to: pixelData.count, by: bytesPerPixel) {
                let r = Float32(pixelData[i]) / 255.0
                let g = Float32(pixelData[i + 1]) / 255.0
                let b = Float32(pixelData[i + 2]) / 255.0
                normalizedPixels.append(r)
                normalizedPixels.append(g)
                normalizedPixels.append(b)
            }
        }
        
        return normalizedPixels.withUnsafeBufferPointer { buffer in
            return Data(buffer: buffer)
        }
    }
    
    /**
     * Resize image to target size
     */
    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    /**
     * Fallback Laplacian blur detection (from original implementation)
     */
    private func calculateLaplacianBlurScore(image: UIImage) -> Double {
        guard let cgImage = image.cgImage else { return 0.0 }
        
        let width = cgImage.width
        let height = cgImage.height
        
        // Create bitmap context to access pixel data
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return 0.0 }
        
        // Draw image into context
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let pixelData = context.data else { return 0.0 }
        let data = pixelData.bindMemory(to: UInt8.self, capacity: width * height * bytesPerPixel)
        
        // Convert to grayscale and apply Laplacian kernel
        var variance = 0.0
        var count = 0
        
        // Sample every 4th pixel for performance
        let step = 4
        for y in stride(from: step, to: height - step, by: step) {
            for x in stride(from: step, to: width - step, by: step) {
                let idx = (y * width + x) * bytesPerPixel
                
                // Convert to grayscale
                let r = Double(data[idx])
                let g = Double(data[idx + 1])
                let b = Double(data[idx + 2])
                let gray = 0.299 * r + 0.587 * g + 0.114 * b
                
                // Calculate neighbors for 3x3 Laplacian kernel
                let neighbors: [Double] = [
                    getGrayscaleValue(data: data, x: x-1, y: y-1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x, y: y-1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x+1, y: y-1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x-1, y: y, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x+1, y: y, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x-1, y: y+1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x, y: y+1, width: width, bytesPerPixel: bytesPerPixel),
                    getGrayscaleValue(data: data, x: x+1, y: y+1, width: width, bytesPerPixel: bytesPerPixel)
                ]
                
                // Apply 3x3 Laplacian kernel
                let laplacian = -neighbors[0] - neighbors[1] - neighbors[2] +
                               -neighbors[3] + 8 * gray - neighbors[4] +
                               -neighbors[5] - neighbors[6] - neighbors[7]
                
                variance += laplacian * laplacian
                count += 1
            }
        }
        
        return count > 0 ? variance / Double(count) : 0.0
    }
    
    private func getGrayscaleValue(data: UnsafePointer<UInt8>, x: Int, y: Int, width: Int, bytesPerPixel: Int) -> Double {
        let idx = (y * width + x) * bytesPerPixel
        let r = Double(data[idx])
        let g = Double(data[idx + 1])
        let b = Double(data[idx + 2])
        return 0.299 * r + 0.587 * g + 0.114 * b
    }
    


    /**
     * Detect blur with detailed confidence scores using hybrid approach
     * @param image Input UIImage
     * @return Dictionary with comprehensive blur detection results
     */
    func detectBlurWithConfidence(image: UIImage) -> [String: Any] {
        // Try text recognition first if enabled
        if useTextRecognition, let textHelper = textBlurHelper, textHelper.getIsInitialized() {
            let textResult = textHelper.detectBlurWithConfidence(image: image)
            let hasText = textResult["hasText"] as? Bool ?? false
            
            if hasText {
                // Image contains text, use text recognition result
                let isBlur = textResult["isBlur"] as? Bool
                let textConfidence = textResult["textConfidence"] as? Double
                
                var result: [String: Any] = [:]
                result["method"] = "text_recognition"
                result["isBlur"] = isBlur
                result["textConfidence"] = textConfidence
                result["wordCount"] = textResult["wordCount"]
                result["readableWords"] = textResult["readableWords"]
                result["hasText"] = true
                result["boundingBoxes"] = textResult["boundingBoxes"]
                
                // Set blur/sharp confidence based on text recognition result
                if let isBlur = isBlur, let textConfidence = textConfidence {
                    if isBlur {
                        // Image is blurry - high blur confidence, low sharp confidence
                        result["blurConfidence"] = 1.0
                        result["sharpConfidence"] = 0.0
                    } else {
                        // Image is sharp - low blur confidence, high sharp confidence
                        result["blurConfidence"] = 1.0 - textConfidence
                        result["sharpConfidence"] = textConfidence
                    }
                } else {
                    // Default values if confidence not available
                    result["blurConfidence"] = (isBlur ?? false) ? 1.0 : 0.0
                    result["sharpConfidence"] = (isBlur ?? true) ? 0.0 : 1.0
                }
                
                
                return result
            }
        }
        
        // Fallback to TFLite model
        return detectBlurWithTFLiteConfidence(image: image)
    }
    
    /**
     * Detect blur with detailed confidence scores using TFLite only
     * @param image Input UIImage
     * @return Dictionary with isBlur, blurConfidence, and sharpConfidence
     */
    func detectBlurWithTFLiteConfidence(image: UIImage) -> [String: Any] {
        guard isInitialized, let interpreter = interpreter else {
            let laplacianScore = calculateLaplacianBlurScore(image: image)
            let isBlur = laplacianScore < 150
            let normalizedScore = max(0.0, min(1.0, laplacianScore / 300.0))
            let sharpConfidence = normalizedScore
            let blurConfidence = 1.0 - normalizedScore
            
            return [
                "method": "laplacian",
                "isBlur": isBlur,
                "blurConfidence": blurConfidence,
                "sharpConfidence": sharpConfidence,
                "laplacianScore": laplacianScore,
                "hasText": false,
                "boundingBoxes": [[Double]]()
            ]
        }
        
        do {
            // Preprocess image for model input
            guard let inputData = preprocessImage(image) else {
                let laplacianScore = calculateLaplacianBlurScore(image: image)
                let isBlur = laplacianScore < 150
                let normalizedScore = max(0.0, min(1.0, laplacianScore / 300.0))
                let sharpConfidence = normalizedScore
                let blurConfidence = 1.0 - normalizedScore
                
                return [
                    "method": "laplacian",
                    "isBlur": isBlur,
                    "blurConfidence": blurConfidence,
                    "sharpConfidence": sharpConfidence,
                    "laplacianScore": laplacianScore,
                    "hasText": false
                ]
            }
            
            // Copy input data to interpreter
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference
            try interpreter.invoke()
            
            // Get output data
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            
            // Parse output probabilities (assuming float32 output)
            let probabilities = outputData.withUnsafeBytes { bytes in
                Array(bytes.bindMemory(to: Float32.self))
            }
            
            // probabilities[0] = blur probability, probabilities[1] = sharp probability
            let blurConfidence = probabilities.count > 0 ? Double(probabilities[0]) : 0.0
            let sharpConfidence = probabilities.count > 1 ? Double(probabilities[1]) : 0.0
            
            // Determine if image is blurry using TFLite confidence
            let isBlur = (blurConfidence >= 0.99 || sharpConfidence < 0.1)

            
            return [
                "method": "tflite",
                "isBlur": isBlur,
                "blurConfidence": blurConfidence,
                "sharpConfidence": sharpConfidence,
                "hasText": false,
                "boundingBoxes": [[Double]]()
            ]
            
        } catch {
            // Fallback to Laplacian algorithm
            let laplacianScore = calculateLaplacianBlurScore(image: image)
            let isBlur = laplacianScore < 150
            let normalizedScore = max(0.0, min(1.0, laplacianScore / 300.0))
            let sharpConfidence = normalizedScore
            let blurConfidence = 1.0 - normalizedScore
            
            return [
                "method": "laplacian",
                "isBlur": isBlur,
                "blurConfidence": blurConfidence,
                "sharpConfidence": sharpConfidence,
                "laplacianScore": laplacianScore,
                "hasText": false,
                "boundingBoxes": [[Double]]()
            ]
        }
    }

    /**
     * Check if image is blurry
     * @param image Input image
     * @return true if image is blurry, false if sharp
     */
    func isBlurry(image: UIImage) -> Bool {
        let result = detectBlur(image: image)
        return result == 1.0
    }

    /**
     * Get blur percentage (0-100%) - Deprecated, use isBlurry() instead
     * @param image Input image
     * @return Blur percentage where 0% = sharp, 100% = very blurry
     */
    @available(*, deprecated, message: "Use isBlurry() instead")
    func getBlurPercentage(image: UIImage) -> Double {
        // Convert boolean result to percentage for backward compatibility
        return isBlurry(image: image) ? 100.0 : 0.0
    }
    
    /**
     * Enable or disable text recognition for blur detection
     * @param enable true to enable text recognition, false to use only TFLite
     */
    func setTextRecognitionEnabled(_ enable: Bool) {
        useTextRecognition = enable
    }
    
    /**
     * Enable or disable dictionary check in text recognition
     * @param enable true to enable dictionary check
     */
    func setDictionaryCheckEnabled(_ enable: Bool) {
        textBlurHelper?.setDictionaryCheckEnabled(enable)
    }
    
    /**
     * Check if text recognition is enabled
     * @return true if text recognition is enabled
     */
    func isTextRecognitionEnabled() -> Bool {
        return useTextRecognition
    }
    
    /**
     * Get text recognition helper instance
     * @return TextRecognitionBlurHelper instance
     */
    func getTextBlurHelper() -> TextRecognitionBlurHelper? {
        return textBlurHelper
    }
    
    /**
     * Clean up resources
     */
    func close() {
        interpreter = nil
        textBlurHelper?.close()
        textBlurHelper = nil
        isInitialized = false
    }
    
    /**
     * Check if TFLite model is properly initialized
     */
    func getIsInitialized() -> Bool {
        return isInitialized
    }
} 

import Foundation
import UIKit
import TensorFlowLite
import MLKitObjectDetection
import MLKitTextRecognition
import MLKitVision

/**
 * Comprehensive Blur Detection Helper for iOS
 * Implements the 3-step pipeline: Object Detection -> Text Detection -> Full-Image Blur Detection
 */
class BlurDetectionHelper {
    
    private static let TAG = "BlurDetectionHelper"
    private static let INPUT_WIDTH = 600 // Model's expected input width
    private static let INPUT_HEIGHT = 600 // Model's expected input height
    private static let BATCH_SIZE = 1 // Model expects a batch size of 1
    private static let NUM_CHANNELS = 3 // RGB
    private static let NUM_CLASSES = 2 // blur, sharp
    
    // Timeout settings
    private static let TIMEOUT_MS: TimeInterval = 5.0 // 5 second timeout
    
    // Text recognition settings
    private static let MIN_WORD_CONFIDENCE: Double = 0.8 // 80% confidence threshold
    private static let AT_LEAST_N_PERCENT_OF_WORDS_ARE_READABLE: Double = 0.6 // 60% of words are readable
    private static let AT_LEAST_N_PERCENT_OF_AVERAGE_CONFIDENCE: Double = 0.85 // 85% of average confidence

    // Method based confidence threshold
    private static let MIN_SHARP_CONFIDENCE_FOR_OBJECT_DETECTION: Double = 0.45 // 45% confidence threshold
    private static let MIN_SHARP_CONFIDENCE_FOR_TEXT_DETECTION: Double = 0.09 // 9% confidence threshold
    private static let MIN_SHARP_CONFIDENCE_FOR_FULL_IMAGE: Double = 0.65 // 65% confidence threshold
    	
    private var interpreter: Interpreter?
    private var isInitialized = false
    
    // Configuration flags
    private var useObjectDetection = true
    private var useTextRecognition = true
    private var useDictionaryCheck = false
    
    // Dictionary for text validation
    private var commonWords: Set<String>
    
    // Executor for parallel processing
    private let processingQueue = DispatchQueue(label: "blur.detection.queue", qos: .userInitiated)
    
    // ML Kit detectors
    private var objectDetector: ObjectDetector?
    private var textRecognizer: TextRecognizer?
    
    // Text recognition result structure
    private struct TextRecognitionResult {
        let isReadable: Bool
        let averageConfidence: Double
        let totalWords: Int
        let readableWords: Int
        let boundingBoxes: [[Double]]
    }
    
    init() {
        // Initialize common words dictionary
        commonWords = Set<String>()
        initializeCommonWords()
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
            
            // Initialize ML Kit Object Detector
            let objectDetectorOptions = ObjectDetectorOptions()
            objectDetectorOptions.detectorMode = .singleImage
            objectDetectorOptions.shouldEnableClassification = true
            objectDetector = ObjectDetector.objectDetector(options: objectDetectorOptions)
            
            // Initialize ML Kit Text Recognizer (updated API)
            let textOptions = TextRecognizerOptions()
            textRecognizer = TextRecognizer.textRecognizer(options: textOptions)
            
            isInitialized = true
            return true
            
        } catch {
            isInitialized = false
            logError(error, context: "initialize()")
            return false
        }
    }
    
    /**
     * Main blur detection method implementing the 3-step pipeline
     * @param image Input UIImage
     * @return Blur confidence score (0.0 = sharp, 1.0 = very blurry)
     */
    func detectBlur(image: UIImage) -> Double {
        let result = detectBlurWithConfidence(image: image)
        let isBlur = result["isBlur"] as? Bool ?? false
        return isBlur ? 1.0 : 0.0
    }
    
    /**
     * Detect blur with detailed confidence scores using the 3-step pipeline
     * @param image Input UIImage
     * @return Dictionary with comprehensive blur detection results
     */
    func detectBlurWithConfidence(image: UIImage) -> [String: Any] {
        var result: [String: Any] = [:]
        
        if !isInitialized {
            result["isBlur"] = false
            result["method"] = "none"
            result["error"] = "Blur detector not initialized"
            return result
        }
        
        // Step 1: Object Detection (Preferred Path)
        if useObjectDetection {
            print("\(Self.TAG): Step 1: Trying Object Detection")
            let objects = detectObjects(image: image)
            
            if !objects.isEmpty {
                print("\(Self.TAG): Objects detected: \(objects.count)")
                
                // Process ROIs from detected objects
                var roiResults: [[String: Any]] = []
                var boundingBoxes: [[Double]] = []
                var anyBlurry = false
                
                for object in objects {
                    let boundingBox = object.frame
                    // Always record the detected bounding box for JS consumers
                    boundingBoxes.append([
                        Double(boundingBox.minX),
                        Double(boundingBox.minY),
                        Double(boundingBox.maxX),
                        Double(boundingBox.maxY)
                    ])


                    print("\(Self.TAG): Object Detection Result isBlur: \(boundingBox.minX), \(boundingBox.minY), \(boundingBox.maxX), \(boundingBox.maxY)")
                    
                    // Crop ROI from original image
                    if let roi = cropImage(image: image, rect: boundingBox) {
                        // Run TFLite blur detection on ROI
                        var roiResult = detectBlurWithTFLiteConfidence(image: roi, minSharpConfidence: Self.MIN_SHARP_CONFIDENCE_FOR_OBJECT_DETECTION)
                        
                        print("\(Self.TAG): Object Detection ROI Result isBlur: \(roiResult["isBlur"] ?? false)")
                        roiResult["boundingBox"] = [
                            boundingBox.minX,
                            boundingBox.minY,
                            boundingBox.maxX,
                            boundingBox.maxY
                        ]
                        roiResults.append(roiResult)
                        
                        let isBlur = roiResult["isBlur"] as? Bool ?? false
                        if isBlur {
                            anyBlurry = true
                        }
                    }
                }
                
                result["method"] = "object_detection"
                result["isBlur"] = anyBlurry
                result["roiResults"] = roiResults
                result["objectCount"] = objects.count
                result["boundingBoxes"] = boundingBoxes
                return result
            }
        }
        
        // Step 2: Text Detection (Fallback if Object Not Found)
        if useTextRecognition {
            print("\(Self.TAG): Step 2: Trying Text Detection")
            let textResult = detectTextWithBoundingBoxes(image: image)
            
            if textResult.totalWords > 0 {
                print("\(Self.TAG): Text detected: \(textResult.totalWords) words")
                
                // Combine all detected text areas into a single bounding box
                let topTextAreas = combineAllTextAreas(boundingBoxes: textResult.boundingBoxes)
                
                if !topTextAreas.isEmpty {
                    // Process ROIs from text areas
                    let roiResults = processROIsInParallel(image: image, rois: topTextAreas)
                    
                    // Collect top-level bounding boxes for JS consumers
                    let boundingBoxes: [[Double]] = topTextAreas.map { roi in
                        [
                            Double(roi.minX),
                            Double(roi.minY),
                            Double(roi.maxX),
                            Double(roi.maxY)
                        ]
                    }
                    
                    // Check if any ROI is blurry
                    var anyBlurry = false
                    for roiResult in roiResults {
                        let isBlur = roiResult["isBlur"] as? Bool ?? false
                        if isBlur {
                            anyBlurry = true
                            break
                        }
                    }
                    
                    result["method"] = "text_detection"
                    result["isBlur"] = anyBlurry
                    result["roiResults"] = roiResults
                    result["textConfidence"] = textResult.averageConfidence
                    result["wordCount"] = textResult.totalWords
                    result["readableWords"] = textResult.readableWords
                    result["boundingBoxes"] = boundingBoxes
                    return result
                }
            }
        }
        
        // Step 3: Full-Image Blur Detection (Final Fallback)
        print("\(Self.TAG): Step 3: Using Full-Image Blur Detection")
        return detectBlurWithTFLiteConfidence(image: image, minSharpConfidence: Self.MIN_SHARP_CONFIDENCE_FOR_FULL_IMAGE)
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
            let sharpConfidence = probabilities.count > 1 ? Double(probabilities[1]) : 0.0
            
            // Determine if image is blurry using TFLite confidence
            let isBlur = sharpConfidence < Self.MIN_SHARP_CONFIDENCE_FOR_FULL_IMAGE

            
            // Return 1.0 for blur, 0.0 for sharp (to maintain double return type)
            return isBlur ? 1.0 : 0.0
            
        } catch {
            logError(error, context: "detectBlurWithTFLite(image:)")
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
        // Determine model input shape dynamically from the interpreter if available
        var modelWidth = Self.INPUT_WIDTH
        var modelHeight = Self.INPUT_HEIGHT
        var modelChannels = Self.NUM_CHANNELS
        var isQuantizedInput = false
        if let interpreter = interpreter, let inputTensor = try? interpreter.input(at: 0) {
            let dims = inputTensor.shape.dimensions
            if dims.count == 4 {
                // NHWC
                modelHeight = dims[1]
                modelWidth = dims[2]
                modelChannels = dims[3]
            }
            isQuantizedInput = (inputTensor.dataType == .uInt8)
        }
        
        // Resize image to model's expected dimensions
        guard let resizedImage = resizeImage(image, to: CGSize(width: modelWidth, height: modelHeight)) else {
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
        
        // Prepare output buffer based on expected tensor type and channels (NHWC)
        if isQuantizedInput {
            var quantizedPixels = [UInt8]()
            quantizedPixels.reserveCapacity(Self.BATCH_SIZE * width * height * modelChannels)
            
            for _ in 0..<Self.BATCH_SIZE {
                for i in stride(from: 0, to: pixelData.count, by: bytesPerPixel) {
                    let r = pixelData[i]
                    let g = pixelData[i + 1]
                    let b = pixelData[i + 2]
                    if modelChannels == 1 {
                        // Luma approximation
                        let gray = UInt8(min(255, Int(0.299 * Double(r) + 0.587 * Double(g) + 0.114 * Double(b))))
                        quantizedPixels.append(gray)
                    } else {
                        // Assume RGB ordering
                        quantizedPixels.append(r)
                        quantizedPixels.append(g)
                        quantizedPixels.append(b)
                    }
                }
            }
            return quantizedPixels.withUnsafeBufferPointer { Data(buffer: $0) }
        } else {
            // Float32 normalized to [0,1]
            var normalizedPixels = [Float32]()
            normalizedPixels.reserveCapacity(Self.BATCH_SIZE * width * height * modelChannels)
            
            for _ in 0..<Self.BATCH_SIZE {
                for i in stride(from: 0, to: pixelData.count, by: bytesPerPixel) {
                    let r = Float32(pixelData[i]) / 255.0
                    let g = Float32(pixelData[i + 1]) / 255.0
                    let b = Float32(pixelData[i + 2]) / 255.0
                    if modelChannels == 1 {
                        let gray = 0.299 * r + 0.587 * g + 0.114 * b
                        normalizedPixels.append(gray)
                    } else {
                        normalizedPixels.append(r)
                        normalizedPixels.append(g)
                        normalizedPixels.append(b)
                    }
                }
            }
            return normalizedPixels.withUnsafeBufferPointer { Data(buffer: $0) }
        }
    }
    
    /**
     * Resize image to target size while maintaining aspect ratio
     */
    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        let targetSize = size
        let imageSize = image.size
        
        // Calculate aspect ratio preserving dimensions
        let widthRatio = targetSize.width / imageSize.width
        let heightRatio = targetSize.height / imageSize.height
        let scaleFactor = min(widthRatio, heightRatio)
        
        let scaledSize = CGSize(
            width: imageSize.width * scaleFactor,
            height: imageSize.height * scaleFactor
        )
        
        // Create a square canvas with the target size
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        
        // Calculate position to center the scaled image
        let x = (targetSize.width - scaledSize.width) / 2
        let y = (targetSize.height - scaledSize.height) / 2
        let drawRect = CGRect(x: x, y: y, width: scaledSize.width, height: scaledSize.height)
        
        // Fill background with black (or white) to pad the image
        UIColor.black.setFill()
        UIRectFill(CGRect(origin: .zero, size: targetSize))
        
        // Draw the scaled image centered
        image.draw(in: drawRect)
        
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
     * Detect blur with detailed confidence scores using TFLite only
     * @param image Input UIImage
     * @return Dictionary with isBlur, blurConfidence, and sharpConfidence
     */
    func detectBlurWithTFLiteConfidence(image: UIImage, minSharpConfidence: Double) -> [String: Any] {
        guard isInitialized, let interpreter = interpreter else {
            let laplacianScore = calculateLaplacianBlurScore(image: image)
            let isBlur = laplacianScore < 150
            let normalizedScore = max(0.0, min(1.0, laplacianScore / 300.0))
            let sharpConfidence = normalizedScore
            let blurConfidence = 1.0 - normalizedScore

            print(" TFLite Blur Detection Result: laplacianScore \(blurConfidence), \(sharpConfidence)")
            
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
                
                print(" TFLite Blur Detection Result: laplacianScore2 \(blurConfidence), \(sharpConfidence)")

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
            let isBlur = sharpConfidence < minSharpConfidence

            print(" TFLite Blur Detection Result: \(blurConfidence), \(sharpConfidence)")
            
            return [
                "method": "tflite",
                "isBlur": isBlur,
                "blurConfidence": blurConfidence,
                "sharpConfidence": sharpConfidence,
                "hasText": false,
                "boundingBoxes": [[Double]]()
            ]
            
        } catch {
            logError(error, context: "detectBlurWithTFLiteConfidence(image:)")
            // Fallback to Laplacian algorithm
            let laplacianScore = calculateLaplacianBlurScore(image: image)
            let isBlur = laplacianScore < 150
            let normalizedScore = max(0.0, min(1.0, laplacianScore / 300.0))
            let sharpConfidence = normalizedScore
            let blurConfidence = 1.0 - normalizedScore
            
            print(" TFLite Blur Detection Result: laplacianScore3 \(blurConfidence), \(sharpConfidence)")

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
     * Check if text recognition is enabled
     * @return true if text recognition is enabled
     */
    func isTextRecognitionEnabled() -> Bool {
        return useTextRecognition
    }
    
    
    /**
     * Detect objects in the image using ML Kit Object Detection
     * @param image Input UIImage
     * @return List of detected objects
     */
    private func detectObjects(image: UIImage) -> [Object] {
        guard let objectDetector = objectDetector else { 
            print("\(Self.TAG): Object detector not initialized")
            return [] 
        }
        
        let visionImage = VisionImage(image: image)
        visionImage.orientation = image.imageOrientation
        
        var detectedObjects: [Object] = []
        let semaphore = DispatchSemaphore(value: 0)
        
        objectDetector.process(visionImage) { objects, error in
            defer { semaphore.signal() }
            
            if let error = error {
                print("\(Self.TAG): Object detection failed: \(error.localizedDescription)")
                return
            }
            
            guard let objects = objects, !objects.isEmpty else {
                print("\(Self.TAG): No objects detected")
                return
            }
            
            detectedObjects = objects
            print("\(Self.TAG): Detected \(objects.count) objects")
        }
        
        let timeout = DispatchTime.now() + Self.TIMEOUT_MS
        if semaphore.wait(timeout: timeout) == .timedOut {
            print("\(Self.TAG): Object detection timed out")
        }
        
        return detectedObjects
    }
    
    /**
     * Get the top N largest text areas and combine them into a single bounding box
     * @param boundingBoxes List of bounding boxes
     * @param topN Number of top areas to combine
     * @return List containing a single combined bounding box
     */
    private func combineAllTextAreas(boundingBoxes: [[Double]]) -> [CGRect] {
        
        if boundingBoxes.isEmpty {
            return []
        }
        
        // Convert to CGRects
        let rects = boundingBoxes.compactMap { box -> CGRect? in
            guard box.count >= 4 else { return nil }
            return CGRect(x: box[0], y: box[1], width: box[2] - box[0], height: box[3] - box[1])
        }
        
        if rects.isEmpty {
            return []
        }
        
        // Find the min/max across all boxes to create one combined bounding box
        var minX = CGFloat.greatestFiniteMagnitude
        var minY = CGFloat.greatestFiniteMagnitude
        var maxX = -CGFloat.greatestFiniteMagnitude
        var maxY = -CGFloat.greatestFiniteMagnitude
        
        for rect in rects {
            minX = min(minX, rect.minX)
            minY = min(minY, rect.minY)
            maxX = max(maxX, rect.maxX)
            maxY = max(maxY, rect.maxY)
        }
        
        let combinedRect = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
        
        print("\(Self.TAG): Combined \(rects.count) text areas into single bounding box: " +
              "(\(minX), \(minY), \(maxX), \(maxY))")
        
        return [combinedRect]
    }
    
    /**
     * Process multiple ROIs in parallel for blur detection
     * @param image Original image
     * @param rois List of regions of interest
     * @return List of blur detection results for each ROI
     */
    private func processROIsInParallel(image: UIImage, rois: [CGRect]) -> [[String: Any]] {
        var results: [[String: Any]] = []
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "roi.processing", attributes: .concurrent)
        
        for roi in rois {
            group.enter()
            queue.async {
                defer { group.leave() }
                
                if let cropped = self.cropImage(image: image, rect: roi) {
                    var result = self.detectBlurWithTFLiteConfidence(image: cropped, minSharpConfidence: Self.MIN_SHARP_CONFIDENCE_FOR_TEXT_DETECTION)
                    result["boundingBox"] = [
                        roi.minX,
                        roi.minY,
                        roi.maxX,
                        roi.maxY
                    ]
                    
                    DispatchQueue.main.async {
                        results.append(result)
                    }
                }
            }
        }
        
        let timeout = DispatchTime.now() + Self.TIMEOUT_MS
        _ = group.wait(timeout: timeout)
        
        return results
    }
    
    /**
    * Crop an image to the specified rectangle
    * Supports both normalized (0-1) and pixel-based rectangles
    * Handles orientation, scale, and floating-point tolerance
    */
    private func cropImage(image: UIImage, rect: CGRect) -> UIImage? {

        print("cropImage image: \(image.size.width), \(image.size.height)")
        print("cropImage image.scale: \(image.scale)")
        print("cropImage image.imageOrientation: \(image.imageOrientation)")
        print("cropImage rect: minX: \(rect.minX), minY: \(rect.minY), maxX: \(rect.maxX), maxY: \(rect.maxY), width: \(rect.width), height:\(rect.height)")

        guard let cgImage = image.cgImage else { return nil }
        
        // First normalize the image to handle orientation
        let format = UIGraphicsImageRendererFormat()
        format.scale = image.scale
        format.opaque = false
        
        let renderer = UIGraphicsImageRenderer(size: image.size, format: format)
        let normalizedImage = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
        guard let normalizedCGImage = normalizedImage.cgImage else { return nil }
        
        let imageSize = CGSize(width: normalizedCGImage.width, height: normalizedCGImage.height)
        
        // Detect normalized rect with tolerance
        let epsilon: CGFloat = 0.0001
        let isNormalized =
            rect.minX >= -epsilon &&
            rect.minY >= -epsilon &&
            rect.maxX <= 1.0 + epsilon &&
            rect.maxY <= 1.0 + epsilon &&
            rect.width <= 1.0 + epsilon &&
            rect.height <= 1.0 + epsilon
        
        let cropRect: CGRect
        if isNormalized {
            cropRect = CGRect(
                x: rect.minX * imageSize.width,
                y: rect.minY * imageSize.height,
                width: rect.width * imageSize.width,
                height: rect.height * imageSize.height
            )
        } else {
            cropRect = rect
        }
        
        // Clamp to image bounds
        let clampedRect = CGRect(
            x: max(0, cropRect.minX),
            y: max(0, cropRect.minY),
            width: min(imageSize.width - max(0, cropRect.minX), cropRect.width),
            height: min(imageSize.height - max(0, cropRect.minY), cropRect.height)
        )
        
        guard clampedRect.width > 0 && clampedRect.height > 0 else { return nil }
        
        // Perform crop
        if let croppedCGImage = normalizedCGImage.cropping(to: clampedRect) {
            return UIImage(cgImage: croppedCGImage, scale: image.scale, orientation: .up)
        }
        
        return nil
    }

    
    /**
     * Initialize common English words for dictionary check
     */
    private func initializeCommonWords() {
        let words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with",
            "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
            "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
            "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
            "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
            "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
            "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "was", "are",
            "were", "been", "has", "had", "having", "does", "did", "doing", "can", "could", "should",
            "would", "may", "might", "must", "shall", "will", "am", "being", "became", "become", "becomes"
        ]
        
        commonWords = Set(words)
    }
    
    /**
     * Clean up resources
     */
    func close() {
        interpreter = nil
        objectDetector = nil
        textRecognizer = nil
        isInitialized = false
    }
    
    /**
     * Check if TFLite model is properly initialized
     */
    func getIsInitialized() -> Bool {
        return isInitialized
    }
    
    /**
     * Enable or disable object detection
     * @param enable true to enable object detection
     */
    func setObjectDetectionEnabled(_ enable: Bool) {
        useObjectDetection = enable
    }
    
    /**
     * Detect text in image with bounding boxes
     * @param image Input UIImage
     * @return TextRecognitionResult with bounding boxes
     */
    private func detectTextWithBoundingBoxes(image: UIImage) -> TextRecognitionResult {
        guard let textRecognizer = textRecognizer else {
            print("\(Self.TAG): Text recognizer not initialized")
            return TextRecognitionResult(isReadable: false, averageConfidence: 0.0, totalWords: 0, readableWords: 0, boundingBoxes: [])
        }
        
        let visionImage = VisionImage(image: image)
        visionImage.orientation = image.imageOrientation
        
        var recognitionResult: TextRecognitionResult?
        var recognitionError: Error?
        
        // Use semaphore for synchronous execution
        let semaphore = DispatchSemaphore(value: 0)
        
        textRecognizer.process(visionImage) { [weak self] text, error in
            defer { semaphore.signal() }
            
            if let error = error {
                recognitionError = error
                return
            }
            
            guard let text = text else {
                recognitionResult = TextRecognitionResult(isReadable: false, averageConfidence: 0.0, totalWords: 0, readableWords: 0, boundingBoxes: [])
                return
            }
            
            recognitionResult = self?.analyzeMLKitTextConfidence(text: text)
        }
        
        // Wait for completion with timeout
        let timeout = DispatchTime.now() + Self.TIMEOUT_MS
        if semaphore.wait(timeout: timeout) == .timedOut {
            recognitionError = NSError(domain: "TextRecognitionError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Text recognition timed out"])
        }
        
        if let error = recognitionError {
            print("\(Self.TAG): Text recognition error: \(error.localizedDescription)")
            return TextRecognitionResult(isReadable: false, averageConfidence: 0.0, totalWords: 0, readableWords: 0, boundingBoxes: [])
        }
        
        return recognitionResult ?? TextRecognitionResult(isReadable: false, averageConfidence: 0.0, totalWords: 0, readableWords: 0, boundingBoxes: [])
    }
    
    /**
     * Analyze text confidence and extract bounding boxes from ML Kit Text result
     * @param text ML Kit Text result
     * @return TextRecognitionResult with analysis
     */
    private func analyzeMLKitTextConfidence(text: Text) -> TextRecognitionResult {
        var totalWords = 0
        var readableWords = 0
        var totalConfidence = 0.0
        var allText = ""
        var readableText = ""
        var boundingBoxes: [[Double]] = []
        
        for block in text.blocks {
            for line in block.lines {
                for element in line.elements {
                    let elementText = element.text
                    let mlkitConfidence = 0.5
                    
                    allText += elementText + " "
                    
                    // Extract bounding box coordinates from element
                    let frame = element.frame
                    let box: [Double] = [
                        Double(frame.minX),  // left
                        Double(frame.minY),  // top  
                        Double(frame.maxX),  // right
                        Double(frame.maxY)   // bottom
                    ]
                    boundingBoxes.append(box)
                    
                    // Split text into words for individual analysis
                    let words = elementText.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
                    
                    for word in words {
                        let trimmedWord = word.trimmingCharacters(in: .punctuationCharacters)
                        if !trimmedWord.isEmpty {
                            totalWords += 1
                            
                            // Estimate word confidence based on ML Kit confidence and text characteristics
                            let wordConfidence = estimateWordConfidence(word: trimmedWord, visionConfidence: Double(mlkitConfidence))
                            totalConfidence += wordConfidence
                            
                            if wordConfidence >= Self.MIN_WORD_CONFIDENCE {
                                // Optional dictionary check for enhanced validation
                                if !useDictionaryCheck || isInDictionary(word: trimmedWord) {
                                    readableWords += 1
                                    readableText += trimmedWord + " "
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let averageConfidence = totalWords > 0 ? totalConfidence / Double(totalWords) : 0.0
        
        // Image is readable if:
        // 1. We have text AND
        // 2. At least 60% of words are readable OR average confidence is high
        let isReadable = totalWords > 0 &&
                        (readableWords >= max(1, Int(Double(totalWords) * Self.AT_LEAST_N_PERCENT_OF_WORDS_ARE_READABLE)) ||
                         averageConfidence >= Self.AT_LEAST_N_PERCENT_OF_AVERAGE_CONFIDENCE)
        
        return TextRecognitionResult(isReadable: isReadable, averageConfidence: averageConfidence, totalWords: totalWords, readableWords: readableWords, boundingBoxes: boundingBoxes)
    }
    
    /**
     * Estimate word confidence based on Vision confidence and text characteristics
     * @param word The actual word text
     * @param visionConfidence Confidence from Vision framework
     * @return Estimated confidence score
     */
    private func estimateWordConfidence(word: String, visionConfidence: Double) -> Double {
        // Start with base confidence (similar to Android implementation)
        var confidence = 0.55
        
        // Apply text characteristic adjustments (similar to Android implementation)
        
        // Check text length (very short or very long words might be less reliable)
        if word.count >= 3 && word.count <= 15 {
            confidence += 0.2 // Boost confidence for reasonable length words
        }
        
        // Check for common patterns that indicate good recognition
        let alphanumericPattern = "^[a-zA-Z0-9\\s\\-\\.]+$"
        if word.range(of: alphanumericPattern, options: .regularExpression) != nil {
            confidence += 0.15
        }
        
        // Check for mixed case (indicates good character recognition)
        let hasLowercase = word.range(of: "[a-z]", options: .regularExpression) != nil
        let hasUppercase = word.range(of: "[A-Z]", options: .regularExpression) != nil
        if hasLowercase && hasUppercase {
            confidence += 0.1
        }
        
        // Check for numbers (often well recognized)
        if word.range(of: "\\d", options: .regularExpression) != nil {
            confidence += 0.1
        }
        
        // Penalize for special characters that might indicate poor recognition
        let specialCharPattern = "[^a-zA-Z0-9\\s\\-\\.]"
        if word.range(of: specialCharPattern, options: .regularExpression) != nil {
            confidence -= 0.1
        }
        
        return max(0.0, min(1.0, confidence))
    }
    
    /**
     * Check if word is in common dictionary
     * @param word Word to check
     * @return true if word is in dictionary
     */
    private func isInDictionary(word: String) -> Bool {
        guard !commonWords.isEmpty else { return false }
        
        let cleanWord = word.lowercased().replacingOccurrences(of: "[^a-zA-Z]", with: "", options: .regularExpression)
        return commonWords.contains(cleanWord)
    }
    
    // MARK: - Error Logging
    private func logError(_ error: Error, context: String) {
        let stackTrace = Thread.callStackSymbols.joined(separator: "\n")
        print("\(Self.TAG): ERROR [\(context)] => \(error.localizedDescription)\nStack Trace:\n\(stackTrace)")
    }
} 

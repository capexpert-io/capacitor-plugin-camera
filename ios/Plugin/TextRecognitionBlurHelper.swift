import Foundation
import UIKit
import Vision

/**
 * Text Recognition Blur Detection Helper for iOS
 * Uses Vision framework for text recognition to detect blur/sharp images
 * Based on text readability and confidence scores
 */
class TextRecognitionBlurHelper {
    
    private static let TAG = "TextRecognitionBlurHelper"
    private static let MIN_WORD_CONFIDENCE = 0.7 // 70% confidence threshold
    private static let AT_LEAST_N_PERCENT_OF_WORDS_ARE_READABLE = 0.6 // 60% of words are readable
    private static let AT_LEAST_N_PERCENT_OF_AVERAGE_CONFIDENCE = 0.85 // 85% of average confidence
    
    private var isInitialized = false
    private var commonWords: Set<String>
    private var useDictionaryCheck = false
    
    // Text recognition result structure
    private struct TextRecognitionResult {
        let isReadable: Bool
        let averageConfidence: Double
        let totalWords: Int
        let readableWords: Int
    }
    
    init() {
        // Initialize common English words for dictionary check
        commonWords = Set<String>()
        initializeCommonWords()
    }
    
    /**
     * Initialize the text recognition helper
     * @return true if initialization successful
     */
    func initialize() -> Bool {
        // Vision framework is available by default on iOS, no specific initialization needed
        isInitialized = true
        return true
    }
    
    /**
     * Detect if image is blurry based on text recognition confidence
     * @param image Input UIImage
     * @return true if image is blurry (text not readable), false if sharp
     */
    func isBlurry(image: UIImage) -> Bool {
        if !isInitialized {
            return false // Default to not blurry if not initialized
        }
        
        
        do {
            let result = try detectTextWithConfidence(image: image)
            
            let isBlurry = !result.isReadable
            
            return isBlurry
        } catch {
            return false // Default to not blurry on error
        }
    }
    
    /**
     * Detect blur with detailed confidence scores
     * @param image Input UIImage
     * @return Dictionary with isBlur, textConfidence, wordCount, and readableWords
     */
    func detectBlurWithConfidence(image: UIImage) -> [String: Any] {
        var result: [String: Any] = [:]
        
        if !isInitialized {
            result["isBlur"] = false
            result["textConfidence"] = 0.0
            result["wordCount"] = 0
            result["readableWords"] = 0
            result["hasText"] = false
            result["error"] = "Text recognition helper not initialized"
            return result
        }
        
        do {
            let recognitionResult = try detectTextWithConfidence(image: image)
            
            result["isBlur"] = !recognitionResult.isReadable
            result["textConfidence"] = recognitionResult.averageConfidence
            result["wordCount"] = recognitionResult.totalWords
            result["readableWords"] = recognitionResult.readableWords
            result["hasText"] = recognitionResult.totalWords > 0
            
            
            return result
            
        } catch {
            result["isBlur"] = false
            result["textConfidence"] = 0.0
            result["wordCount"] = 0
            result["readableWords"] = 0
            result["hasText"] = false
            result["error"] = error.localizedDescription
            return result
        }
    }
    
    /**
     * Detect text in image with confidence analysis using Vision framework
     * @param image Input UIImage
     * @return TextRecognitionResult with confidence and readability info
     */
    private func detectTextWithConfidence(image: UIImage) throws -> TextRecognitionResult {
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "TextRecognitionError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to get CGImage from UIImage"])
        }
        
        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        var recognitionResult: TextRecognitionResult?
        var recognitionError: Error?
        
        // Use semaphore for synchronous execution
        let semaphore = DispatchSemaphore(value: 0)
        
        let request = VNRecognizeTextRequest { [weak self] (request, error) in
            defer { semaphore.signal() }
            
            if let error = error {
                recognitionError = error
                return
            }
            
            guard let observations = request.results as? [VNRecognizedTextObservation] else {
                recognitionResult = TextRecognitionResult(isReadable: false, averageConfidence: 0.0, totalWords: 0, readableWords: 0)
                return
            }
            
            recognitionResult = self?.analyzeTextConfidence(observations: observations)
        }
        
        // Configure text recognition for better accuracy
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        
        // Perform text recognition
        try requestHandler.perform([request])
        
        // Wait for completion with timeout
        let timeout = DispatchTime.now() + .seconds(5)
        if semaphore.wait(timeout: timeout) == .timedOut {
            throw NSError(domain: "TextRecognitionError", code: 2, userInfo: [NSLocalizedDescriptionKey: "Text recognition timed out"])
        }
        
        if let error = recognitionError {
            throw error
        }
        
        return recognitionResult ?? TextRecognitionResult(isReadable: false, averageConfidence: 0.0, totalWords: 0, readableWords: 0)
    }
    
    /**
     * Analyze text confidence and determine readability
     * @param observations Recognized text observations from Vision framework
     * @return TextRecognitionResult with analysis
     */
    private func analyzeTextConfidence(observations: [VNRecognizedTextObservation]) -> TextRecognitionResult {
        var totalWords = 0
        var readableWords = 0
        var totalConfidence = 0.0
        var allText = ""
        var readableText = ""
        
        
        for observation in observations {
            // Get the top candidate for each observation
            guard let topCandidate = observation.topCandidates(1).first else { continue }
            
            let text = topCandidate.string
            let visionConfidence = Double(topCandidate.confidence)
            
            allText += text + " "
            
            // Split text into words for individual analysis
            let words = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
            
            for word in words {
                let trimmedWord = word.trimmingCharacters(in: .punctuationCharacters)
                if !trimmedWord.isEmpty {
                    totalWords += 1
                    
                    // Estimate word confidence based on Vision confidence and text characteristics
                    let wordConfidence = estimateWordConfidence(word: trimmedWord, visionConfidence: visionConfidence)
                    totalConfidence += wordConfidence
                    
                    
                    if wordConfidence >= Self.MIN_WORD_CONFIDENCE {
                        // Optional dictionary check for enhanced validation
                        if !useDictionaryCheck || isInDictionary(word: trimmedWord) {
                            readableWords += 1
                            readableText += trimmedWord + " "
                        } else {
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
        
        
        return TextRecognitionResult(isReadable: isReadable, averageConfidence: averageConfidence, totalWords: totalWords, readableWords: readableWords)
    }
    
    /**
     * Estimate word confidence based on Vision confidence and text characteristics
     * @param word The actual word text
     * @param visionConfidence Confidence from Vision framework
     * @return Estimated confidence score
     */
    private func estimateWordConfidence(word: String, visionConfidence: Double) -> Double {
        // Start with Vision framework confidence
        var confidence = visionConfidence
        
        // Apply text characteristic adjustments (similar to Android implementation)
        
        // Check text length (very short or very long words might be less reliable)
        if word.count >= 3 && word.count <= 15 {
            confidence += 0.1 // Boost confidence for reasonable length words
        }
        
        // Check for common patterns that indicate good recognition
        let alphanumericPattern = "^[a-zA-Z0-9\\s\\-\\.]+$"
        if word.range(of: alphanumericPattern, options: .regularExpression) != nil {
            confidence += 0.05
        }
        
        // Check for mixed case (indicates good character recognition)
        let hasLowercase = word.range(of: "[a-z]", options: .regularExpression) != nil
        let hasUppercase = word.range(of: "[A-Z]", options: .regularExpression) != nil
        if hasLowercase && hasUppercase {
            confidence += 0.05
        }
        
        // Check for numbers (often well recognized)
        if word.range(of: "\\d", options: .regularExpression) != nil {
            confidence += 0.05
        }
        
        // Penalize for special characters that might indicate poor recognition
        let specialCharPattern = "[^a-zA-Z0-9\\s\\-\\.]"
        if word.range(of: specialCharPattern, options: .regularExpression) != nil {
            confidence -= 0.05
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
     * Enable or disable dictionary check for enhanced readability detection
     * @param enable true to enable dictionary check
     */
    func setDictionaryCheckEnabled(_ enable: Bool) {
        useDictionaryCheck = enable
    }
    
    /**
     * Check if text recognition helper is properly initialized
     * @return true if initialized
     */
    func getIsInitialized() -> Bool {
        return isInitialized
    }
    
    /**
     * Clean up resources
     */
    func close() {
        isInitialized = false
    }
}

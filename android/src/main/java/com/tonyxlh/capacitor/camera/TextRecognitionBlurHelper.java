package com.tonyxlh.capacitor.camera;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Text Recognition Blur Detection Helper using Google ML Kit
 * Detects blur/sharp images based on text readability and confidence scores
 */
public class TextRecognitionBlurHelper {
    private static final String TAG = "TextRecognitionBlurHelper";
    private static final double MIN_WORD_CONFIDENCE = 0.8; // 80% confidence threshold
    private static final int TIMEOUT_MS = 5000; // 5 second timeout for text recognition
    private static final double AT_LEAST_N_PERCENT_OF_WORDS_ARE_READABLE = 0.6; // 60% of words are readable
    private static final double AT_LEAST_N_PERCENT_OF_AVERAGE_CONFIDENCE = 0.85; // 85% of average confidence
    
    private TextRecognizer textRecognizer;
    private boolean isInitialized = false;
    
    // Optional dictionary for enhanced readability check
    private Set<String> commonWords;
    private boolean useDictionaryCheck = false;

    public TextRecognitionBlurHelper() {
        // Initialize common English words for dictionary check
        initializeCommonWords();
    }

    /**
     * Initialize the text recognizer
     * @param context Android context
     * @return true if initialization successful
     */
    public boolean initialize(Context context) {
        try {
            textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
            isInitialized = true;
            Log.d(TAG, "Text recognizer initialized successfully");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize text recognizer: " + e.getMessage(), e);
            isInitialized = false;
            return false;
        }
    }

    /**
     * Detect if image is blurry based on text recognition confidence
     * @param bitmap Input image bitmap
     * @return true if image is blurry (text not readable), false if sharp
     */
    public boolean isBlurry(Bitmap bitmap) {
        if (!isInitialized || textRecognizer == null) {
            Log.w(TAG, "Text recognizer not initialized");
            return false; // Default to not blurry if not initialized
        }

        Log.d(TAG, "Starting text recognition blur detection...");
        try {
            TextRecognitionResult result = detectTextWithConfidence(bitmap);
            Log.d(TAG, String.format("Text recognition result: %s (confidence: %.3f, words: %d/%d)", 
                    result.isReadable ? "READABLE" : "BLURRY", 
                    result.averageConfidence, result.readableWords, result.totalWords));
            
            boolean isBlurry = !result.isReadable;
            Log.d(TAG, String.format("Blur detection result: %s (returning %.1f)", 
                    isBlurry ? "BLURRY" : "SHARP", isBlurry ? 1.0 : 0.0));
            
            return isBlurry;
        } catch (Exception e) {
            Log.e(TAG, "Error during text recognition: " + e.getMessage(), e);
            return false; // Default to not blurry on error
        }
    }

    /**
     * Detect blur with detailed confidence scores
     * @param bitmap Input image bitmap
     * @return Map with isBlur, textConfidence, wordCount, and readableWords
     */
    public java.util.Map<String, Object> detectBlurWithConfidence(Bitmap bitmap) {
        java.util.Map<String, Object> result = new java.util.HashMap<>();
        
        if (!isInitialized || textRecognizer == null) {
            Log.w(TAG, "Text recognizer not initialized");
            result.put("isBlur", false);
            result.put("textConfidence", 0.0);
            result.put("wordCount", 0);
            result.put("readableWords", 0);
            result.put("error", "Text recognizer not initialized");
            return result;
        }

        try {
            TextRecognitionResult recognitionResult = detectTextWithConfidence(bitmap);
            
            result.put("isBlur", !recognitionResult.isReadable);
            result.put("textConfidence", recognitionResult.averageConfidence);
            result.put("wordCount", recognitionResult.totalWords);
            result.put("readableWords", recognitionResult.readableWords);
            result.put("hasText", recognitionResult.totalWords > 0);
            
            Log.d(TAG, String.format("Text Recognition Result - Readable: %s, Confidence: %.3f, Words: %d/%d",
                    recognitionResult.isReadable, recognitionResult.averageConfidence, 
                    recognitionResult.readableWords, recognitionResult.totalWords));
            
            return result;
            
        } catch (Exception e) {
            Log.e(TAG, "Error during text recognition: " + e.getMessage(), e);
            result.put("isBlur", false);
            result.put("textConfidence", 0.0);
            result.put("wordCount", 0);
            result.put("readableWords", 0);
            result.put("error", e.getMessage());
            return result;
        }
    }

    /**
     * Detect text in image with confidence analysis
     * @param bitmap Input image bitmap
     * @return TextRecognitionResult with confidence and readability info
     */
    private TextRecognitionResult detectTextWithConfidence(Bitmap bitmap) {
        InputImage image = InputImage.fromBitmap(bitmap, 0);
        
        // Use CountDownLatch for synchronous operation
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<TextRecognitionResult> resultRef = new AtomicReference<>();
        AtomicBoolean hasError = new AtomicBoolean(false);
        
        textRecognizer.process(image)
            .addOnSuccessListener(visionText -> {
                try {
                    TextRecognitionResult result = analyzeTextConfidence(visionText);
                    resultRef.set(result);
                } catch (Exception e) {
                    Log.e(TAG, "Error analyzing text confidence: " + e.getMessage(), e);
                    hasError.set(true);
                } finally {
                    latch.countDown();
                }
            })
            .addOnFailureListener(e -> {
                Log.e(TAG, "Text recognition failed: " + e.getMessage(), e);
                hasError.set(true);
                latch.countDown();
            });

        try {
            // Wait for result with timeout
            boolean completed = latch.await(TIMEOUT_MS, TimeUnit.MILLISECONDS);
            if (!completed) {
                Log.w(TAG, "Text recognition timed out after " + TIMEOUT_MS + "ms");
                return new TextRecognitionResult(false, 0.0, 0, 0);
            }
            
            if (hasError.get()) {
                return new TextRecognitionResult(false, 0.0, 0, 0);
            }
            
            TextRecognitionResult result = resultRef.get();
            return result != null ? result : new TextRecognitionResult(false, 0.0, 0, 0);
            
        } catch (InterruptedException e) {
            Log.e(TAG, "Text recognition interrupted: " + e.getMessage(), e);
            Thread.currentThread().interrupt();
            return new TextRecognitionResult(false, 0.0, 0, 0);
        }
    }

    /**
     * Analyze text confidence and determine readability
     * @param visionText Recognized text from ML Kit
     * @return TextRecognitionResult with analysis
     */
    private TextRecognitionResult analyzeTextConfidence(Text visionText) {
        int totalWords = 0;
        int readableWords = 0;
        double totalConfidence = 0.0;
        StringBuilder allText = new StringBuilder();
        StringBuilder readableText = new StringBuilder();
        
        Log.d(TAG, "=== Text Recognition Analysis ===");
        
        for (Text.TextBlock block : visionText.getTextBlocks()) {
            Log.d(TAG, "Text Block: " + block.getText());
            allText.append(block.getText()).append(" ");
            
            for (Text.Line line : block.getLines()) {
                Log.d(TAG, "  Line: " + line.getText());
                
                for (Text.Element element : line.getElements()) {
                    String text = element.getText().trim();
                    if (!text.isEmpty()) {
                        totalWords++;
                        
                        // Get confidence score (ML Kit doesn't provide per-word confidence)
                        // We'll use a heuristic based on text quality
                        double confidence = estimateWordConfidence(element, text);
                        totalConfidence += confidence;
                        
                        Log.d(TAG, String.format("    Word: '%s' | Confidence: %.3f | Readable: %s", 
                                text, confidence, confidence >= MIN_WORD_CONFIDENCE ? "YES" : "NO"));
                        
                        if (confidence >= MIN_WORD_CONFIDENCE) {
                            // Optional dictionary check for enhanced validation
                            if (!useDictionaryCheck || isInDictionary(text)) {
                                readableWords++;
                                readableText.append(text).append(" ");
                                Log.d(TAG, String.format("      ✓ Added to readable words (dict check: %s)", 
                                        useDictionaryCheck ? (isInDictionary(text) ? "PASS" : "FAIL") : "SKIPPED"));
                            } else {
                                Log.d(TAG, "      ✗ Failed dictionary check");
                            }
                        }
                    }
                }
            }
        }
        
        double averageConfidence = totalWords > 0 ? totalConfidence / totalWords : 0.0;
        
            // Image is readable if:
            // 1. We have text AND
            // 2. At least 60% of words are readable OR average confidence is high
            boolean isReadable = totalWords > 0 && 
                               (readableWords >= Math.max(1, totalWords * AT_LEAST_N_PERCENT_OF_WORDS_ARE_READABLE) || averageConfidence >= AT_LEAST_N_PERCENT_OF_AVERAGE_CONFIDENCE);
        
        // Log summary
        Log.d(TAG, "=== Text Recognition Summary ===");
        Log.d(TAG, "All detected text: '" + allText.toString().trim() + "'");
        Log.d(TAG, "Readable text: '" + readableText.toString().trim() + "'");
        Log.d(TAG, String.format("Total words: %d | Readable words: %d | Average confidence: %.3f", 
                totalWords, readableWords, averageConfidence));
        Log.d(TAG, String.format("Readability threshold: %.1f%% (need %.1f%%) | Result: %s", 
                totalWords > 0 ? (readableWords * 100.0 / totalWords) : 0.0,
                AT_LEAST_N_PERCENT_OF_WORDS_ARE_READABLE * 100.0,
                isReadable ? "READABLE" : "BLURRY"));
        Log.d(TAG, "================================");
        
        return new TextRecognitionResult(isReadable, averageConfidence, totalWords, readableWords);
    }

    /**
     * Estimate word confidence based on element properties
     * @param element Text element from ML Kit
     * @param text The actual text content
     * @return Estimated confidence score
     */
    private double estimateWordConfidence(Text.Element element, String text) {
        // Base confidence estimation based on text characteristics
        double confidence = 0.55; // Start with 55% base confidence
        
        // Check text length (very short or very long words might be less reliable)
        if (text.length() >= 3 && text.length() <= 15) {
            confidence += 0.2;
        }
        
        // Check for common patterns that indicate good recognition
        if (text.matches("[a-zA-Z0-9\\s\\-\\.]+")) {
            confidence += 0.15;
        }
        
        // Check for mixed case (indicates good character recognition)
        if (text.matches(".*[a-z].*") && text.matches(".*[A-Z].*")) {
            confidence += 0.1;
        }
        
        // Check for numbers (often well recognized)
        if (text.matches(".*\\d.*")) {
            confidence += 0.1;
        }
        
        // Penalize for special characters that might indicate poor recognition
        if (text.matches(".*[^a-zA-Z0-9\\s\\-\\.].*")) {
            confidence -= 0.1;
        }
        
        return Math.max(0.0, Math.min(1.0, confidence));
    }

    /**
     * Check if word is in common dictionary
     * @param word Word to check
     * @return true if word is in dictionary
     */
    private boolean isInDictionary(String word) {
        if (commonWords == null || word == null) {
            return false;
        }
        
        String cleanWord = word.toLowerCase().replaceAll("[^a-zA-Z]", "");
        return commonWords.contains(cleanWord);
    }

    /**
     * Initialize common English words for dictionary check
     */
    private void initializeCommonWords() {
        commonWords = new HashSet<>();
        
        // Add common English words (sample set - can be expanded)
        String[] words = {
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
        };
        
        for (String word : words) {
            commonWords.add(word);
        }
    }

    /**
     * Enable or disable dictionary check for enhanced readability detection
     * @param enable true to enable dictionary check
     */
    public void setDictionaryCheckEnabled(boolean enable) {
        this.useDictionaryCheck = enable;
        Log.d(TAG, "Dictionary check " + (enable ? "enabled" : "disabled"));
    }

    /**
     * Check if text recognizer is properly initialized
     * @return true if initialized
     */
    public boolean isInitialized() {
        return isInitialized;
    }

    /**
     * Clean up resources
     */
    public void close() {
        if (textRecognizer != null) {
            textRecognizer.close();
            textRecognizer = null;
        }
        isInitialized = false;
    }

    /**
     * Result class for text recognition analysis
     */
    private static class TextRecognitionResult {
        final boolean isReadable;
        final double averageConfidence;
        final int totalWords;
        final int readableWords;

        TextRecognitionResult(boolean isReadable, double averageConfidence, int totalWords, int readableWords) {
            this.isReadable = isReadable;
            this.averageConfidence = averageConfidence;
            this.totalWords = totalWords;
            this.readableWords = readableWords;
        }
    }
}

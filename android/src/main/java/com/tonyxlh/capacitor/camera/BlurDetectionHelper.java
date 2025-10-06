package com.tonyxlh.capacitor.camera;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.util.Log;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.ObjectDetection;
import com.google.mlkit.vision.objects.ObjectDetector;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.Canvas;
import org.tensorflow.lite.DataType;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Comprehensive Blur Detection Helper
 * Implements the 3-step pipeline: Object Detection -> Text Detection -> Full-Image Blur Detection
 */
public class BlurDetectionHelper {
    private static final String TAG = "BlurDetectionHelper";
    private static final String MODEL_FILENAME = "blur_detection_model.tflite";
    private static int INPUT_SIZE = 600; // Will be updated based on actual model input size
    private static final int NUM_CLASSES = 2; // blur, sharp
    
    // Timeout settings
    private static final int TIMEOUT_MS = 5000; // 5 second timeout
    
    // Text recognition settings
    private static final double MIN_WORD_CONFIDENCE = 0.8; // 80% confidence threshold
    private static final double AT_LEAST_N_PERCENT_OF_WORDS_ARE_READABLE = 0.6; // 60% of words are readable
    private static final double AT_LEAST_N_PERCENT_OF_AVERAGE_CONFIDENCE = 0.85; // 85% of average confidence

    // Method based confidence threshold
    private static final double MIN_SHARP_CONFIDENCE_FOR_OBJECT_DETECTION = 0.75; // 75% confidence threshold
    private static final double MIN_SHARP_CONFIDENCE_FOR_TEXT_DETECTION = 0.15; // 15% confidence threshold
    private static final double MIN_SHARP_CONFIDENCE_FOR_FULL_IMAGE = 0.75; // 70% confidence threshold
    
    // TFLite components
    private Interpreter tflite;
    private ImageProcessor imageProcessor;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    
    // ML Kit components
    private ObjectDetector objectDetector;
    private TextRecognizer textRecognizer;
    
    // Configuration flags
    private boolean isInitialized = false;
    private boolean useObjectDetection = true;
    private boolean useTextRecognition = true;
    private boolean useDictionaryCheck = false;
    
    // Dictionary for text validation
    private Set<String> commonWords;
    
    // Executor for parallel processing
    private ExecutorService executorService;


    public BlurDetectionHelper() {
        // Initialize image processor for MobileNetV2 preprocessing with aspect ratio preservation
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(INPUT_SIZE, INPUT_SIZE)) // This preserves aspect ratio by cropping/padding
                .build();
        
        // Initialize common words dictionary
        initializeCommonWords();
        
        // Initialize executor service for parallel processing
        executorService = Executors.newFixedThreadPool(3);
    }

    /**
     * Initialize all components (TFLite, Object Detection, Text Recognition)
     * @param context Android context to access assets
     * @return true if initialization successful
     */
    public boolean initialize(Context context) {
        boolean tfliteInitialized = false;
        boolean objectDetectorInitialized = false;
        boolean textRecognizerInitialized = false;
        
        // Initialize TFLite model
        try {
            // Load model from assets
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(context, MODEL_FILENAME);
            
            // Configure interpreter options for better performance
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4); // Use multiple threads for better performance
            
            // Try to use GPU acceleration if available
            try {
                options.setUseXNNPACK(true);
            } catch (Exception e) {
                // XNNPACK not available, using CPU
            }
            
            tflite = new Interpreter(tfliteModel, options);
            
            // Initialize input and output buffers
            inputImageBuffer = new TensorImage(tflite.getInputTensor(0).dataType());
            outputProbabilityBuffer = TensorBuffer.createFixedSize(
                    tflite.getOutputTensor(0).shape(),
                    tflite.getOutputTensor(0).dataType()
            );
            
            // Update INPUT_SIZE based on actual model input shape
            int[] inputShape = tflite.getInputTensor(0).shape();
            if (inputShape.length >= 3) {
                int modelInputSize = inputShape[1]; // height dimension
                if (modelInputSize != INPUT_SIZE) {
                    INPUT_SIZE = modelInputSize;
                    
                    // Recreate image processor with correct size and aspect ratio preservation
                    imageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(INPUT_SIZE, INPUT_SIZE)) // Preserves aspect ratio
                        .build();
                }
            }
            
            tfliteInitialized = true;
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize TFLite model: " + e.getMessage());
        }
        
        // Initialize Object Detector
        try {
            ObjectDetectorOptions detectorOptions = new ObjectDetectorOptions.Builder()
                    .setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
                    .enableClassification()  // Enable classification to get confidence scores
                    .build();
            
            objectDetector = ObjectDetection.getClient(detectorOptions);
            objectDetectorInitialized = true;
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize Object Detector: " + e.getMessage());
        }
        
        // Initialize Text Recognizer
        try {
            textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
            textRecognizerInitialized = true;
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize Text Recognizer: " + e.getMessage());
        }
        
        // Set initialization status
        isInitialized = tfliteInitialized || objectDetectorInitialized || textRecognizerInitialized;
        
        Log.d(TAG, "Initialization status - TFLite: " + tfliteInitialized + 
                   ", Object Detection: " + objectDetectorInitialized + 
                   ", Text Recognition: " + textRecognizerInitialized);
        
        return isInitialized;
    }

    /**
     * Main blur detection method implementing the 3-step pipeline
     * @param bitmap Input image bitmap
     * @return Blur confidence score (0.0 = sharp, 1.0 = very blurry)
     */
    public double detectBlur(Bitmap bitmap) {
        Map<String, Object> result = detectBlurWithConfidence(bitmap);
        Boolean isBlur = (Boolean) result.get("isBlur");
        return (isBlur != null && isBlur) ? 1.0 : 0.0;
    }
    
    /**
     * Detect blur with detailed confidence scores using the 3-step pipeline
     * @param bitmap Input image bitmap
     * @return Map with comprehensive blur detection results
     */
    public Map<String, Object> detectBlurWithConfidence(Bitmap bitmap) {
        Map<String, Object> result = new HashMap<>();
        
        if (!isInitialized) {
            result.put("isBlur", false);
            result.put("method", "none");
            result.put("error", "Blur detector not initialized");
            return result;
        }
        
        try {
            // Step 1: Object Detection (Preferred Path)
            if (useObjectDetection && objectDetector != null) {
                Log.d(TAG, "Step 1: Trying Object Detection");
                List<DetectedObject> objects = detectObjects(bitmap);
                
                if (!objects.isEmpty()) {
                    Log.d(TAG, "Objects detected: " + objects.size());
                    
                    // Process ROIs from detected objects
                    List<Map<String, Object>> roiResults = new ArrayList<>();
                    boolean anyBlurry = false;
                    
                    for (DetectedObject object : objects) {
                        Rect boundingBox = object.getBoundingBox();
                        if (boundingBox != null) {
                            // Crop ROI from original image
                            Bitmap roi = cropBitmap(bitmap, boundingBox);
                            if (roi != null) {
                                // Run TFLite blur detection on ROI
                                Map<String, Object> roiResult = detectBlurWithTFLiteConfidence(roi, MIN_SHARP_CONFIDENCE_FOR_OBJECT_DETECTION);

                                Log.d(TAG, "Object Detection ROI Result isBlur: " + roiResult.get("isBlur"));
                                roiResult.put("boundingBox", boundingBox);
                                roiResults.add(roiResult);
                                
                                Boolean isBlur = (Boolean) roiResult.get("isBlur");
                                if (isBlur != null && isBlur) {
                                    anyBlurry = true;
                                }
                            }
                        }
                    }
                    
                    result.put("method", "object_detection");
                    result.put("isBlur", anyBlurry);
                    result.put("roiResults", roiResults);
                    result.put("objectCount", objects.size());
                    return result;
                }
            }
            
            // Step 2: Text Detection (Fallback if Object Not Found)
            if (useTextRecognition && textRecognizer != null) {
                Log.d(TAG, "Step 2: Trying Text Detection");
                TextRecognitionResult textResult = detectTextWithBoundingBoxes(bitmap);
                
                if (textResult.totalWords > 0) {
                    Log.d(TAG, "Text detected: " + textResult.totalWords + " words");
                    
                    // Combine all detected text areas into a single bounding box
                    List<Rect> topTextAreas = combineAllTextAreas(textResult.boundingBoxes);
                    
                    if (!topTextAreas.isEmpty()) {
                        // Process ROIs from text areas
                        List<Map<String, Object>> roiResults = processROIsInParallel(bitmap, topTextAreas);
                        
                        // Check if any ROI is blurry
                        boolean anyBlurry = false;
                        for (Map<String, Object> roiResult : roiResults) {
                            Boolean isBlur = (Boolean) roiResult.get("isBlur");
                            if (isBlur != null && isBlur) {
                                anyBlurry = true;
                                break;
                            }
                        }
                        
                        result.put("method", "text_detection");
                        result.put("isBlur", anyBlurry);
                        result.put("roiResults", roiResults);
                        result.put("textConfidence", textResult.averageConfidence);
                        result.put("wordCount", textResult.totalWords);
                        result.put("readableWords", textResult.readableWords);
                        return result;
                    }
                }
            }
            
            // Step 3: Full-Image Blur Detection (Final Fallback)
            Log.d(TAG, "Step 3: Using Full-Image Blur Detection");
            return detectBlurWithTFLiteConfidence(bitmap, MIN_SHARP_CONFIDENCE_FOR_FULL_IMAGE);
            
        } catch (Exception e) {
            Log.e(TAG, "Error in blur detection pipeline: " + e.getMessage());
            result.put("isBlur", false);
            result.put("method", "error");
            result.put("error", e.getMessage());
            return result;
        }
    }

    /**
     * Detect blur in image using TFLite model only
     * @param bitmap Input image bitmap
     * @return Blur confidence score (0.0 = sharp, 1.0 = very blurry)
     */
    public double detectBlurWithTFLite(Bitmap bitmap) {
        if (!isInitialized || tflite == null) {
            double laplacianScore = calculateLaplacianBlurScore(bitmap);
            return laplacianScore;
        }
        

        try {
            // Use the original bitmap directly (no image enhancement)
            Bitmap processedBitmap = bitmap;
            
            // Preprocess image for model (resize and potential enhancement)
            inputImageBuffer.load(processedBitmap);
            inputImageBuffer = imageProcessor.process(inputImageBuffer);
            
            // Ensure black padding for better accuracy (matches iOS implementation)
            ensureBlackPadding(inputImageBuffer);

            // Get tensor buffer
            ByteBuffer tensorBuffer = inputImageBuffer.getBuffer();

            // Check if we need normalization based on data types
            ByteBuffer inferenceBuffer;
            if (inputImageBuffer.getDataType() == DataType.UINT8 && tflite.getInputTensor(0).dataType() == DataType.FLOAT32) {
                inferenceBuffer = normalizeImageBuffer(tensorBuffer);
            } else if (inputImageBuffer.getDataType() == DataType.FLOAT32) {
                // Check if values are in [0,1] range or [0,255] range
                inferenceBuffer = checkAndNormalizeFloat32Buffer(tensorBuffer);
            } else {
                inferenceBuffer = tensorBuffer;
            }

            // Run inference
            tflite.run(inferenceBuffer, outputProbabilityBuffer.getBuffer().rewind());

            // Get output probabilities
            float[] probabilities = outputProbabilityBuffer.getFloatArray();
            
            // probabilities[0] = blur probability, probabilities[1] = sharp probability
            double blurConfidence = probabilities.length > 0 ? probabilities[0] : 0.0;
            double sharpConfidence = probabilities.length > 1 ? probabilities[1] : 0.0;

            // Determine if image is blurry using TFLite confidence
            boolean isBlur = sharpConfidence < MIN_SHARP_CONFIDENCE_FOR_FULL_IMAGE;
            
            
            // Return 1.0 for blur, 0.0 for sharp (to maintain double return type)
            return isBlur ? 1.0 : 0.0;
            
        } catch (Exception e) {
            // Fallback to Laplacian algorithm
            double laplacianScore = calculateLaplacianBlurScore(bitmap);
            boolean isBlur = laplacianScore < 150;
            return isBlur ? 1.0 : 0.0;
        }
    }

    /**
     * Check if float32 buffer needs normalization and normalize if needed
     * @param float32Buffer Input buffer with float32 pixel values
     * @return Normalized float32 buffer (values in [0,1] range)
     */
    private ByteBuffer checkAndNormalizeFloat32Buffer(ByteBuffer float32Buffer) {
        float32Buffer.rewind();
        FloatBuffer floatBuffer = float32Buffer.asFloatBuffer();
        
        // Sample a few values to check if they're in [0,1] or [0,255] range
        float maxSample = 0.0f;
        int sampleCount = Math.min(100, floatBuffer.remaining());
        
        for (int i = 0; i < sampleCount; i++) {
            float value = Math.abs(floatBuffer.get());
            maxSample = Math.max(maxSample, value);
        }
        
        // If max value is > 1.5, assume it's in [0,255] range and needs normalization
        if (maxSample > 1.5f) {
            return normalizeFloat32Buffer(float32Buffer);
        } else {
            float32Buffer.rewind();
            return float32Buffer;
        }
    }

    /**
     * Normalize float32 buffer from [0,255] to [0,1]
     * @param float32Buffer Input buffer with float32 pixel values in [0,255] range
     * @return Normalized float32 buffer
     */
         private ByteBuffer normalizeFloat32Buffer(ByteBuffer float32Buffer) {
         int pixelCount = INPUT_SIZE * INPUT_SIZE * 3;
         ByteBuffer normalizedBuffer = ByteBuffer.allocateDirect(pixelCount * 4);
         normalizedBuffer.order(ByteOrder.nativeOrder());
         FloatBuffer normalizedFloats = normalizedBuffer.asFloatBuffer();
         
         float32Buffer.rewind();
         FloatBuffer sourceFloats = float32Buffer.asFloatBuffer();
         
         while (sourceFloats.hasRemaining() && normalizedFloats.hasRemaining()) {
             float pixelValue = sourceFloats.get();
             float normalizedValue = pixelValue / 255.0f;
             normalizedFloats.put(normalizedValue);
         }
         
         normalizedBuffer.rewind();
         return normalizedBuffer;
     }

    /**
     * Ensure black padding in the processed image buffer for better accuracy
     * @param tensorImage Processed tensor image
     */
    private void ensureBlackPadding(TensorImage tensorImage) {
        ByteBuffer buffer = tensorImage.getBuffer();
        DataType dataType = tensorImage.getDataType();
        
        if (dataType == DataType.FLOAT32) {
            // For float32, ensure padding areas are 0.0 (black)
            FloatBuffer floatBuffer = buffer.asFloatBuffer();
            int totalPixels = INPUT_SIZE * INPUT_SIZE * 3;
            
            // Check if we need to fill with zeros (black)
            for (int i = 0; i < totalPixels; i++) {
                if (floatBuffer.get(i) < 0.001f) { // Near zero values
                    floatBuffer.put(i, 0.0f); // Ensure exact zero (black)
                }
            }
        } else if (dataType == DataType.UINT8) {
            // For uint8, ensure padding areas are 0 (black)
            buffer.rewind();
            while (buffer.hasRemaining()) {
                byte value = buffer.get();
                if (value == 0) {
                    buffer.put(buffer.position() - 1, (byte) 0); // Ensure exact zero
                }
            }
        }
    }

    /**
     * Normalize image buffer from uint8 [0,255] to float32 [0,1]
     * @param uint8Buffer Input buffer with uint8 pixel values
     * @return Normalized float32 buffer
     */
         private ByteBuffer normalizeImageBuffer(ByteBuffer uint8Buffer) {
         // Create float32 buffer with proper size and byte order
         int pixelCount = INPUT_SIZE * INPUT_SIZE * 3;
         ByteBuffer float32Buffer = ByteBuffer.allocateDirect(pixelCount * 4); // 4 bytes per float
         float32Buffer.order(ByteOrder.nativeOrder());
         FloatBuffer floatBuffer = float32Buffer.asFloatBuffer();
         
         // Reset uint8 buffer position
         uint8Buffer.rewind();
         
         // Convert each uint8 pixel to normalized float32
         while (uint8Buffer.hasRemaining() && floatBuffer.hasRemaining()) {
             int pixelValue = uint8Buffer.get() & 0xFF; // Convert to unsigned int
             float normalizedValue = pixelValue / 255.0f; // Normalize to [0,1]
             floatBuffer.put(normalizedValue);
         }
         
         float32Buffer.rewind();
         return float32Buffer;
     }

    /**
     * Fallback Laplacian blur detection (from original implementation)
     */
    private double calculateLaplacianBlurScore(Bitmap bitmap) {
        if (bitmap == null) return 0.0;
        
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        
        // Convert to grayscale for better blur detection
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        
        double[] grayscale = new double[width * height];
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;
            grayscale[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
        
        // Apply Laplacian kernel for edge detection
        double variance = 0.0;
        int count = 0;
        
        // Sample every 4th pixel for performance
        int step = 4;
        for (int y = step; y < height - step; y += step) {
            for (int x = step; x < width - step; x += step) {
                int idx = y * width + x;
                
                // 3x3 Laplacian kernel
                double laplacian = 
                    -grayscale[idx - width - 1] - grayscale[idx - width] - grayscale[idx - width + 1] +
                    -grayscale[idx - 1] + 8 * grayscale[idx] - grayscale[idx + 1] +
                    -grayscale[idx + width - 1] - grayscale[idx + width] - grayscale[idx + width + 1];
                
                variance += laplacian * laplacian;
                count++;
            }
        }
        
        return count > 0 ? variance / count : 0.0;
    }


    /**
     * Detect blur with detailed confidence scores using TFLite only
     * @param bitmap Input image bitmap
     * @return Map with isBlur, blurConfidence, and sharpConfidence
     */
    public java.util.Map<String, Object> detectBlurWithTFLiteConfidence(Bitmap bitmap, double minSharpConfidence) {
        java.util.Map<String, Object> result = new java.util.HashMap<>();
        
        if (!isInitialized || tflite == null) {
            double laplacianScore = calculateLaplacianBlurScore(bitmap);
            boolean isBlur = laplacianScore < 150;
            double normalizedScore = Math.max(0.0, Math.min(1.0, laplacianScore / 300.0));
            double sharpConfidence = normalizedScore;
            double blurConfidence = 1.0 - normalizedScore;
            
            result.put("method", "laplacian");
            result.put("isBlur", isBlur);
            result.put("blurConfidence", blurConfidence);
            result.put("sharpConfidence", sharpConfidence);
            result.put("laplacianScore", laplacianScore);
            result.put("hasText", false);
            result.put("boundingBoxes", new java.util.ArrayList<>());
            return result;
        }

        try {
            // Use the original bitmap directly (no image enhancement)
            Bitmap processedBitmap = bitmap;
            
            // Preprocess image for model (resize and potential enhancement)
            inputImageBuffer.load(processedBitmap);
            inputImageBuffer = imageProcessor.process(inputImageBuffer);
            
            // Ensure black padding for better accuracy (matches iOS implementation)
            ensureBlackPadding(inputImageBuffer);

            // Get tensor buffer
            ByteBuffer tensorBuffer = inputImageBuffer.getBuffer();

            // Check if we need normalization based on data types
            ByteBuffer inferenceBuffer;
            if (inputImageBuffer.getDataType() == DataType.UINT8 && tflite.getInputTensor(0).dataType() == DataType.FLOAT32) {
                inferenceBuffer = normalizeImageBuffer(tensorBuffer);
            } else if (inputImageBuffer.getDataType() == DataType.FLOAT32) {
                // Check if values are in [0,1] range or [0,255] range
                inferenceBuffer = checkAndNormalizeFloat32Buffer(tensorBuffer);
            } else {
                inferenceBuffer = tensorBuffer;
            }

            // Run inference
            tflite.run(inferenceBuffer, outputProbabilityBuffer.getBuffer().rewind());

            // Get output probabilities
            float[] probabilities = outputProbabilityBuffer.getFloatArray();
            
            // probabilities[0] = blur probability, probabilities[1] = sharp probability
            double blurConfidence = probabilities.length > 0 ? probabilities[0] : 0.0;
            double sharpConfidence = probabilities.length > 1 ? probabilities[1] : 0.0;

            Log.d(TAG, "TFLite Blur confidence: " + blurConfidence + " Sharp confidence: " + sharpConfidence);

            // Determine if image is blurry using TFLite confidence
            boolean isBlur = sharpConfidence < minSharpConfidence;
            
            Log.d(TAG, "TFLite Blur detection isBlur: " + isBlur);
            
            result.put("isBlur", isBlur);
            result.put("method", "tflite");
            result.put("blurConfidence", blurConfidence);
            result.put("sharpConfidence", sharpConfidence);
            result.put("boundingBoxes", new java.util.ArrayList<>());
            return result;
            
        } catch (Exception e) {
            // Fallback to Laplacian algorithm
            double laplacianScore = calculateLaplacianBlurScore(bitmap);
            boolean isBlur = laplacianScore < 150;
            double normalizedScore = Math.max(0.0, Math.min(1.0, laplacianScore / 300.0));
            double sharpConfidence = normalizedScore;
            double blurConfidence = 1.0 - normalizedScore;
            
            result.put("isBlur", isBlur);
            result.put("blurConfidence", blurConfidence);
            result.put("sharpConfidence", sharpConfidence);
            result.put("boundingBoxes", new java.util.ArrayList<>());
            return result;
        }
    }

    /**
     * Check if image is blurry
     * @param bitmap Input image
     * @return true if image is blurry, false if sharp
     */
    public boolean isBlurry(Bitmap bitmap) {
        double result = detectBlur(bitmap);
        return result == 1.0;
    }

    /**
     * Get blur percentage (0-100%) - Deprecated, use isBlurry() instead
     * @param bitmap Input image
     * @return Blur percentage where 0% = sharp, 100% = very blurry
     */
    @Deprecated
    public double getBlurPercentage(Bitmap bitmap) {
        // Convert boolean result to percentage for backward compatibility
        return isBlurry(bitmap) ? 100.0 : 0.0;
    }


    /**
     * Detect objects in the image using ML Kit Object Detection
     * @param bitmap Input image
     * @return List of detected objects
     */
    private List<DetectedObject> detectObjects(Bitmap bitmap) {
        if (objectDetector == null) {
            return new ArrayList<>();
        }
        
        InputImage image = InputImage.fromBitmap(bitmap, 0);
        
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<List<DetectedObject>> resultRef = new AtomicReference<>(new ArrayList<>());
        
        objectDetector.process(image)
            .addOnSuccessListener(detectedObjects -> {
                resultRef.set(detectedObjects);
                latch.countDown();
            })
            .addOnFailureListener(e -> {
                Log.e(TAG, "Object detection failed: " + e.getMessage());
                latch.countDown();
            });
        
        try {
            latch.await(TIMEOUT_MS, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        return resultRef.get();
    }
    
    /**
     * Detect text in image with bounding boxes
     * @param bitmap Input image
     * @return TextRecognitionResult with bounding boxes
     */
    private TextRecognitionResult detectTextWithBoundingBoxes(Bitmap bitmap) {
        if (textRecognizer == null) {
            return new TextRecognitionResult(false, 0.0, 0, 0, new ArrayList<>());
        }
        
        InputImage image = InputImage.fromBitmap(bitmap, 0);
        
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<TextRecognitionResult> resultRef = new AtomicReference<>();
        AtomicBoolean hasError = new AtomicBoolean(false);
        
        textRecognizer.process(image)
            .addOnSuccessListener(visionText -> {
                try {
                    TextRecognitionResult result = analyzeTextConfidence(visionText);
                    resultRef.set(result);
                } catch (Exception e) {
                    hasError.set(true);
                } finally {
                    latch.countDown();
                }
            })
            .addOnFailureListener(e -> {
                hasError.set(true);
                latch.countDown();
            });
        
        try {
            boolean completed = latch.await(TIMEOUT_MS, TimeUnit.MILLISECONDS);
            if (!completed || hasError.get()) {
                return new TextRecognitionResult(false, 0.0, 0, 0, new ArrayList<>());
            }
            
            TextRecognitionResult result = resultRef.get();
            return result != null ? result : new TextRecognitionResult(false, 0.0, 0, 0, new ArrayList<>());
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return new TextRecognitionResult(false, 0.0, 0, 0, new ArrayList<>());
        }
    }
    
    /**
     * Analyze text confidence and extract bounding boxes
     * @param visionText Recognized text from ML Kit
     * @return TextRecognitionResult with analysis
     */
    private TextRecognitionResult analyzeTextConfidence(Text visionText) {
        int totalWords = 0;
        int readableWords = 0;
        double totalConfidence = 0.0;
        List<Rect> boundingBoxes = new ArrayList<>();
        
        for (Text.TextBlock block : visionText.getTextBlocks()) {
            Rect blockBoundingBox = block.getBoundingBox();
            if (blockBoundingBox != null) {
                boundingBoxes.add(blockBoundingBox);
            }
            
            for (Text.Line line : block.getLines()) {
                for (Text.Element element : line.getElements()) {
                    String text = element.getText().trim();
                    if (!text.isEmpty()) {
                        totalWords++;
                        
                        // Estimate word confidence
                        double confidence = estimateWordConfidence(element, text);
                        totalConfidence += confidence;
                        
                        if (confidence >= MIN_WORD_CONFIDENCE) {
                            if (!useDictionaryCheck || isInDictionary(text)) {
                                readableWords++;
                            }
                        }
                    }
                }
            }
        }
        
        double averageConfidence = totalWords > 0 ? totalConfidence / totalWords : 0.0;
        
        // Image is readable if we have text and sufficient readable words or high confidence
        boolean isReadable = totalWords > 0 && 
                           (readableWords >= Math.max(1, totalWords * AT_LEAST_N_PERCENT_OF_WORDS_ARE_READABLE) || 
                            averageConfidence >= AT_LEAST_N_PERCENT_OF_AVERAGE_CONFIDENCE);
        
        return new TextRecognitionResult(isReadable, averageConfidence, totalWords, readableWords, boundingBoxes);
    }
    
    /**
     * Combine all detected text areas into a single bounding box
     * @param boundingBoxes List of bounding boxes
     * @return List containing a single combined bounding box
     */
    private List<Rect> combineAllTextAreas(List<Rect> boundingBoxes) {
        if (boundingBoxes.isEmpty()) {
            return new ArrayList<>();
        }
        
        // Find the minimum and maximum coordinates across all boxes
        int minLeft = Integer.MAX_VALUE;
        int minTop = Integer.MAX_VALUE;
        int maxRight = Integer.MIN_VALUE;
        int maxBottom = Integer.MIN_VALUE;
        
        for (Rect rect : boundingBoxes) {
            if (rect == null) continue;
            minLeft = Math.min(minLeft, rect.left);
            minTop = Math.min(minTop, rect.top);
            maxRight = Math.max(maxRight, rect.right);
            maxBottom = Math.max(maxBottom, rect.bottom);
        }
        
        if (minLeft == Integer.MAX_VALUE) {
            return new ArrayList<>();
        }
        
        Rect combinedRect = new Rect(minLeft, minTop, maxRight, maxBottom);
        
        Log.d(TAG, "Combined " + boundingBoxes.size() + " text areas into single bounding box: " +
                "(" + minLeft + ", " + minTop + ", " + maxRight + ", " + maxBottom + ")");

        // Minimum width and height
        int minWidth = 100;
        int minHeight = 20;
        if (combinedRect.width() < minWidth || combinedRect.height() < minHeight) {
            return new ArrayList<>();
        }
        
        List<Rect> result = new ArrayList<>();
        result.add(combinedRect);
        return result;
    }
    
    /**
     * Process multiple ROIs in parallel for blur detection
     * @param bitmap Original bitmap
     * @param rois List of regions of interest
     * @return List of blur detection results for each ROI
     */
    private List<Map<String, Object>> processROIsInParallel(Bitmap bitmap, List<Rect> rois) {
        List<Map<String, Object>> results = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch latch = new CountDownLatch(rois.size());
        
        for (Rect roi : rois) {
            executorService.execute(() -> {
                try {
                    Bitmap cropped = cropBitmap(bitmap, roi);
                    if (cropped != null) {
                        Map<String, Object> result = detectBlurWithTFLiteConfidence(cropped, MIN_SHARP_CONFIDENCE_FOR_TEXT_DETECTION);
                        result.put("boundingBox", roi);
                        results.add(result);
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Error processing ROI: " + e.getMessage());
                } finally {
                    latch.countDown();
                }
            });
        }
        
        try {
            latch.await(TIMEOUT_MS, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        return results;
    }
    
    /**
     * Crop a bitmap to the specified bounding box
     * @param source Source bitmap
     * @param rect Bounding box to crop
     * @return Cropped bitmap or null if invalid
     */
    private Bitmap cropBitmap(Bitmap source, Rect rect) {
        try {
            // Ensure bounds are within image
            int left = Math.max(0, rect.left);
            int top = Math.max(0, rect.top);
            int right = Math.min(source.getWidth(), rect.right);
            int bottom = Math.min(source.getHeight(), rect.bottom);
            
            int width = right - left;
            int height = bottom - top;
            
            if (width > 0 && height > 0) {
                return Bitmap.createBitmap(source, left, top, width, height);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error cropping bitmap: " + e.getMessage());
        }
        return null;
    }
    
    /**
     * Estimate word confidence based on element properties
     * @param element Text element from ML Kit
     * @param text The actual text content
     * @return Estimated confidence score
     */
    private double estimateWordConfidence(Text.Element element, String text) {
        double confidence = 0.55; // Base confidence
        
        // Text length check
        if (text.length() >= 3 && text.length() <= 15) {
            confidence += 0.2;
        }
        
        // Common patterns check
        if (text.matches("[a-zA-Z0-9\\s\\-\\.]+")) {
            confidence += 0.15;
        }
        
        // Mixed case check
        if (text.matches(".*[a-z].*") && text.matches(".*[A-Z].*")) {
            confidence += 0.1;
        }
        
        // Numbers check
        if (text.matches(".*\\d.*")) {
            confidence += 0.1;
        }
        
        // Special characters penalty
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
        
        // Add common English words
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
     * Clean up resources
     */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (objectDetector != null) {
            objectDetector.close();
            objectDetector = null;
        }
        if (textRecognizer != null) {
            textRecognizer.close();
            textRecognizer = null;
        }
        if (executorService != null) {
            executorService.shutdown();
            executorService = null;
        }
        isInitialized = false;
    }

    /**
     * Check if blur detector is properly initialized
     */
    public boolean isInitialized() {
        return isInitialized;
    }
    
    /**
     * Enable or disable object detection
     * @param enable true to enable object detection
     */
    public void setObjectDetectionEnabled(boolean enable) {
        this.useObjectDetection = enable;
    }
    
    /**
     * Enable or disable text recognition
     * @param enable true to enable text recognition
     */
    public void setTextRecognitionEnabled(boolean enable) {
        this.useTextRecognition = enable;
    }
    
    /**
     * Enable or disable dictionary check in text recognition
     * @param enable true to enable dictionary check
     */
    public void setDictionaryCheckEnabled(boolean enable) {
        this.useDictionaryCheck = enable;
    }
    
    /**
     * Result class for text recognition analysis
     */
    private static class TextRecognitionResult {
        final boolean isReadable;
        final double averageConfidence;
        final int totalWords;
        final int readableWords;
        final List<Rect> boundingBoxes;

        TextRecognitionResult(boolean isReadable, double averageConfidence, int totalWords, 
                            int readableWords, List<Rect> boundingBoxes) {
            this.isReadable = isReadable;
            this.averageConfidence = averageConfidence;
            this.totalWords = totalWords;
            this.readableWords = readableWords;
            this.boundingBoxes = boundingBoxes;
        }
    }
} 
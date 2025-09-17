package com.tonyxlh.capacitor.camera;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

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

/**
 * TensorFlow Lite Blur Detection Helper
 * Based on MobileNetV2 model trained for blur detection
 */
public class BlurDetectionHelper {
    private static final String TAG = "BlurDetectionHelper";
    private static final String MODEL_FILENAME = "blur_detection_model.tflite";
    private static int INPUT_SIZE = 224; // Will be updated based on actual model input size
    private static final int NUM_CLASSES = 2; // blur, sharp
    
    private Interpreter tflite;
    private ImageProcessor imageProcessor;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    private boolean isInitialized = false;
    
    // Text recognition blur detection helper
    private TextRecognitionBlurHelper textBlurHelper;
    private boolean useTextRecognition = true;


    public BlurDetectionHelper() {
        // Initialize image processor for MobileNetV2 preprocessing
        // Note: Manual normalization will be done in detectBlur method
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(INPUT_SIZE, INPUT_SIZE))
                .add(new ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .build();
        
        // Initialize text recognition blur helper
        textBlurHelper = new TextRecognitionBlurHelper();
    }

    /**
     * Initialize the TFLite model
     * @param context Android context to access assets
     * @return true if initialization successful
     */
    public boolean initialize(Context context) {
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
            // Use the same data type as the model's input tensor
            inputImageBuffer = new TensorImage(tflite.getInputTensor(0).dataType());
            outputProbabilityBuffer = TensorBuffer.createFixedSize(
                    tflite.getOutputTensor(0).shape(),
                    tflite.getOutputTensor(0).dataType()
            );
            
            // Initialize text recognition helper
            textBlurHelper.initialize(context);
            
            // Update INPUT_SIZE based on actual model input shape
            int[] inputShape = tflite.getInputTensor(0).shape();
            
            // Update INPUT_SIZE based on actual model input shape
            // Expected format: [batch, height, width, channels] or [batch, channels, height, width]
            if (inputShape.length >= 3) {
                // Assume format is [batch, height, width, channels]
                int modelInputSize = inputShape[1]; // height dimension
                if (modelInputSize != INPUT_SIZE) {
                    INPUT_SIZE = modelInputSize;
                    
                    // Recreate image processor with correct size
                    imageProcessor = new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(INPUT_SIZE, INPUT_SIZE))
                        .add(new ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                        .build();
                }
            }
            
            isInitialized = true;
            return true;
            
        } catch (IOException e) {
            isInitialized = false;
            return false;
        } catch (Exception e) {
            isInitialized = false;
            return false;
        }
    }

    /**
     * Detect blur in image using hybrid approach (Text Recognition + TFLite)
     * @param bitmap Input image bitmap
     * @return Blur confidence score (0.0 = sharp, 1.0 = very blurry)
     */
    public double detectBlur(Bitmap bitmap) {
        // First try text recognition if enabled
        if (useTextRecognition && textBlurHelper != null && textBlurHelper.isInitialized()) {
            try {
                java.util.Map<String, Object> textResult = textBlurHelper.detectBlurWithConfidence(bitmap);
                Boolean hasText = (Boolean) textResult.get("hasText");
                
                if (hasText != null && hasText) {
                    // Image contains text, use text recognition result
                    Boolean isBlur = (Boolean) textResult.get("isBlur");
                    Double textConfidence = (Double) textResult.get("textConfidence");
                    Integer wordCount = (Integer) textResult.get("wordCount");
                    Integer readableWords = (Integer) textResult.get("readableWords");
                    
                    
                    if (isBlur != null) {
                        double blurConfidence = isBlur ? 1.0 : 0.0;
                        return blurConfidence;
                    } else {
                    }
                } else {
                }
            } catch (Exception e) {
            }
        }
        
        // Fallback to TFLite model
        return detectBlurWithTFLite(bitmap);
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
            boolean isBlur = (blurConfidence >= 0.99 || sharpConfidence < 0.1);
            
            
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
     * Detect blur with detailed confidence scores using hybrid approach
     * @param bitmap Input image bitmap
     * @return Map with comprehensive blur detection results
     */
    public java.util.Map<String, Object> detectBlurWithConfidence(Bitmap bitmap) {
        java.util.Map<String, Object> result = new java.util.HashMap<>();
        
        // Try text recognition first if enabled
        if (useTextRecognition && textBlurHelper != null && textBlurHelper.isInitialized()) {
            try {
                java.util.Map<String, Object> textResult = textBlurHelper.detectBlurWithConfidence(bitmap);
                Boolean hasText = (Boolean) textResult.get("hasText");
                
                if (hasText != null && hasText) {
                    // Image contains text, use text recognition result
                    Boolean isBlur = (Boolean) textResult.get("isBlur");
                    Double textConfidence = (Double) textResult.get("textConfidence");
                    
                    result.put("method", "text_recognition");
                    result.put("isBlur", isBlur);
                    result.put("textConfidence", textConfidence);
                    result.put("wordCount", textResult.get("wordCount"));
                    result.put("readableWords", textResult.get("readableWords"));
                    result.put("hasText", true);
                    
                    // Set blur/sharp confidence based on text recognition result
                    if (isBlur != null && textConfidence != null) {
                        if (isBlur) {
                            // Image is blurry - high blur confidence, low sharp confidence
                            result.put("blurConfidence", 1.0);
                            result.put("sharpConfidence", 0.0);
                        } else {
                            // Image is sharp - low blur confidence, high sharp confidence
                            result.put("blurConfidence", 1.0 - textConfidence);
                            result.put("sharpConfidence", textConfidence);
                        }
                    } else {
                        // Default values if confidence not available
                        result.put("blurConfidence", isBlur != null && isBlur ? 1.0 : 0.0);
                        result.put("sharpConfidence", isBlur != null && !isBlur ? 1.0 : 0.0);
                    }
                    
                    
                    return result;
                }
            } catch (Exception e) {
            }
        }
        
        // Fallback to TFLite model
        return detectBlurWithTFLiteConfidence(bitmap);
    }

    /**
     * Detect blur with detailed confidence scores using TFLite only
     * @param bitmap Input image bitmap
     * @return Map with isBlur, blurConfidence, and sharpConfidence
     */
    public java.util.Map<String, Object> detectBlurWithTFLiteConfidence(Bitmap bitmap) {
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
            return result;
        }

        try {
            // Use the original bitmap directly (no image enhancement)
            Bitmap processedBitmap = bitmap;
            
            // Preprocess image for model (resize and potential enhancement)
            inputImageBuffer.load(processedBitmap);
            inputImageBuffer = imageProcessor.process(inputImageBuffer);

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
            boolean isBlur = (blurConfidence >= 0.99 || sharpConfidence < 0.1);
            
            
            result.put("isBlur", isBlur);
            result.put("blurConfidence", blurConfidence);
            result.put("sharpConfidence", sharpConfidence);
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
     * Enable or disable text recognition for blur detection
     * @param enable true to enable text recognition, false to use only TFLite
     */
    public void setTextRecognitionEnabled(boolean enable) {
        this.useTextRecognition = enable;
    }

    /**
     * Enable or disable dictionary check in text recognition
     * @param enable true to enable dictionary check
     */
    public void setDictionaryCheckEnabled(boolean enable) {
        if (textBlurHelper != null) {
            textBlurHelper.setDictionaryCheckEnabled(enable);
        }
    }

    /**
     * Check if text recognition is enabled
     * @return true if text recognition is enabled
     */
    public boolean isTextRecognitionEnabled() {
        return useTextRecognition;
    }

    /**
     * Get text recognition helper instance
     * @return TextRecognitionBlurHelper instance
     */
    public TextRecognitionBlurHelper getTextBlurHelper() {
        return textBlurHelper;
    }

    /**
     * Clean up resources
     */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (textBlurHelper != null) {
            textBlurHelper.close();
            textBlurHelper = null;
        }
        isInitialized = false;
    }

    /**
     * Check if TFLite model is properly initialized
     */
    public boolean isInitialized() {
        return isInitialized;
    }


} 
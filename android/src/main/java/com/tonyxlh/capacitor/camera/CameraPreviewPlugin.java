package com.tonyxlh.capacitor.camera;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.CameraState;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.FocusMeteringResult;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.MeteringPointFactory;
import androidx.camera.core.Preview;
import androidx.camera.core.SurfaceOrientedMeteringPointFactory;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.PendingRecording;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoCapture;
import androidx.camera.video.MediaStoreOutputOptions;
import androidx.camera.video.VideoRecordEvent;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.util.Consumer;
import androidx.lifecycle.LifecycleOwner;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;
import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONException;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

@CapacitorPlugin(
  name = "CameraPreview",
  permissions = {
    @Permission(strings = {Manifest.permission.CAMERA}, alias = CameraPreviewPlugin.CAMERA),
  }
)
public class CameraPreviewPlugin extends Plugin {
    // Permission alias constants
    static final String CAMERA = "camera";
    static final String MICROPHONE = "microphone";
    private String callbackID;
    private PreviewView previewView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ExecutorService exec;
    private Camera camera;
    private CameraSelector cameraSelector;
    private ProcessCameraProvider cameraProvider;
    private Preview preview;
    private ImageCapture imageCapture;
    private UseCaseGroup useCaseGroup;
    private ImageAnalysis imageAnalysis;
    private Recorder recorder;
    private Recording currentRecording;
    private PluginCall stopRecordingCall;
    private PluginCall takeSnapshotCall;
    private PluginCall saveFrameCall;
    private int desiredWidth = 1920;
    private int desiredHeight = 1080;
    private CameraState previousCameraStatus;
    private ScanRegion scanRegion;

    static public Bitmap frameTaken;

    // Store the desired JPEG quality, set during initialization
    private int desiredJpegQuality = 95; // Default to high quality
    private BlurDetectionHelper blurDetectionHelper; // TFLite blur detection

    @PluginMethod
    public void initialize(PluginCall call) {
        getActivity().runOnUiThread(new Runnable() {
        @RequiresApi(api = Build.VERSION_CODES.P)
        public void run() {
            // Get quality parameter from initialization, default to 95 if not specified
            if (call.hasOption("quality")) {
                desiredJpegQuality = call.getInt("quality");
                // Ensure quality is within valid range
                desiredJpegQuality = Math.max(1, Math.min(100, desiredJpegQuality));
            }

            previewView = new PreviewView(getContext());
            previewView.setScaleType(PreviewView.ScaleType.FIT_CENTER);
            previewView.setBackgroundColor(Color.TRANSPARENT);
            FrameLayout.LayoutParams cameraPreviewParams = new FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
            );
            ((ViewGroup) bridge.getWebView().getParent()).addView(previewView, cameraPreviewParams);
            bridge.getWebView().bringToFront();

            exec = Executors.newSingleThreadExecutor();
            cameraProviderFuture = ProcessCameraProvider.getInstance(getContext());
            
            // Initialize TFLite blur detection helper
            blurDetectionHelper = new BlurDetectionHelper();
            boolean tfliteInitialized = blurDetectionHelper.initialize(getContext());
            
            cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
                // Auto-optimize for photo capture on initialization with specified quality
                setupUseCases(false); // Always use photo-optimized mode
                call.resolve();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
                call.reject(e.getMessage());
            }
            }, ContextCompat.getMainExecutor(getContext()));
        }
        });
    }

    private void setupUseCases(boolean enableVideo) {
        // Auto-detect maximum resolution for better zoom quality
        Size resolution = getOptimalResolution();

        // Enhanced Preview setup for better quality
        Preview.Builder previewBuilder = new Preview.Builder();
        if (resolution != null) {
        previewBuilder.setTargetResolution(resolution);
        } else {
            // Fallback: let CameraX choose the best resolution automatically
            Log.d("Camera", "Using CameraX auto-resolution selection");
        }
        preview = previewBuilder.build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // Enhanced ImageAnalysis setup
        ImageAnalysis.Builder imageAnalysisBuilder = new ImageAnalysis.Builder();
        if (resolution != null) {
            imageAnalysisBuilder.setTargetResolution(resolution);
        }
        imageAnalysisBuilder.setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setImageQueueDepth(1); // Optimize for latest frame

        imageAnalysis = imageAnalysisBuilder.build();

        // Configure image analysis for better focus performance
        imageAnalysis.setAnalyzer(exec, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                if (takeSnapshotCall != null || saveFrameCall != null) {
                    @SuppressLint("UnsafeOptInUsageError")
                    Bitmap bitmap = BitmapUtils.getBitmap(image);
                    if (scanRegion != null) {
                        int left, top, width, height;
                        if (scanRegion.measuredByPercentage == 0) {
                            left = scanRegion.left;
                            top = scanRegion.top;
                            width = scanRegion.right - scanRegion.left;
                            height = scanRegion.bottom - scanRegion.top;
                        } else {
                            left = (int) ((double) scanRegion.left / 100 * bitmap.getWidth());
                            top = (int) ((double) scanRegion.top / 100 * bitmap.getHeight());
                            width = (int) ((double) scanRegion.right / 100 * bitmap.getWidth() - left);
                            height = (int) ((double) scanRegion.bottom / 100 * bitmap.getHeight() - top);
                        }
                        bitmap = Bitmap.createBitmap(bitmap, left, top, width, height, null, false);
                    }
                    if (takeSnapshotCall != null) {
                        int desiredQuality = 85;
                        if (takeSnapshotCall.hasOption("quality")) {
                            desiredQuality = takeSnapshotCall.getInt("quality");
                        }
                        String base64 = bitmap2Base64(bitmap, desiredQuality);
                        JSObject result = new JSObject();
                        result.put("base64", base64);
                        
                        // Only detect blur if checkBlur option is true
                        boolean shouldCheckBlur = takeSnapshotCall.getBoolean("checkBlur", false);
                        if (shouldCheckBlur) {
                            double confidence = calculateBlurConfidence(bitmap);
                            result.put("confidence", confidence);
                        } else {
                            Log.d("Camera", "Blur detection disabled for performance");
                        }
                        
                        takeSnapshotCall.resolve(result);
                        takeSnapshotCall = null;
                    }
                    if (saveFrameCall != null) {
                        frameTaken = bitmap;
                        JSObject result = new JSObject();
                        result.put("success", true);
                        saveFrameCall.resolve(result);
                        saveFrameCall = null;
                    }
                }
                image.close();
            }
        });

        // Enhanced ImageCapture setup for optimal photo quality with max resolution
        ImageCapture.Builder imageCaptureBuilder = new ImageCapture.Builder();
        if (resolution != null) {
        imageCaptureBuilder.setTargetResolution(resolution);
        }
        imageCaptureBuilder.setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY) // Prioritize quality over speed
        .setJpegQuality(desiredJpegQuality); // Use quality set during initialization

        // Add image stabilization if available (API 28+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        imageCaptureBuilder.setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY);
        }

        imageCapture = imageCaptureBuilder.build();

        if (enableVideo) {
            Quality quality = Quality.HD;
            QualitySelector qualitySelector = QualitySelector.from(quality);
            Recorder.Builder recorderBuilder = new Recorder.Builder();
            recorderBuilder.setQualitySelector(qualitySelector);
            recorder = recorderBuilder.build();
            VideoCapture videoCapture = VideoCapture.withOutput(recorder);
            useCaseGroup = new UseCaseGroup.Builder()
                    .addUseCase(preview)
                    .addUseCase(imageAnalysis)
                    .addUseCase(videoCapture)
                    .build();
        }else{
            useCaseGroup = new UseCaseGroup.Builder()
                    .addUseCase(preview)
                    .addUseCase(imageAnalysis)
                    .addUseCase(imageCapture)
                    .build();
        }
    }

    /**
     * Get the optimal (maximum) resolution supported by the device for better zoom quality
     * Uses high-quality resolution options with CameraX auto-selection fallback
     */
    private Size getOptimalResolution() {
        try {
        // Use high-quality resolution options for better zoom quality
        // These are commonly supported resolutions across Android devices
        int orientation = getContext().getResources().getConfiguration().orientation;
        
        // High-quality resolution options (in order of preference)
        Size[] preferredResolutions;
        
        if (orientation == Configuration.ORIENTATION_PORTRAIT) {
            preferredResolutions = new Size[]{
            new Size(3024, 4032), // 12MP portrait
            new Size(2160, 3840), // 4K portrait  
            new Size(2448, 3264), // 8MP portrait
            new Size(1920, 2560), // 5MP portrait
            new Size(1440, 1920), // 3MP portrait
            new Size(1080, 1920), // Full HD portrait
            };
        } else {
            preferredResolutions = new Size[]{
            new Size(4032, 3024), // 12MP landscape
            new Size(3840, 2160), // 4K landscape
            new Size(3264, 2448), // 8MP landscape  
            new Size(2560, 1920), // 5MP landscape
            new Size(1920, 1440), // 3MP landscape
            new Size(1920, 1080), // Full HD landscape
            };
        }
        
        // Return the first (highest quality) option - CameraX will adapt if not supported
        Size optimalResolution = preferredResolutions[0];
        return optimalResolution;
        
        } catch (Exception e) {
            Log.e("Camera", "Error selecting optimal resolution: " + e.getMessage());
        }
        
        // Fallback: return null to let CameraX auto-select
        return null;
    }

    /**
     * Initialize responsive auto-focus on camera start
     */
    private void initializeResponsiveAutoFocus() {
        if (camera != null && previewView != null) {
            try {
                // Set initial focus to center for faster startup
                MeteringPointFactory factory = previewView.getMeteringPointFactory();
                float centerX = previewView.getWidth() / 2.0f;
                float centerY = previewView.getHeight() / 2.0f;
                MeteringPoint centerPoint = factory.createPoint(centerX, centerY);
                
                // Create responsive auto-focus action with fast duration for quick transitions
                FocusMeteringAction initialFocus = new FocusMeteringAction.Builder(centerPoint)
                    .setAutoCancelDuration(1, TimeUnit.SECONDS) // Fast 1 second for responsive focus
                    .build();
                
                camera.getCameraControl().startFocusAndMetering(initialFocus);
                
                // Enable continuous auto-focus by starting a background focus monitoring
                startContinuousAutoFocus();
            } catch (Exception e) {
                Log.e("Camera", "Failed to initialize responsive auto-focus: " + e.getMessage());
            }
        }
    }

    @PluginMethod
    public void startCamera(PluginCall call) {
        getActivity().runOnUiThread(new Runnable() {
            public void run() {
                try {
                    // Validate that all required components are ready
                    if (cameraProvider == null) {
                        call.reject("Camera provider not initialized");
                        return;
                    }
                    if (cameraSelector == null) {
                        call.reject("Camera selector not initialized");
                        return;
                    }
                    if (useCaseGroup == null) {
                        // Re-initialize use cases if they were cleared
                        setupUseCases(false);
                        if (useCaseGroup == null) {
                            call.reject("Camera use cases not initialized");
                            return;
                        }
                    }
                    if (previewView == null) {
                        call.reject("Preview view not initialized");
                        return;
                    }

                    // Ensure preview surface is properly connected
                    if (preview != null) {
                        preview.setSurfaceProvider(previewView.getSurfaceProvider());
                    }

                    // Make UI changes first
                    previewView.setVisibility(View.VISIBLE);
                    previewView.setBackgroundColor(Color.BLACK);
                    makeWebViewTransparent();

                    // Small delay to ensure preview surface is ready
                    exec.execute(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                Thread.sleep(50); // Brief delay for surface initialization
                                
                                getActivity().runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        try {
                                            // Bind camera to lifecycle
                                            camera = cameraProvider.bindToLifecycle((LifecycleOwner) getContext(), cameraSelector, useCaseGroup);
                                            
                                            // Initialize responsive auto-focus for better performance
                                            initializeResponsiveAutoFocus();
                                            
                                            triggerOnPlayed();
                                            call.resolve();
                                        } catch (Exception e) {
                                            e.printStackTrace();
                                            call.reject("Failed to bind camera: " + e.getMessage());
                                        }
                                    }
                                });
                            } catch (InterruptedException e) {
                                call.reject("Camera start interrupted: " + e.getMessage());
                            }
                        }
                    });
                } catch (Exception e) {
                    e.printStackTrace();
                    call.reject(e.getMessage());
                }
            }
        });
    }

    @PluginMethod
    public void stopCamera(PluginCall call) {
        getActivity().runOnUiThread(new Runnable() {
            public void run() {
                try {
                    restoreWebViewBackground();
                    if (previewView != null) {
                        previewView.setVisibility(View.INVISIBLE);
                        previewView.setBackgroundColor(Color.BLACK);
                    }
                    if (cameraProvider != null) {
                        cameraProvider.unbindAll();
                    }
                    // Null out references to help GC and ensure release
                    camera = null;
                    imageCapture = null;
                    preview = null;
                    imageAnalysis = null;
                    useCaseGroup = null;
                    recorder = null;
                    currentRecording = null;
                    call.resolve();
                } catch (Exception e) {
                    call.reject(e.getMessage());
                }
            }
        });
    }

    private void makeWebViewTransparent() {
        bridge.getWebView().setTag(bridge.getWebView().getBackground());
        bridge.getWebView().setBackgroundColor(Color.TRANSPARENT);
    }

    private void restoreWebViewBackground() {
        bridge.getWebView().setBackground((Drawable) bridge.getWebView().getTag());
    }

    @PluginMethod
    public void toggleTorch(PluginCall call) {
        try {
            boolean torchOn = call.getBoolean("on", true);
            
            if (torchOn) {
                camera.getCameraControl().enableTorch(true);
            } else {
                camera.getCameraControl().enableTorch(false);
            }
            
            // Check if it's front camera and adjust background accordingly
            if (cameraSelector != null && cameraSelector.getLensFacing() == CameraSelector.LENS_FACING_FRONT) {
                if (torchOn) {
                    previewView.setBackgroundColor(Color.WHITE);
                } else {
                    previewView.setBackgroundColor(Color.BLACK);
                }
            } else {
                // For back camera, ensure everything is back to normal
                previewView.setBackgroundColor(Color.BLACK);
            }

            call.resolve();
        } catch (Exception e) {
            call.reject(e.getMessage());
        }
    }

    @PluginMethod
    public void setScanRegion(PluginCall call) {
        JSObject region = call.getObject("region");
        try {
            scanRegion = new ScanRegion(region.getInt("top"),
                    region.getInt("bottom"),
                    region.getInt("left"),
                    region.getInt("right"),
                    region.getInt("measuredByPercentage"));
        } catch (JSONException e) {
            call.reject(e.getMessage());
        }
        call.resolve();
    }

    @PluginMethod
    public void setZoom(PluginCall call) {
        if (call.hasOption("factor") && camera != null) {
            Float factor = call.getFloat("factor");
            try {
                camera.getCameraControl().setZoomRatio(factor);
                
                // Automatically trigger focus after zoom change for better UX
                ExecutorService zoomFocusExecutor = Executors.newSingleThreadExecutor();
                zoomFocusExecutor.execute(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            // Wait briefly for zoom to settle - faster for responsive focus
                            Thread.sleep(150);
                            
                            getActivity().runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    try {
                                        // Trigger auto-focus at center after zoom change
                                        if (previewView != null && camera != null) {
                                            MeteringPointFactory factory = previewView.getMeteringPointFactory();
                                            float centerX = previewView.getWidth() / 2.0f;
                                            float centerY = previewView.getHeight() / 2.0f;
                                            MeteringPoint centerPoint = factory.createPoint(centerX, centerY);
                                            
                                            // Use fast focus settings for responsive zoom-triggered focus
                                            FocusMeteringAction zoomFocusAction = new FocusMeteringAction.Builder(centerPoint,
                                                FocusMeteringAction.FLAG_AF | FocusMeteringAction.FLAG_AE)
                                                .setAutoCancelDuration(1, TimeUnit.SECONDS) // Fast 1 second for zoom focus
                                                .build();
                                            
                                            // Trigger focus after zoom change - simplified without result handling
                                            camera.getCameraControl().startFocusAndMetering(zoomFocusAction);
                                        }
                                    } catch (Exception e) {
                                        Log.d("Camera", "Auto-focus after zoom failed: " + e.getMessage());
                                    }
                                }
                            });
                        } catch (InterruptedException e) {
                            Log.d("Camera", "Zoom focus interrupted");
                        }
                    }
                });
                
            } catch (Exception e) {
                e.printStackTrace();
                call.resolve();
            }
        }
        call.resolve();
    }

    @PluginMethod
    public void setFocus(PluginCall call) {
        
        JSObject response = new JSObject();
        if (!call.hasOption("x") || !call.hasOption("y")) {
            response.put("success", false);
            call.resolve(response);
        return;
        }

        Float x = call.getFloat("x");
        Float y = call.getFloat("y");

        // Check for null values
        if (x == null || y == null) {
            response.put("success", false);
            call.resolve(response);
            return;
        }

        // Validate coordinate ranges (should be 0-1 for normalized coordinates)
        if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) {
        response.put("success", false);
        call.resolve(response);
        return;
        }

        if (previewView == null || camera == null) {
          response.put("success", false);
          call.resolve(response);
          return;
        }

        getActivity().runOnUiThread(new Runnable() {
        @Override
        public void run() {
            try {
            // Only cancel if focus has been stable for a while to reduce multiple tap issues
            // This prevents interrupting legitimate focus operations
            
            // Use PreviewView's built-in MeteringPointFactory for proper coordinate transformation
            MeteringPointFactory factory = previewView.getMeteringPointFactory();

            // Convert normalized coordinates to preview coordinates
            float previewX = x * previewView.getWidth();
            float previewY = y * previewView.getHeight();

            MeteringPoint focusPoint = factory.createPoint(previewX, previewY);

            // Get configurable options with fast defaults for responsive focus
            boolean includeExposure = Boolean.TRUE.equals(call.getBoolean("includeExposure", true));
            Integer autoCancelDurationParam = call.getInt("autoCancelDurationSeconds");
            // Reduced to 1 second for fast responsive focus during near/far transitions
            int autoCancelDuration = autoCancelDurationParam != null ? autoCancelDurationParam : 1;

            // Build the focus and metering action with optimized settings
            FocusMeteringAction.Builder builder;

            // Set auto-focus flags with optimized configuration
            if (includeExposure) {
                // Use both AF and AE flags for the same point
                builder = new FocusMeteringAction.Builder(focusPoint, FocusMeteringAction.FLAG_AF | FocusMeteringAction.FLAG_AE);
            } else {
                // Use only AF flag for faster focus-only operations
                builder = new FocusMeteringAction.Builder(focusPoint, FocusMeteringAction.FLAG_AF);
            }

            // Set shorter auto-cancel duration for responsive focus
            builder.setAutoCancelDuration(autoCancelDuration, TimeUnit.SECONDS);

            FocusMeteringAction action = builder.build();

            // Start focus and metering with result callback
            ListenableFuture<FocusMeteringResult> future = camera.getCameraControl().startFocusAndMetering(action);

            // Simple callback to return focus result
            future.addListener(new Runnable() {
                @Override
                public void run() {
                    JSObject response = new JSObject();
                    try {
                        FocusMeteringResult result = future.get();
                        response.put("success", true);
                        response.put("autoFocusSuccessful", result.isFocusSuccessful());
                        response.put("x", x);
                        response.put("y", y);

                        // If focus failed, try a backup focus attempt to reduce need for multiple taps
                        if (!result.isFocusSuccessful()) {
                            performBackupFocus(previewX, previewY);
                        } else {
                            // If manual focus was successful, maintain it with a follow-up action
                            maintainFocusAtPoint(previewX, previewY);
                        }
                        
                        call.resolve(response);
                    } catch (Exception e) {
                        response.put("success", false);
                        Log.e("Camera", "Focus operation failed", e);
                        call.resolve(response);
                    }
                }
            }, ContextCompat.getMainExecutor(getContext()));

            } catch (Exception e) {
            Log.e("Camera", "Error setting focus", e);
            response.put("success", false);
            response.put("error", e.getMessage());
            call.resolve(response);
            }
        }
        });
    }

    @PluginMethod
    public void setAutoFocusMode(PluginCall call) {
        if (camera == null) {
        call.reject("Camera not initialized");
        return;
        }

        getActivity().runOnUiThread(new Runnable() {
        @Override
        public void run() {
            try {
            String mode = call.getString("mode", "continuous");
            boolean enableFastTransitions = call.getBoolean("enableFastTransitions", true);
            boolean enableAdaptiveFocus = call.getBoolean("enableAdaptiveFocus", true);

            switch (mode.toLowerCase()) {
                case "auto":
                // Default auto-focus behavior with optimization
                if (enableFastTransitions) {
                    // Reset focus to enable faster transitions
                    camera.getCameraControl().cancelFocusAndMetering();
                }
                break;
                case "manual":
                // For manual focus, we would need to disable auto-focus
                // This is more complex and would require camera2 interop
                Log.w("Camera", "Manual focus mode not fully supported in CameraX");
                break;
                case "continuous":
                // Enhanced continuous focus mode with adaptive focusing
                if (enableFastTransitions) {
                    // Cancel and restart for faster focus transitions
                    camera.getCameraControl().cancelFocusAndMetering();
                }
                if (enableAdaptiveFocus) {
                    // Start adaptive continuous focus for better near/far transitions
                    startAdaptiveContinuousFocus();
                }
                break;
                case "macro":
                // Optimized for close-up focusing
                Log.i("Camera", "Macro focus mode - optimized for close distances");
                break;
                case "infinity":
                // Optimized for far distance focusing
                // Focus at center point with infinity bias
                if (previewView != null) {
                    MeteringPointFactory factory = previewView.getMeteringPointFactory();
                    float centerX = previewView.getWidth() / 2.0f;
                    float centerY = previewView.getHeight() / 2.0f;
                    MeteringPoint centerPoint = factory.createPoint(centerX, centerY);
                    
                    FocusMeteringAction infinityAction = new FocusMeteringAction.Builder(centerPoint)
                        .setAutoCancelDuration(1, TimeUnit.SECONDS)
                        .build();
                    
                    camera.getCameraControl().startFocusAndMetering(infinityAction);
                }
                break;
                default:
                call.reject("Unsupported focus mode: " + mode);
                return;
            }

            JSObject result = new JSObject();
            result.put("success", true);
            result.put("mode", mode);
            result.put("enableFastTransitions", enableFastTransitions);
            result.put("enableAdaptiveFocus", enableAdaptiveFocus);
            call.resolve(result);
            } catch (Exception e) {
            call.reject("Error setting auto focus mode: " + e.getMessage());
            }
        }
        });
    }

    /**
     * Start adaptive continuous focus for better handling of near/far object transitions
     */
    private void startAdaptiveContinuousFocus() {
        if (camera == null || previewView == null) return;
        
        ExecutorService adaptiveFocusExecutor = Executors.newSingleThreadExecutor();
        
        adaptiveFocusExecutor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    // Create multiple focus points for better scene coverage
                    float[] focusPoints = {
                        0.3f, 0.3f,  // Top-left quadrant
                        0.7f, 0.3f,  // Top-right quadrant  
                        0.5f, 0.5f,  // Center
                        0.3f, 0.7f,  // Bottom-left quadrant
                        0.7f, 0.7f   // Bottom-right quadrant
                    };
                    
                    int pointIndex = 0;
                    
                    while (camera != null && camera.getCameraInfo().getCameraState().getValue().getType() == CameraState.Type.OPEN) {
                        Thread.sleep(600); // Focus check every 600ms for faster near/far transitions
                        
                        final int currentPointIndex = pointIndex;
                        
                        getActivity().runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                try {
                                    if (camera != null && previewView != null) {
                                        MeteringPointFactory factory = previewView.getMeteringPointFactory();
                                        
                                        // Use current focus point from the array
                                        float x = focusPoints[currentPointIndex * 2] * previewView.getWidth();
                                        float y = focusPoints[currentPointIndex * 2 + 1] * previewView.getHeight();
                                        
                                        MeteringPoint adaptivePoint = factory.createPoint(x, y);
                                        
                                        FocusMeteringAction adaptiveAction = new FocusMeteringAction.Builder(adaptivePoint)
                                            .setAutoCancelDuration(1, TimeUnit.SECONDS) // Fast 1 second for adaptive focus
                                            .build();
                                        
                                        camera.getCameraControl().startFocusAndMetering(adaptiveAction);
                                    }
                                } catch (Exception e) {
                                    Log.d("Camera", "Adaptive focus failed: " + e.getMessage());
                                }
                            }
                        });
                        
                        // Cycle through focus points
                        pointIndex = (pointIndex + 1) % (focusPoints.length / 2);
                    }
                } catch (InterruptedException e) {
                    Log.d("Camera", "Adaptive continuous focus stopped");
                }
            }
        });
    }

    @PluginMethod
    public void resetFocus(PluginCall call) {
        if (camera == null) {
        call.reject("Camera not initialized");
        return;
        }

        getActivity().runOnUiThread(new Runnable() {
        @Override
        public void run() {
            try {
            // Cancel any ongoing focus operations
            camera.getCameraControl().cancelFocusAndMetering();

            // Restart enhanced continuous autofocus for better stability
            boolean restartContinuous = call.getBoolean("restartContinuous", true);
            if (restartContinuous && previewView != null) {
                // Set focus to center of screen to restart continuous AF with improved settings
                MeteringPointFactory factory = previewView.getMeteringPointFactory();
                float centerX = previewView.getWidth() / 2.0f;
                float centerY = previewView.getHeight() / 2.0f;
                MeteringPoint centerPoint = factory.createPoint(centerX, centerY);
                
                FocusMeteringAction restartAction = new FocusMeteringAction.Builder(centerPoint)
                    .setAutoCancelDuration(1, TimeUnit.SECONDS) // Fast 1 second for responsive restart
                    .build();
                
                camera.getCameraControl().startFocusAndMetering(restartAction);
                
                // Restart the continuous auto-focus monitoring
                startContinuousAutoFocus();
            }

            JSObject result = new JSObject();
            result.put("success", true);
            result.put("restartedContinuous", restartContinuous);
            call.resolve(result);
            } catch (Exception e) {
            call.reject("Error resetting focus: " + e.getMessage());
            }
        }
        });
    }

    @PluginMethod
    public void selectCamera(PluginCall call) {
        getActivity().runOnUiThread(new Runnable() {
            @RequiresApi(api = Build.VERSION_CODES.P)
            public void run() {
                if (call.hasOption("cameraID")) {
                    try {
                        String cameraID = call.getString("cameraID");
                        if (cameraID.equals("Front-Facing Camera")) {
                            cameraSelector = new CameraSelector.Builder()
                                    .requireLensFacing(CameraSelector.LENS_FACING_FRONT).build();
                        } else {
                            cameraSelector = new CameraSelector.Builder()
                                    .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
                        }
                        if (camera != null) {
                            if (camera.getCameraInfo().getCameraState().getValue().getType() == CameraState.Type.OPEN) {
                                cameraProvider.unbindAll();
                                setupUseCases(false);
                                camera = cameraProvider.bindToLifecycle((LifecycleOwner) getContext(), cameraSelector, useCaseGroup);
                                triggerOnPlayed();
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                        call.reject(e.getMessage());
                        return;
                    }
                }
                JSObject result = new JSObject();
                result.put("success", true);
                call.resolve(result);
            }
        });
    }

    @PluginMethod
    public void setLayout(PluginCall call){
        if (previewView != null) {
            getActivity().runOnUiThread(new Runnable() {
                public void run() {
                    if (call.hasOption("width") && call.hasOption("height") && call.hasOption("left") && call.hasOption("top")) {
                        try{
                            double width = getLayoutValue(call.getString("width"),true);
                            double height = getLayoutValue(call.getString("height"),false);
                            double left = getLayoutValue(call.getString("left"),true);
                            double top = getLayoutValue(call.getString("top"),false);
                            previewView.setX((int) left);
                            previewView.setY((int) top);
                            ViewGroup.LayoutParams cameraPreviewParams = previewView.getLayoutParams();
                            cameraPreviewParams.width = (int) width;
                            cameraPreviewParams.height = (int) height;
                            previewView.setLayoutParams(cameraPreviewParams);
                        }catch(Exception e) {
                            Log.d("Camera",e.getMessage());
                        }
                    }
                    call.resolve();
                }
            });
        }else{
            call.reject("Camera not initialized");
        }
    }
    private double getLayoutValue(String value,boolean isWidth) {
        if (value.indexOf("%") != -1) {
            double percent = Double.parseDouble(value.substring(0,value.length()-1))/100;
            if (isWidth) {
                return percent * Resources.getSystem().getDisplayMetrics().widthPixels;
            }else{
                return percent * Resources.getSystem().getDisplayMetrics().heightPixels;
            }
        }
        if (value.indexOf("px") != -1) {
            return Double.parseDouble(value.substring(0,value.length()-2));
        }
        try {
            return Double.parseDouble(value);
        }catch(Exception e) {
            if (isWidth) {
                return Resources.getSystem().getDisplayMetrics().widthPixels;
            }else{
                return Resources.getSystem().getDisplayMetrics().heightPixels;
            }
        }
    }
    private void triggerOnPlayed() {
        try {
            JSObject onPlayedResult = new JSObject();
            @SuppressLint("RestrictedApi")
            String res = imageAnalysis.getAttachedSurfaceResolution().getWidth() + "x" + imageAnalysis.getAttachedSurfaceResolution().getHeight();
            onPlayedResult.put("resolution", res);
            notifyListeners("onPlayed", onPlayedResult);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @SuppressLint("RestrictedApi")
    @PluginMethod
    public void getAllCameras(PluginCall call) {
        JSObject result = new JSObject();
        JSArray cameras = new JSArray();
        cameras.put("Back-Facing Camera");
        cameras.put("Front-Facing Camera");
        result.put("cameras", cameras);
        call.resolve(result);
    }

    @SuppressLint("RestrictedApi")
    @PluginMethod
    public void getSelectedCamera(PluginCall call) {
        if (cameraSelector == null) {
            call.reject("not initialized");
        } else {
            JSObject result = new JSObject();
            String cameraID = "Back-Facing Camera";
            if (cameraSelector.getLensFacing() == CameraSelector.LENS_FACING_FRONT) {
                cameraID = "Front-Facing Camera";
            }
            result.put("selectedCamera", cameraID);
            call.resolve(result);
        }
    }

    @PluginMethod
    public void setResolution(PluginCall call) {
        getActivity().runOnUiThread(new Runnable() {
            @RequiresApi(api = Build.VERSION_CODES.P)
            public void run() {
                if (call.hasOption("resolution")) {
                    try {
                        int res = call.getInt("resolution");
                        int width = 1280;
                        int height = 720;
                        if (res == 1) {
                            width = 640;
                            height = 480;
                        } else if (res == 2) {
                            width = 1280;
                            height = 720;
                        } else if (res == 3) {
                            width = 1920;
                            height = 1080;
                        } else if (res == 4) {
                            width = 2560;
                            height = 1440;
                        } else if (res == 5) {
                            width = 3840;
                            height = 2160;
                        }
                        desiredHeight = height;
                        desiredWidth = width;
                        CameraState.Type status = null;
                        if (camera != null) {
                            status = camera.getCameraInfo().getCameraState().getValue().getType();
                            if (status == CameraState.Type.OPEN) {
                                cameraProvider.unbindAll();
                            }
                        }
                        setupUseCases(false);
                        if (camera != null) {
                            if (status == CameraState.Type.OPEN) {
                                camera = cameraProvider.bindToLifecycle((LifecycleOwner) getContext(), cameraSelector, useCaseGroup);
                                triggerOnPlayed();
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                        call.reject(e.getMessage());
                        return;
                    }
                }
                JSObject result = new JSObject();
                result.put("success", true);
                call.resolve(result);
            }
        });
    }

    @SuppressLint("RestrictedApi")
    @PluginMethod
    public void getResolution(PluginCall call) {
        if (camera == null) {
            call.reject("Camera not initialized");
        } else {
            try {
                JSObject result = new JSObject();
                result.put("resolution", imageAnalysis.getAttachedSurfaceResolution().getWidth() + "x" + imageAnalysis.getAttachedSurfaceResolution().getHeight());
                call.resolve(result);
            } catch (Exception e) {
                call.reject(e.getMessage());
            }

        }
    }

    static public Bitmap getBitmap() {
        try {
            return frameTaken;
        } catch (Exception e) {
            return null;
        }
    }

    @PluginMethod
    public void takeSnapshot(PluginCall call) {
        if (camera == null) {
            call.reject("Camera not initialized.");
            return;
        }
        
        try {
            call.setKeepAlive(true);
            takeSnapshotCall = call;
        } catch (Exception e) {
            call.reject("Failed to take snapshot: " + e.getMessage());
        }
    }

    @PluginMethod
    public void saveFrame(PluginCall call) {
        call.setKeepAlive(true);
        saveFrameCall = call;
    }

    @PluginMethod
    public void takePhoto(PluginCall call) {
        if (camera == null) {
            call.reject("Camera not initialized.");
            return;
        }
        getActivity().runOnUiThread(new Runnable() {
            @RequiresApi(api = Build.VERSION_CODES.P)
            public void run() {
                if (useCaseGroup.getUseCases().contains(imageCapture) == false) {
                    cameraProvider.unbindAll();
                    setupUseCases(false);
                    camera = cameraProvider.bindToLifecycle((LifecycleOwner) getContext(), cameraSelector, useCaseGroup);
                }
                File file;
                if (call.hasOption("pathToSave")) {
                    file = new File(call.getString("pathToSave"));
                } else {
                    File dir = getContext().getExternalCacheDir();
                    file = new File(dir, new Date().getTime() + ".jpg");
                }
                ImageCapture.OutputFileOptions outputFileOptions =
                        new ImageCapture.OutputFileOptions.Builder(file).build();
                imageCapture.takePicture(outputFileOptions, exec,
                        new ImageCapture.OnImageSavedCallback() {
                            @Override
                            public void onImageSaved(ImageCapture.OutputFileResults outputFileResults) {
                                JSObject result = new JSObject();
                                if (call.getBoolean("includeBase64", false)) {
                                    String base64 = Base64.encodeToString(convertFileToByteArray(file), Base64.DEFAULT);
                                    result.put("base64", base64);
                                    
                                    // Detect blur if base64 is included
                                    Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
                                    if (bitmap != null) {
                                        double confidence = calculateBlurConfidence(bitmap);
                                        result.put("confidence", confidence);
                                    }
                                }
                                result.put("path", file.getAbsolutePath());
                            }

                            @Override
                            public void onError(@NonNull ImageCaptureException exception) {
                                call.reject(exception.getMessage());
                            }
                        }
                );
            }
        });
    }

    @RequiresApi(api = Build.VERSION_CODES.P)
    @PluginMethod
    public void startRecording(PluginCall call) {
        if (camera == null) {
            call.reject("Camera not initialized.");
            return;
        }
        getActivity().runOnUiThread(new Runnable() {
            @RequiresApi(api = Build.VERSION_CODES.P)
            public void run() {
                cameraProvider.unbindAll();
                setupUseCases(true);
                camera = cameraProvider.bindToLifecycle((LifecycleOwner) getContext(), cameraSelector, useCaseGroup);
                if (recorder != null) {
                    // create MediaStoreOutputOptions for our recorder: resulting our recording!
                    String name = "CameraX-recording-" + System.currentTimeMillis() + ".mp4";
                    ContentValues contentValues = new ContentValues();
                    contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, name);

                    MediaStoreOutputOptions mediaStoreOutput = new MediaStoreOutputOptions.Builder(
                            getContext().getContentResolver(),
                            MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
                            .setContentValues(contentValues)
                            .build();

                    // configure Recorder and Start recording to the mediaStoreOutput.

                    PendingRecording pendingRecording = recorder.prepareRecording(getContext(), mediaStoreOutput);

                    if (ActivityCompat.checkSelfPermission(getContext(), Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                        // TODO: Consider calling
                        //    ActivityCompat#requestPermissions
                        // here to request the missing permissions, and then overriding
                        //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                        //                                          int[] grantResults)
                        // to handle the case where the user grants the permission. See the documentation
                        // for ActivityCompat#requestPermissions for more details.
                    }else{
                        pendingRecording.withAudioEnabled();
                    }
                    Consumer<VideoRecordEvent> captureListener = new Consumer<VideoRecordEvent>() {
                        @Override
                        public void accept(VideoRecordEvent videoRecordEvent) {
                            if (videoRecordEvent instanceof VideoRecordEvent.Finalize) {
                                Uri uri = ((VideoRecordEvent.Finalize) videoRecordEvent).getOutputResults().getOutputUri();
                                String path = uri.getPath();

                                if (stopRecordingCall != null) {
                                    JSObject result = new JSObject();
                                    if (stopRecordingCall.getBoolean("includeBase64",false)) {
                                        try {
                                            InputStream iStream = getContext().getContentResolver().openInputStream(uri);
                                            byte[] inputData = getBytes(iStream);
                                            String base64 = Base64.encodeToString(inputData, Base64.DEFAULT);
                                            result.put("base64",base64);
                                        } catch (IOException e) {
                                            throw new RuntimeException(e);
                                        }
                                    }
                                    result.put("path",path);
                                    recorder = null;
                                    stopRecordingCall.resolve(result);
                                    stopRecordingCall = null;
                                }
                            }
                        }
                    };
                    currentRecording = pendingRecording.start(getContext().getMainExecutor(),captureListener);
                    call.resolve();
                }else{
                    call.reject("Recording is not ready");
                }
            }
        });
    }

    private byte[] getBytes(InputStream inputStream) throws IOException {
        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
        int bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];

        int len = 0;
        while ((len = inputStream.read(buffer)) != -1) {
            byteBuffer.write(buffer, 0, len);
        }
        return byteBuffer.toByteArray();
    }

    @PluginMethod
    public void stopRecording(PluginCall call){
        getActivity().runOnUiThread(new Runnable() {

            @Override
            public void run() {
                call.setKeepAlive(true);
                stopRecordingCall = call;
                currentRecording.stop();
            }
        });
    }

    public static byte[] convertFileToByteArray(File f) {
        byte[] byteArray = null;
        try {
            InputStream inputStream = new FileInputStream(f);
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] b = new byte[1024 * 8];
            int bytesRead = 0;
            while ((bytesRead = inputStream.read(b)) != -1) {
                bos.write(b, 0, bytesRead);
            }

            byteArray = bos.toByteArray();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return byteArray;
    }

    public static String bitmap2Base64(Bitmap bitmap,int quality) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream);
        return Base64.encodeToString(outputStream.toByteArray(), Base64.DEFAULT);
    }

    @PluginMethod
    public void isOpen(PluginCall call){
        if (camera != null) {
            JSObject result = new JSObject();
            if (camera.getCameraInfo().getCameraState().getValue().getType() == CameraState.Type.OPEN) {
                result.put("isOpen",true);
            }else{
                result.put("isOpen",false);
            }
            call.resolve(result);
        }else {
            call.reject("Camera not initialized.");
        }
    }

    @Override
    protected void handleOnPause() {
        if (camera != null && cameraProvider != null) {
            CameraState cameraStatus = camera.getCameraInfo().getCameraState().getValue();
            previousCameraStatus = cameraStatus;
            if (cameraStatus.getType() == CameraState.Type.OPEN) {
                cameraProvider.unbindAll();
            }
            // Null out references
            camera = null;
            imageCapture = null;
            preview = null;
            imageAnalysis = null;
            useCaseGroup = null;
            recorder = null;
            currentRecording = null;
        }
        
        // Clean up TFLite resources
        // if (blurDetectionHelper != null) {
        //     blurDetectionHelper.close();
        // }
        
        super.handleOnPause();
    }

    @Override
    protected void handleOnResume() {
        if (camera != null) {
            if (previousCameraStatus.getType() == CameraState.Type.OPEN) {
                camera = cameraProvider.bindToLifecycle((LifecycleOwner) getContext(), cameraSelector, useCaseGroup);
            }
        }
        super.handleOnResume();
    }

    @Override
    protected void handleOnConfigurationChanged(Configuration newConfig) {
        notifyListeners("onOrientationChanged",null);
        super.handleOnConfigurationChanged(newConfig);
    }



    @PluginMethod
    public void requestCameraPermission(PluginCall call) {
        boolean hasCameraPerms = getPermissionState(CAMERA) == PermissionState.GRANTED;
        if (hasCameraPerms == false) {
            String[] aliases = new String[] { CAMERA };
            requestPermissionForAliases(aliases, call, "cameraPermissionsCallback");
        }else{
            call.resolve();
        }
    }

    @PermissionCallback
    private void cameraPermissionsCallback(PluginCall call) {
        boolean hasCameraPerms = getPermissionState(CAMERA) == PermissionState.GRANTED;
        if (hasCameraPerms) {
            call.resolve();
        }else {
            call.reject("Permission not granted.");
        }
    }

    @PluginMethod
    public void requestMicroPhonePermission(PluginCall call) {
        boolean hasCameraPerms = getPermissionState(MICROPHONE) == PermissionState.GRANTED;
        if (hasCameraPerms == false) {
            String[] aliases = new String[] { MICROPHONE };
            requestPermissionForAliases(aliases, call, "microphonePermissionsCallback");
        }else{
            call.resolve();
        }
    }

    @PermissionCallback
    private void microphonePermissionsCallback(PluginCall call) {
        boolean hasPerms = getPermissionState(MICROPHONE) == PermissionState.GRANTED;
        if (hasPerms) {
            call.resolve();
        }else {
            call.reject("Permission not granted.");
        }
    }

    @PluginMethod
    public void detectBlur(PluginCall call) {
        String imageString = call.getString("image");
        if (imageString == null) {
            call.reject("Image parameter is required");
            return;
        }
        
        try {
            // Convert base64 string to Bitmap
            String base64String = imageString;
            if (imageString.startsWith("data:")) {
                base64String = imageString.substring(imageString.indexOf(",") + 1);
            }
            
            byte[] decodedBytes = Base64.decode(base64String, Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.length);
            
            if (bitmap == null) {
                call.reject("Invalid image data");
                return;
            }
            
            // Use the new confidence detection method
            if (blurDetectionHelper != null && blurDetectionHelper.isInitialized()) {
                java.util.Map<String, Object> result = blurDetectionHelper.detectBlurWithConfidence(bitmap);
                
                JSObject jsResult = new JSObject();
                jsResult.put("isBlur", result.get("isBlur"));
                jsResult.put("blurConfidence", result.get("blurConfidence"));
                jsResult.put("sharpConfidence", result.get("sharpConfidence"));
                
                call.resolve(jsResult);
            } else {
                // Fallback to Laplacian algorithm with confidence scores
                double laplacianScore = calculateLaplacianBlurScore(bitmap);
                boolean isBlur = laplacianScore < 150;
                double normalizedScore = Math.max(0.0, Math.min(1.0, laplacianScore / 300.0));
                double sharpConfidence = normalizedScore;
                double blurConfidence = 1.0 - normalizedScore;
                
                JSObject result = new JSObject();
                result.put("isBlur", isBlur);
                result.put("blurConfidence", blurConfidence);
                result.put("sharpConfidence", sharpConfidence);
                
                call.resolve(result);
            }
            
        } catch (Exception e) {
            call.reject("Failed to process image: " + e.getMessage());
        }
    }

    @PluginMethod
    public void getOrientation(PluginCall call) {
        int orientation = getContext().getResources().getConfiguration().orientation;
        JSObject result = new JSObject();
        if (orientation == Configuration.ORIENTATION_PORTRAIT) {
            result.put("orientation","PORTRAIT");
        }else{
            result.put("orientation","LANDSCAPE");
        }
        call.resolve(result);
    }

    /**
     * Calculate if image is blurry using TFLite model (with Laplacian fallback)
     * Returns true if blurry, false if sharp
     */
    private boolean calculateBlurResult(Bitmap bitmap) {
        if (bitmap == null) return false;
        
        // Use TFLite model if available, otherwise fallback to Laplacian
        if (blurDetectionHelper != null && blurDetectionHelper.isInitialized()) {
            return blurDetectionHelper.isBlurry(bitmap);
        } else {
            // Fallback to original Laplacian algorithm
            double laplacianScore = calculateLaplacianBlurScore(bitmap);
            return laplacianScore < 50;
        }
    }

    /**
     * Calculate blur confidence score (0-1 where 1 is sharp, 0 is blurry)
     */
    private double calculateBlurConfidence(Bitmap bitmap) {
        if (bitmap == null) return 0.0;
        
        // Use TFLite model if available for detailed confidence
        if (blurDetectionHelper != null && blurDetectionHelper.isInitialized()) {
            java.util.Map<String, Object> result = blurDetectionHelper.detectBlurWithConfidence(bitmap);
            Boolean isBlur = (Boolean) result.get("isBlur");
            Double blurConfidence = (Double) result.get("blurConfidence");
            Double sharpConfidence = (Double) result.get("sharpConfidence");
            
            // Return the actual blurConfidence value for more nuanced results
            if (blurConfidence != null) {
                return blurConfidence;
            } else if (isBlur != null) {
                // Fallback to boolean if confidence not available
                return isBlur ? 1.0 : 0.0;
            } else {
                return 0.0;
            }
        } else {
            // Fallback to Laplacian algorithm with confidence calculation
            double laplacianScore = calculateLaplacianBlurScore(bitmap);
            // Normalize to 0-1 range (higher score = sharper image)
            return Math.max(0.0, Math.min(1.0, laplacianScore / 300.0));
        }
    }
    
    /**
     * Original Laplacian blur detection (fallback)
     * Returns raw Laplacian variance score (will be converted to percentage by BlurDetectionHelper)
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
        
        // Sample every 4th pixel for performance (similar to web implementation)
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
     * Start continuous auto-focus monitoring for better focus stability
     */
    private void startContinuousAutoFocus() {
        if (camera != null && previewView != null) {
            // Use a separate executor for continuous focus to avoid blocking
            ExecutorService focusExecutor = Executors.newSingleThreadExecutor();
            
            focusExecutor.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        while (camera != null && camera.getCameraInfo().getCameraState().getValue().getType() == CameraState.Type.OPEN) {
                            Thread.sleep(800); // Check every 800ms for faster transitions
                            
                            getActivity().runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    try {
                                        // Trigger auto-focus at center to maintain continuous focus
                                        MeteringPointFactory factory = previewView.getMeteringPointFactory();
                                        float centerX = previewView.getWidth() / 2.0f;
                                        float centerY = previewView.getHeight() / 2.0f;
                                        MeteringPoint centerPoint = factory.createPoint(centerX, centerY);
                                        
                                        FocusMeteringAction continuousAction = new FocusMeteringAction.Builder(centerPoint)
                                            .setAutoCancelDuration(1, TimeUnit.SECONDS) // Fast 1 second for responsive transitions
                                            .build();
                                        
                                        camera.getCameraControl().startFocusAndMetering(continuousAction);
                                    } catch (Exception e) {
                                        Log.d("Camera", "Continuous focus update failed: " + e.getMessage());
                                    }
                                }
                            });
                        }
                    } catch (InterruptedException e) {
                        Log.d("Camera", "Continuous auto-focus stopped");
                    }
                }
            });
        }
    }

    /**
     * Perform backup focus attempt if initial focus fails
     * This reduces the need for users to tap multiple times
     */
    private void performBackupFocus(float previewX, float previewY) {
        if (camera == null || previewView == null) return;
        
        // Wait a moment for the camera to settle
        ExecutorService backupFocusExecutor = Executors.newSingleThreadExecutor();
        backupFocusExecutor.execute(new Runnable() {
            @Override
            public void run() {
                                    try {
                        Thread.sleep(200); // Wait 200ms for faster backup focus
                    
                    getActivity().runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            try {
                                if (camera != null && previewView != null) {
                                    MeteringPointFactory factory = previewView.getMeteringPointFactory();
                                    MeteringPoint backupPoint = factory.createPoint(previewX, previewY);
                                    
                                    // Try with fast duration for responsive backup focus
                                    FocusMeteringAction backupAction = new FocusMeteringAction.Builder(backupPoint, 
                                        FocusMeteringAction.FLAG_AF | FocusMeteringAction.FLAG_AE)
                                        .setAutoCancelDuration(1, TimeUnit.SECONDS) // Fast 1 second for backup
                                        .build();
                                    
                                    ListenableFuture<FocusMeteringResult> backupFuture = 
                                        camera.getCameraControl().startFocusAndMetering(backupAction);
                                    
                                    backupFuture.addListener(new Runnable() {
                                        @Override
                                        public void run() {
                                            try {
                                                FocusMeteringResult backupResult = backupFuture.get();
                                                if (backupResult.isFocusSuccessful()) {
                                                    maintainFocusAtPoint(previewX, previewY);
                                                } else {
                                                    Log.d("Camera", "Backup focus also failed");
                                                }
                                            } catch (Exception e) {
                                                Log.d("Camera", "Backup focus exception: " + e.getMessage());
                                            }
                                        }
                                    }, ContextCompat.getMainExecutor(getContext()));
                                }
                            } catch (Exception e) {
                                Log.d("Camera", "Backup focus setup failed: " + e.getMessage());
                            }
                        }
                    });
                } catch (InterruptedException e) {
                    Log.d("Camera", "Backup focus interrupted");
                }
            }
        });
    }

    /**
     * Maintain focus at a specific point with repeated focus actions for stability
     */
    private void maintainFocusAtPoint(float previewX, float previewY) {
        if (camera == null || previewView == null) return;
        
        // Use a separate executor for focus maintenance
        ExecutorService focusMaintainExecutor = Executors.newSingleThreadExecutor();
        
        focusMaintainExecutor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    // Maintain focus for 2 seconds with quick refocus for responsive transitions
                    for (int i = 0; i < 2; i++) {
                        Thread.sleep(1000); // Wait 1 second between focus actions
                        
                        getActivity().runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                try {
                                    if (camera != null && camera.getCameraInfo().getCameraState().getValue().getType() == CameraState.Type.OPEN) {
                                        MeteringPointFactory factory = previewView.getMeteringPointFactory();
                                        MeteringPoint maintainPoint = factory.createPoint(previewX, previewY);
                                        
                                        FocusMeteringAction maintainAction = new FocusMeteringAction.Builder(maintainPoint)
                                            .setAutoCancelDuration(1, TimeUnit.SECONDS) // Fast 1 second maintenance
                                            .build();
                                        
                                        camera.getCameraControl().startFocusAndMetering(maintainAction);
                                    }
                                } catch (Exception e) {
                                    Log.d("Camera", "Focus maintenance failed: " + e.getMessage());
                                }
                            }
                        });
                    }
                } catch (InterruptedException e) {
                    Log.d("Camera", "Focus maintenance interrupted");
                }
            }
        });
    }

}
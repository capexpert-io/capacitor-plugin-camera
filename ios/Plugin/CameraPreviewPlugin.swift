import Foundation
import UIKit
import Capacitor
import AVFoundation

/**
 * Please read the Capacitor iOS Plugin Development Guide
 * here: https://capacitorjs.com/docs/plugins/ios
 */
@objc(CameraPreviewPlugin)
public class CameraPreviewPlugin: CAPPlugin, AVCaptureVideoDataOutputSampleBufferDelegate, AVCapturePhotoCaptureDelegate, AVCaptureFileOutputRecordingDelegate {
    
    var previewView: PreviewView!
    var captureSession: AVCaptureSession!
    var photoOutput: AVCapturePhotoOutput!
    var videoOutput: AVCaptureVideoDataOutput!
    var movieFileOutput: AVCaptureMovieFileOutput!
    var takeSnapshotCall: CAPPluginCall! = nil
    var takePhotoCall: CAPPluginCall! = nil
    var stopRecordingCall: CAPPluginCall! = nil
    var getResolutionCall: CAPPluginCall! = nil
    var saveFrameCall: CAPPluginCall! = nil
    static public var frameTaken:UIImage!
    var triggerPlayRequired = false
    var facingBack = true
    var videoInput:AVCaptureDeviceInput!
    var scanRegion:ScanRegion! = nil
    var lastValidOrientation = "portrait"
    var focusView: UIView?
    var isFocusAnimating = false
    var focusCompletionTimer: Timer?
    var lastFocusTime: Date = Date()
    private let focusThrottleInterval: TimeInterval = 0.5
    var currentCameraDevice: AVCaptureDevice?
    @objc func initialize(_ call: CAPPluginCall) {
        // Check camera permission status first
        let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
        
        if authStatus == .denied || authStatus == .restricted {
            call.reject("Camera permission denied")
            return
        }
        
        // Initialize a camera view for previewing video.
        DispatchQueue.main.sync {
            self.previewView = PreviewView.init(frame: (bridge?.viewController?.view.bounds)!)
            self.webView!.superview!.insertSubview(self.previewView, belowSubview: self.webView!)
            
            // Only initialize capture session if permission is granted
            if authStatus == .authorized {
                initializeCaptureSession(enableVideoRecording: false)
            }
            
            // Add tap gesture for focus
            let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTapToFocus(_:)))
            self.previewView.addGestureRecognizer(tapGesture)
        }
        call.resolve()
    }
    
    @objc func rotated() {
        let bounds = self.webView?.bounds
        if bounds != nil {
            self.previewView.frame = bounds!
            if UIDevice.current.orientation == UIDeviceOrientation.portrait {
                self.previewView.videoPreviewLayer.connection?.videoOrientation = .portrait
                lastValidOrientation = "portrait"
            }else if UIDevice.current.orientation == UIDeviceOrientation.landscapeLeft {
                self.previewView.videoPreviewLayer.connection?.videoOrientation = .landscapeRight
                lastValidOrientation = "landscapeRight"
            }else if UIDevice.current.orientation == UIDeviceOrientation.landscapeRight {
                self.previewView.videoPreviewLayer.connection?.videoOrientation = .landscapeLeft
                lastValidOrientation = "landscapeLeft"
            }
        }
        notifyListeners("onOrientationChanged",data: nil)
    }
    
    @objc func startCamera(_ call: CAPPluginCall) {
        makeWebViewTransparent()
        
        guard let captureSession = self.captureSession else {
            call.reject("Camera not initialized")
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            captureSession.startRunning()
            DispatchQueue.main.async {
                self.triggerOnPlayed()
            }
        }

        call.resolve()
    }

    func destroyCaptureSession() {
        guard let session = self.captureSession else { return }

        // Stop the session
        if session.isRunning {
            session.stopRunning()
        }

        // Remove all inputs
        for input in session.inputs {
            session.removeInput(input)
        }

        // Remove all outputs
        for output in session.outputs {
            session.removeOutput(output)
        }

        // Release session
        self.captureSession = nil
        initializeCaptureSession(enableVideoRecording: false)
    }

    func getCameraAspectRatio() -> CGFloat? {
        guard let videoDevice = self.videoInput?.device else {
            // call.reject("Video device not available")
          return 1;
        }
        
        // Retrieve the format description from the camera's active format.
        let formatDesc = videoDevice.activeFormat.formatDescription
        let dimensions = CMVideoFormatDescriptionGetDimensions(formatDesc)
        let camWidth = CGFloat(dimensions.width)
        let camHeight = CGFloat(dimensions.height)

        return camWidth / camHeight // Aspect Ratio (height / width)
    }

    func updatePreviewLayerFrame() {
      // Make sure you can get the view controller's view
          guard let previewView = self.bridge?.viewController?.view,
                let aspectRatio = getCameraAspectRatio() else { return }

        let screenWidth = previewView.bounds.width
        let previewHeight = screenWidth * aspectRatio // Calculate height

        let screenHeight = previewView.bounds.height
        let previewY = (screenHeight - previewHeight) * 0.5 // Center vertically
      
        self.previewView.frame = CGRect(x: 0, y: previewY, width: screenWidth, height: previewHeight)
    }
    
    func initializeCaptureSession(enableVideoRecording:Bool){
        // Create the capture session.
        self.captureSession = AVCaptureSession()

        // Find the best available camera device (prefer multi-camera systems)
        guard let videoDevice = getBestAvailableCameraDevice() else { return }
        if enableVideoRecording {
            let microphone = AVCaptureDevice.default(for: AVMediaType.audio)
            if microphone != nil {
                let micInput = try? AVCaptureDeviceInput(device: microphone!)
                if captureSession.canAddInput(micInput!) {
                    captureSession.addInput(micInput!)
                }
            }
        }
       
        do {
            // Wrap the video device in a capture device input.
            self.videoInput = try AVCaptureDeviceInput(device: videoDevice)
            // If the input can be added, add it to the session.
            if self.captureSession.canAddInput(videoInput) {
                self.captureSession.addInput(videoInput)
                
                // Set current camera device
                self.currentCameraDevice = videoDevice
                
                // Configure camera device for optimal focus performance (both close and far objects)
                try self.configureCameraForOptimalFocus(device: videoDevice)
                
                self.previewView.videoPreviewLayer.session = self.captureSession
                self.previewView.videoPreviewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
                
                self.videoOutput = AVCaptureVideoDataOutput.init()
                if self.captureSession.canAddOutput(self.videoOutput) {
                    self.captureSession.addOutput(videoOutput)
                }
                
                if let connection = self.previewView.videoPreviewLayer.connection {
                    connection.videoOrientation = .portrait
                }
                if let videoConnection = self.videoOutput.connection(with: .video) {
                    videoConnection.videoOrientation = .portrait
                }
                
                self.photoOutput = AVCapturePhotoOutput()
                self.photoOutput.isHighResolutionCaptureEnabled = true
                
                // Configure photo output for better focus
                if #available(iOS 13.0, *) {
                    self.photoOutput.maxPhotoQualityPrioritization = .quality
                }
                
                if self.captureSession.canAddOutput(self.photoOutput) {
                    self.captureSession.addOutput(photoOutput)
                }
                if enableVideoRecording {
                    self.movieFileOutput = AVCaptureMovieFileOutput()
                    if self.captureSession.canAddOutput(self.movieFileOutput) {
                        self.captureSession.addOutput(movieFileOutput)
                    }
                }

                self.captureSession.sessionPreset = AVCaptureSession.Preset.photo
              
                updatePreviewLayerFrame()
                
                var queue:DispatchQueue
                queue = DispatchQueue(label: "queue")
                self.videoOutput.setSampleBufferDelegate(self as AVCaptureVideoDataOutputSampleBufferDelegate, queue: queue)
                self.videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey : kCVPixelFormatType_32BGRA] as [String : Any]
            }
            
        } catch {
            // Configuration failed. Handle error.
            print(error)
        }
    }
    
    private func configureCameraForOptimalFocus(device: AVCaptureDevice) throws {
        try device.lockForConfiguration()
        
        // Set optimal focus mode for continuous operation
        if device.isFocusModeSupported(.continuousAutoFocus) {
            device.focusMode = .continuousAutoFocus
        }
        
        // Set optimal exposure mode
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }
        
        // Enable subject area change monitoring for responsive focus
        device.isSubjectAreaChangeMonitoringEnabled = true
        
        // Configure white balance for better color accuracy
        if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            device.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        
        // Allow full focus range for both near and far objects
        if device.isAutoFocusRangeRestrictionSupported {
            device.autoFocusRangeRestriction = .none
        }
        
        // Enable smooth auto focus if available (iOS 7+)
        if device.isSmoothAutoFocusSupported {
            device.isSmoothAutoFocusEnabled = true
        }
        
        // Configure for macro focus if supported (iOS 13.0+)
        if #available(iOS 13.0, *) {
            if device.isAutoFocusRangeRestrictionSupported {
                // Set to none to allow both macro and normal focus ranges
                device.autoFocusRangeRestriction = .none
            }
        }
        
        // Enable low light boost for better focus in challenging conditions
        if device.isLowLightBoostSupported {
            device.automaticallyEnablesLowLightBoostWhenAvailable = true
        }
        
        device.unlockForConfiguration()
    }
    
    func takePhotoWithAVFoundation(){
        //self.captureSession.sessionPreset = AVCaptureSession.Preset.hd4K3840x2160
        let photoSettings: AVCapturePhotoSettings
        
        // Use HEIF format if available for better quality
        if #available(iOS 11.0, *), self.photoOutput.availablePhotoCodecTypes.contains(.hevc) {
            photoSettings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.hevc])
        } else {
            photoSettings = AVCapturePhotoSettings()
        }
        
        photoSettings.isHighResolutionPhotoEnabled = true
        
        // Enable auto-focus before capture for better focus accuracy
        if #available(iOS 11.0, *) {
            photoSettings.photoQualityPrioritization = .quality
        }
        
        // Enable auto-focus and auto-exposure for optimal capture
        if #available(iOS 14.1, *) {
            photoSettings.isAutoContentAwareDistortionCorrectionEnabled = true
        }
        
        // Enhanced focus before capture for better close-up performance
        let device = self.videoInput.device
        if device.isFocusModeSupported(.autoFocus) {
            do {
                try device.lockForConfiguration()
                
                // Store previous settings
                let previousFocusMode = device.focusMode
                let previousExposureMode = device.exposureMode
                
                // Configure for optimal photo capture
                device.focusMode = .autoFocus
                
                // Ensure full focus range for close objects
                if device.isAutoFocusRangeRestrictionSupported {
                    device.autoFocusRangeRestriction = .none
                }
                
                // Set exposure mode for photo capture
                if device.isExposureModeSupported(.autoExpose) {
                    device.exposureMode = .autoExpose
                }
                
                device.unlockForConfiguration()
                
                // Wait longer for focus to settle, especially for close objects
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.photoOutput.capturePhoto(with: photoSettings, delegate: self)
                    
                    // Restore previous settings after capture
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                        do {
                            try device.lockForConfiguration()
                            device.focusMode = previousFocusMode
                            device.exposureMode = previousExposureMode
                            device.unlockForConfiguration()
                        } catch {
                            print("Could not restore camera settings: \(error)")
                        }
                    }
                }
            } catch {
                // If focus configuration fails, capture anyway
                print("Could not configure focus for capture: \(error)")
                self.photoOutput.capturePhoto(with: photoSettings, delegate: self)
            }
        } else {
            // Capture immediately if auto focus isn't supported
            self.photoOutput.capturePhoto(with: photoSettings, delegate: self)
        }
    }
    
    public func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("Error:", error)
        } else {
            if let imageData = photo.fileDataRepresentation() {
                var ret = PluginCallResultData()
                var url:URL
                let pathToSave = takePhotoCall.getString("pathToSave", "")
                if pathToSave == "" {
                    url = FileManager.default.temporaryDirectory
                        .appendingPathComponent(UUID().uuidString)
                        .appendingPathExtension("jpeg")
                }else{
                    url = URL(string: "file://\(pathToSave)")!
                }
                if takePhotoCall.getBool("includeBase64", false) {
                    let image = UIImage(data: imageData)
                    let base64 = getBase64FromImage(image: image!, quality: 100.0)
                    ret["base64"] = base64
                }
                do {
                    
                    try imageData.write(to: url)
                    ret["path"] = url.path
                } catch {
                    print(error)
                }
                takePhotoCall.resolve(ret)
                takePhotoCall = nil
            }
        }
    }
    
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection)
    {
        if triggerPlayRequired || getResolutionCall != nil {
            var ret = PluginCallResultData()
            let imageBuffer:CVImageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)!
                    CVPixelBufferLockBaseAddress(imageBuffer, .readOnly)
            let width = CVPixelBufferGetWidth(imageBuffer)
            let height = CVPixelBufferGetHeight(imageBuffer)
            let res = String(width)+"x"+String(height)
            ret["resolution"] = res
            if triggerPlayRequired {
                notifyListeners("onPlayed", data: ret)
                triggerPlayRequired = false
            }
            if getResolutionCall != nil {
                getResolutionCall.resolve(ret)
                getResolutionCall = nil
            }
        }
        if takeSnapshotCall != nil || saveFrameCall != nil {
            guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                print("Failed to get image buffer from sample buffer.")
                return
            }
            let ciImage = CIImage(cvPixelBuffer: imageBuffer)
            guard let cgImage = CIContext().createCGImage(ciImage, from: ciImage.extent) else {
                print("Failed to create bitmap from image.")
                return
            }

            //print("lastValidOrientation: ",lastValidOrientation)
            //print("degree: ",degree)
            let image = UIImage(cgImage: cgImage)
            var normalized = normalizedImage(image)
            if self.scanRegion != nil {
                normalized = croppedUIImage(image: normalized, scanRegion: self.scanRegion)
            }
            if takeSnapshotCall != nil {
                let base64 = getBase64FromImage(image: normalized, quality: 100.0);
                var ret = PluginCallResultData()
                ret["base64"] = base64
                takeSnapshotCall.resolve(ret)
                takeSnapshotCall = nil
            }
            if saveFrameCall != nil {
                CameraPreviewPlugin.frameTaken = normalized
                var ret = PluginCallResultData()
                ret["success"] = true
                saveFrameCall.resolve(ret)
                saveFrameCall = nil
            }
        }
    }
    
    func makeWebViewTransparent(){
        DispatchQueue.main.async {
           self.bridge?.webView!.isOpaque = false
           self.bridge?.webView!.backgroundColor = UIColor.clear
           self.bridge?.webView!.scrollView.backgroundColor = UIColor.clear
       }
    }
    func restoreWebViewBackground(){
        DispatchQueue.main.async {
           self.bridge?.webView!.isOpaque = true
           self.bridge?.webView!.backgroundColor = UIColor.white
           self.bridge?.webView!.scrollView.backgroundColor = UIColor.white
       }
    }
    
    @objc func toggleTorch(_ call: CAPPluginCall) {
        let device = videoInput.device
        if device.hasTorch {
            do {
                try device.lockForConfiguration()
                if call.getBool("on", true) == true {
                    device.torchMode = .on
                } else {
                    device.torchMode = .off
                }
                device.unlockForConfiguration()
            } catch {
                print("Torch could not be used")
            }
        }
        call.resolve()
    }
    
    @objc func stopCamera(_ call: CAPPluginCall) {
        restoreWebViewBackground()
        DispatchQueue.main.sync {
            destroyCaptureSession()
        }
        call.resolve()
    }
    
    
    @objc func setResolution(_ call: CAPPluginCall) {
        let res = call.getInt("resolution", 5)
        let running = self.captureSession.isRunning
        if running {
            self.captureSession.stopRunning()
        }
        if (res == 1){
            self.captureSession.sessionPreset = AVCaptureSession.Preset.vga640x480
        } else if (res == 2){
            self.captureSession.sessionPreset = AVCaptureSession.Preset.hd1280x720
        } else if (res == 3 && facingBack == true){
            self.captureSession.sessionPreset = AVCaptureSession.Preset.hd1920x1080
        } else if (res == 5 && facingBack == true){
            self.captureSession.sessionPreset = AVCaptureSession.Preset.hd4K3840x2160
        }
        if running {
            self.captureSession.startRunning()
            triggerOnPlayed()
        }
        call.resolve()
    }
    
    @objc func getResolution(_ call: CAPPluginCall) {
        call.keepAlive = true
        getResolutionCall = call
    }
    
    @objc func triggerOnPlayed() {
        triggerPlayRequired = true
    }
    
    @objc func getAllCameras(_ call: CAPPluginCall) {
        var ret = PluginCallResultData()
        let array = NSMutableArray();
        array.add("Front-Facing Camera")
        array.add("Back-Facing Camera")
        ret["cameras"] = array
        call.resolve(ret)
    }
    
    @objc func getSelectedCamera(_ call: CAPPluginCall) {
        var ret = PluginCallResultData()
        if facingBack {
            ret["selectedCamera"] = "Back-Facing Camera"
        }else{
            ret["selectedCamera"] = "Front-Facing Camera"
        }
        call.resolve(ret)
    }
    
    @objc func getOrientation(_ call: CAPPluginCall) {
        var ret = PluginCallResultData()
        if UIDevice.current.orientation.isLandscape {
            ret["orientation"] = "LANDSCAPE"
        }else {
            ret["orientation"] = "PORTRAIT"
        }
        call.resolve(ret)
    }
    
    @objc func selectCamera(_ call: CAPPluginCall) {
        let isRunning = self.captureSession.isRunning
        if isRunning {
            self.captureSession.stopRunning()
        }
        let cameraID = call.getString("cameraID", "Back-Facing Camera")
        if cameraID == "Back-Facing Camera" && facingBack == false {
            self.captureSession.removeInput(self.videoInput)
            let videoDevice = captureDevice(with: AVCaptureDevice.Position.back)
            self.videoInput = try? AVCaptureDeviceInput(device: videoDevice!)
            self.captureSession.addInput(self.videoInput)
            
            // Configure focus for the new camera
            if let device = videoDevice {
                try? configureCameraForOptimalFocus(device: device)
                self.currentCameraDevice = device
            }
            
            facingBack = true
        }
        if cameraID == "Front-Facing Camera" && facingBack == true {
            self.captureSession.removeInput(self.videoInput)
            self.captureSession.sessionPreset = AVCaptureSession.Preset.photo
            let videoDevice = captureDevice(with: AVCaptureDevice.Position.front)
            self.videoInput = try? AVCaptureDeviceInput(device: videoDevice!)
            self.captureSession.addInput(self.videoInput)
            
            // Configure focus for the new camera
            if let device = videoDevice {
                try? configureCameraForOptimalFocus(device: device)
                self.currentCameraDevice = device
            }
            
            facingBack = false
        }
        if isRunning {
            self.captureSession.startRunning()
        }
        triggerOnPlayed()
        call.resolve()
    }
    
    @objc func setLayout(_ call: CAPPluginCall) {
        if (self.previewView == nil){
            call.reject("not initialized")
        }else{
            DispatchQueue.main.async {
                let left = self.getLayoutValue(call.getString("left")!,true)
                let top = self.getLayoutValue(call.getString("top")!,false)
                let width = self.getLayoutValue(call.getString("width")!,true)
                let height = self.getLayoutValue(call.getString("height")!,false)
                self.previewView.frame = CGRect.init(x: left, y: top, width: width, height: height)
            }
            call.resolve()
        }
    }
    
    func getLayoutValue(_ value: String,_ isWidth: Bool) -> CGFloat {
       if value.contains("%") {
           let percent = CGFloat(Float(String(value[..<value.lastIndex(of: "%")!]))!/100)
           if isWidth {
               return percent * (self.bridge?.webView!.frame.width)!
           }else{
               return percent * (self.bridge?.webView!.frame.height)!
           }
       }
       if value.contains("px") {
           let num = CGFloat(Float(String(value[..<value.lastIndex(of: "p")!]))!)
           return num
       }
       return CGFloat(Float(value)!)
   }
    
    @objc func startRecording(_ call: CAPPluginCall) {
        DispatchQueue.main.sync {
            if self.captureSession != nil {
                if self.captureSession.isRunning {
                    self.captureSession.stopRunning()
                    self.initializeCaptureSession(enableVideoRecording: true)
                    self.captureSession.startRunning()
                    self.movieFileOutput!.startRecording(to: self.getTemp(), recordingDelegate: self)
                }
            }
        }
        call.resolve()
    }
    
    public func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        if self.stopRecordingCall != nil {
            var ret = PluginCallResultData()
            ret["path"] = outputFileURL.path
            if self.stopRecordingCall.getBool("includeBase64", false) {
                let data = try? Data(contentsOf: outputFileURL)
                ret["base64"] = data?.base64EncodedString()
            }
            if self.captureSession != nil {
                if self.captureSession.isRunning {
                    self.captureSession.stopRunning()
                    self.initializeCaptureSession(enableVideoRecording: false)
                }
            }
            self.stopRecordingCall.resolve(ret)
            self.stopRecordingCall = nil
        }
    }
    
    private func getTemp() -> URL
    {
        let tempName = NSUUID().uuidString
        let tempPath = (NSTemporaryDirectory() as NSString).appendingPathComponent((tempName as NSString).appendingPathExtension("mov")!)
        
        print("Temp path: \(tempPath)")
        
        return URL(fileURLWithPath: tempPath)
    }
    
    @objc func stopRecording(_ call: CAPPluginCall) {
        call.keepAlive = true
        self.stopRecordingCall = call
        self.movieFileOutput.stopRecording()
    }
    
    func captureDevice(with position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        // Enhanced device discovery including all modern camera types
        var deviceTypes: [AVCaptureDevice.DeviceType] = [
            .builtInWideAngleCamera,
            .builtInTelephotoCamera,
            .builtInDualCamera
        ]
        
        // Add newer device types for iOS 13+
        if #available(iOS 13.0, *) {
            deviceTypes.append(.builtInUltraWideCamera)
            deviceTypes.append(.builtInDualWideCamera)
            deviceTypes.append(.builtInTripleCamera)
        }
        
        // Add macro camera for iOS 15+
        if #available(iOS 15.4, *) {
            deviceTypes.append(.builtInLiDARDepthCamera)
        }

        let devices = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: AVMediaType.video,
            position: .unspecified
        ).devices

        // Prefer newer multi-camera systems first
        for device in devices {
            if device.position == position {
                if #available(iOS 13.0, *) {
                    if device.deviceType == .builtInTripleCamera || device.deviceType == .builtInDualWideCamera {
                        return device
                    }
                }
            }
        }
        
        // Fallback to any available camera with the specified position
        for device in devices {
            if device.position == position {
                return device
            }
        }

        return nil
    }
    

    
    private func getBestAvailableCameraDevice() -> AVCaptureDevice? {
        let position: AVCaptureDevice.Position = facingBack ? .back : .front
        
        // Preferred camera types in order
        let preferredTypes: [AVCaptureDevice.DeviceType] = [
            .builtInTripleCamera,      // Best - has all cameras
            .builtInDualWideCamera,    // Good - has Ultra Wide + Wide
            .builtInDualCamera,        // OK - has Wide + Telephoto
            .builtInWideAngleCamera    // Fallback - standard camera
        ]
        
        let allDeviceTypes: [AVCaptureDevice.DeviceType] = [
            .builtInTripleCamera,
            .builtInDualWideCamera,
            .builtInDualCamera,
            .builtInWideAngleCamera,
            .builtInTelephotoCamera,
            .builtInUltraWideCamera
        ]
        
        let devices = AVCaptureDevice.DiscoverySession(
            deviceTypes: allDeviceTypes,
            mediaType: AVMediaType.video,
            position: .unspecified
        ).devices
        
        // Try to find preferred camera types first
        for preferredType in preferredTypes {
            for device in devices {
                if device.position == position && device.deviceType == preferredType {
                    return device
                }
            }
        }
        
        // Fallback to any available camera
        for device in devices {
            if device.position == position {
                return device
            }
        }
        
        return AVCaptureDevice.default(for: .video)
    }
    
    @objc func setScanRegion(_ call: CAPPluginCall) {
        let region = call.getObject("region")
        self.scanRegion = ScanRegion()
        self.scanRegion.top = region?["top"] as! Int
        self.scanRegion.right = region?["right"] as! Int
        self.scanRegion.left = region?["left"] as! Int
        self.scanRegion.bottom = region?["bottom"] as! Int
        self.scanRegion.measuredByPercentage = region?["measuredByPercentage"] as! Int
        call.resolve()
    }
    
    @objc func setZoom(_ call: CAPPluginCall) {
        let device = videoInput.device
        do {
            try device.lockForConfiguration()
            var factor:CGFloat = CGFloat(call.getFloat("factor") ?? 1.0)
            factor = max(factor, device.minAvailableVideoZoomFactor)
            factor = min(factor, device.maxAvailableVideoZoomFactor)
            device.videoZoomFactor = factor
            device.unlockForConfiguration()
        } catch {
            print("Zoom could not be used")
        }
        call.resolve()
    }
    
    @objc func setFocus(_ call: CAPPluginCall) {
        if let x = call.getFloat("x"), let y = call.getFloat("y"),
           x >= 0.0 && x <= 1.0, y >= 0.0 && y <= 1.0 {
            let point = CGPoint(x: CGFloat(x), y: CGFloat(y))
            
            // Check if focus is currently animating and reset if stuck
            if isFocusAnimating {
                resetFocusIfStuck()
            }
            
            focusWithPoint(point: point)
            
            // Calculate the point in the preview layer's coordinate space
            let previewPoint = CGPoint(x: point.x * previewView.bounds.width,
                                     y: point.y * previewView.bounds.height)
            showFocusView(at: previewPoint)
            call.resolve()
        } else {
            call.reject("Invalid coordinates. Provide normalized x,y values (0.0-1.0)")
        }
    }
    
    private func resetFocusIfStuck() {
        // Remove any existing focus indicator
        focusView?.removeFromSuperview()
        focusCompletionTimer?.invalidate()
        isFocusAnimating = false
        
        // Reset focus to continuous mode
        let device = self.videoInput.device
        do {
            try device.lockForConfiguration()
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            device.unlockForConfiguration()
        } catch {
            print("Could not reset focus: \(error)")
        }
    }
    
    @objc func resetFocus(_ call: CAPPluginCall) {
        resetFocusIfStuck()
        
        // Reset to center focus
        let centerPoint = CGPoint(x: 0.5, y: 0.5)
        focusWithPoint(point: centerPoint)
        
        call.resolve()
    }
    
    @objc func handleTapToFocus(_ gesture: UITapGestureRecognizer) {
        let location = gesture.location(in: self.previewView)
        let convertedPoint = self.previewView.videoPreviewLayer.captureDevicePointConverted(fromLayerPoint: location)
        
        focusWithPoint(point: convertedPoint)
        showFocusView(at: location)
    }
    
    func focusWithPoint(point: CGPoint) {
        let device = self.videoInput.device
        
        let now = Date()
        if now.timeIntervalSince(lastFocusTime) < focusThrottleInterval {
            return
        }
        lastFocusTime = now
        
        do {
            try device.lockForConfiguration()
            
            focusCompletionTimer?.invalidate()
            
            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = point
                
                // Use autoFocus for more aggressive focusing on specific points
                if device.isFocusModeSupported(.autoFocus) {
                    device.focusMode = .autoFocus
                    
                    // Set up observer for focus completion
                    NotificationCenter.default.addObserver(
                        self,
                        selector: #selector(subjectAreaDidChange),
                        name: .AVCaptureDeviceSubjectAreaDidChange,
                        object: device
                    )
                } else if device.isFocusModeSupported(.continuousAutoFocus) {
                    device.focusMode = .continuousAutoFocus
                }
            }
            
            if device.isExposurePointOfInterestSupported {
                device.exposurePointOfInterest = point
                
                // Use autoExpose for specific point exposure
                if device.isExposureModeSupported(.autoExpose) {
                    device.exposureMode = .autoExpose
                } else if device.isExposureModeSupported(.continuousAutoExposure) {
                    device.exposureMode = .continuousAutoExposure
                }
            }
            
            // Ensure full focus range is available for close objects
            if device.isAutoFocusRangeRestrictionSupported {
                device.autoFocusRangeRestriction = .none
            }
            
            device.isSubjectAreaChangeMonitoringEnabled = true
            
            device.unlockForConfiguration()
            
            // Switch back to continuous focus after a delay to maintain automatic focus
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                self?.returnToContinuousFocus()
            }
            
            focusCompletionTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: false) { [weak self] _ in
                DispatchQueue.main.async {
                    self?.hideFocusIndicatorWithCompletion()
                }
            }
            
        } catch {
            print("Could not focus: \(error.localizedDescription)")
        }
    }
    
    @objc private func subjectAreaDidChange(notification: NSNotification) {
        DispatchQueue.main.async { [weak self] in
            self?.hideFocusIndicatorWithCompletion()
        }
        
        NotificationCenter.default.removeObserver(self, name: .AVCaptureDeviceSubjectAreaDidChange, object: notification.object)
    }
    
    private func returnToContinuousFocus() {
        let device = self.videoInput.device
        do {
            try device.lockForConfiguration()
            
            // Return to continuous auto focus for automatic operation
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            
            // Return to continuous auto exposure
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            
            device.unlockForConfiguration()
        } catch {
            print("Could not return to continuous focus: \(error)")
        }
    }
    

    
    private func hideFocusIndicatorWithCompletion() {
        guard let focusView = self.focusView else { return }
        
        UIView.animate(withDuration: 0.2, animations: {
            focusView.alpha = 0.0
            focusView.transform = CGAffineTransform(scaleX: 0.8, y: 0.8)
        }) { _ in
            focusView.removeFromSuperview()
            focusView.transform = CGAffineTransform.identity
            self.isFocusAnimating = false
        }
    }
    
    func showFocusView(at point: CGPoint) {
        if isFocusAnimating {
            self.focusView?.removeFromSuperview()
            focusCompletionTimer?.invalidate()
        }
        
        // Create focus view if needed - but make it invisible
        if self.focusView == nil {
            self.focusView = UIView(frame: CGRect(x: 0, y: 0, width: 80, height: 80))
            // Make the focus view completely transparent
            self.focusView?.layer.borderColor = UIColor.clear.cgColor
            self.focusView?.layer.borderWidth = 0.0
            self.focusView?.layer.cornerRadius = 40
            self.focusView?.backgroundColor = .clear
            self.focusView?.alpha = 0.0
            
            // Remove the inner circle to make it completely invisible
            // No inner circle added
        }
        
        self.focusView?.center = point
        self.focusView?.alpha = 0.0  // Keep invisible
        self.focusView?.transform = CGAffineTransform.identity
        self.previewView.addSubview(self.focusView!)
        
        self.isFocusAnimating = true
        
        // Skip the animation since the view is invisible
        // Focus functionality still works, just no visual feedback
    }
    
    @objc func requestCameraPermission(_ call: CAPPluginCall) {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                if granted {
                    // Initialize capture session now that permission is granted
                    if self.previewView != nil && self.captureSession == nil {
                        self.initializeCaptureSession(enableVideoRecording: false)
                    }
                    call.resolve(["granted": true])
                } else {
                    call.resolve(["granted": false])
                }
            }
        }
    }

    @objc func requestMicroPhonePermission(_ call: CAPPluginCall) {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                call.resolve(["granted": granted])
            }
        }
    }
    

    

    
    @objc func isOpen(_ call: CAPPluginCall) {
        var ret = PluginCallResultData()
        ret["isOpen"] = self.captureSession.isRunning
        call.resolve(ret)
    }
    
    @objc static func getBitmap() -> UIImage? {
        return frameTaken
    }
    
    @objc func saveFrame(_ call: CAPPluginCall) {
        call.keepAlive = true
        saveFrameCall = call
    }
    
    @objc func takeSnapshot(_ call: CAPPluginCall) {
        call.keepAlive = true
        takeSnapshotCall = call
    }
    
    func croppedUIImage(image:UIImage,scanRegion:ScanRegion) -> UIImage {
        let cgImage = image.cgImage
        let imgWidth = Double(cgImage!.width)
        let imgHeight = Double(cgImage!.height)
        var regionLeft = Double(scanRegion.left)
        var regionTop = Double(scanRegion.top)
        var regionWidth = Double(scanRegion.right - scanRegion.left)
        var regionHeight = Double(scanRegion.bottom - scanRegion.top)
        if scanRegion.measuredByPercentage == 1 {
            regionLeft = regionLeft / 100  * imgWidth
            regionTop = regionTop / 100  * imgHeight
            regionWidth = regionWidth / 100  * imgWidth
            regionHeight = regionHeight / 100  * imgHeight
        }
        
        // The cropRect is the rect of the image to keep,
        // in this case centered
        let cropRect = CGRect(
            x: regionLeft,
            y: regionTop,
            width: regionWidth,
            height: regionHeight
        ).integral

        let cropped = cgImage?.cropping(
            to: cropRect
        )!
        let image = UIImage(cgImage: cropped!)
        return image
    }
    
    func rotatedUIImage(image:UIImage, degree: Int) -> UIImage {
        var rotatedImage = UIImage()
        switch degree
        {
            case 90:
                rotatedImage = UIImage(cgImage: image.cgImage!, scale: 1.0, orientation: .right)
            case 180:
                rotatedImage = UIImage(cgImage: image.cgImage!, scale: 1.0, orientation: .down)
            default:
                return image
        }
        return rotatedImage
    }
    
    
    
    func normalizedImage(_ image:UIImage) -> UIImage {
        if image.imageOrientation == UIImage.Orientation.up {
            return image
        }
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(in: CGRect(x:0,y:0,width:image.size.width,height:image.size.height))
        let normalized = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext();
        return normalized
    }
    
    func getBase64FromImage(image:UIImage, quality: CGFloat) -> String{
       let dataTmp = image.jpegData(compressionQuality: quality)
       if let data = dataTmp {
           return data.base64EncodedString()
       }
       return ""
    }
    
    @objc func takePhoto(_ call: CAPPluginCall) {
        call.keepAlive = true
        takePhotoCall = call
        takePhotoWithAVFoundation()
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
        focusCompletionTimer?.invalidate()
    }
    
}

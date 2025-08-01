import { WebPlugin } from '@capacitor/core';
import { CameraEnhancer, PlayCallbackInfo } from 'dynamsoft-camera-enhancer';
import { CameraPreviewPlugin, EnumResolution, ScanRegion } from './definitions';
import RecordRTC from 'recordrtc';
CameraEnhancer.defaultUIElementURL = "https://cdn.jsdelivr.net/npm/dynamsoft-camera-enhancer@3.3.9/dist/dce.ui.html";

export class CameraPreviewWeb extends WebPlugin implements CameraPreviewPlugin {
  private camera:CameraEnhancer | undefined;
  private container:HTMLElement | undefined;
  private region:ScanRegion | undefined;
  private recorder:RecordRTC | null | undefined;
  private hasMicrophone: boolean = false;
  async setDefaultUIElementURL(url: string): Promise<void> {
    CameraEnhancer.defaultUIElementURL = url;
  }

  async setElement(ele: HTMLElement): Promise<void> {
    this.container = ele;
  }

  saveFrame(): Promise<{ success: boolean; }> {
    throw new Error('Method not implemented.');
  }

  async getOrientation(): Promise<{"orientation":"PORTRAIT"|"LANDSCAPE"}> {
    let portrait = window.matchMedia("(orientation: portrait)");
    if (portrait.matches) {
      return {orientation:"PORTRAIT"};
    }else{
      return {orientation:"LANDSCAPE"};
    }
  }

  private desiredJpegQuality: number = 0.95; // Default to high quality (0.0-1.0)

  async initialize(options?: { quality?: number }): Promise<void> {
    // Get quality parameter from initialization, default to 95% if not specified
    if (options?.quality !== undefined) {
      this.desiredJpegQuality = Math.max(1, Math.min(100, options.quality)) / 100.0;
      console.log(`Camera initialized with JPEG quality: ${this.desiredJpegQuality}`);
    }
    
    this.camera = await CameraEnhancer.createInstance();
    this.camera.on("played", (playCallBackInfo:PlayCallbackInfo) => {
      this.notifyListeners("onPlayed", {resolution:playCallBackInfo.width+"x"+playCallBackInfo.height});
      try {
        let canvas = this.camera!.getUIElement().getElementsByClassName("dce-video-container")[0].getElementsByTagName("canvas")[0];
        if (canvas) {
          canvas.remove();
        }
      } catch (error) {
        console.log(error);
      }
    });
    if (this.container) {
      await this.camera.setUIElement(this.container);
    }else{
      await this.camera.setUIElement(CameraEnhancer.defaultUIElementURL);
      this.camera.getUIElement().getElementsByClassName("dce-btn-close")[0].remove();
      this.camera.getUIElement().getElementsByClassName("dce-sel-camera")[0].remove();
      this.camera.getUIElement().getElementsByClassName("dce-sel-resolution")[0].remove();
      this.camera.getUIElement().getElementsByClassName("dce-msg-poweredby")[0].remove();
    }
    this.camera.setVideoFit("cover");
    let portrait = window.matchMedia("(orientation: portrait)");
    portrait.addEventListener("change", () => {
      this.notifyListeners("onOrientationChanged", null);
    })
  }

  async getResolution(): Promise<{ resolution: string; }> {
    if (this.camera) {
      let rsl = this.camera.getResolution();
      let resolution:string = rsl[0] + "x" + rsl[1];
      return {resolution: resolution};
    }else{
      throw new Error('Camera not initialized');
    }
  }

  async setResolution(options: { resolution: number; }): Promise<void> {
    if (this.camera) {
      let res = options.resolution;
      let width = 1280;
      let height = 720;
      if (res == EnumResolution.RESOLUTION_480P){
         width = 640;
         height = 480;
      } else if (res == EnumResolution.RESOLUTION_720P){
        width = 1280;
        height = 720;
      } else if (res == EnumResolution.RESOLUTION_1080P){
        width = 1920;
        height = 1080;
      } else if (res == EnumResolution.RESOLUTION_2K){
        width = 2560;
        height = 1440;
      } else if (res == EnumResolution.RESOLUTION_4K){
        width = 3840;
        height = 2160;
      }
      await this.camera.setResolution(width,height);
      return;
    } else {
      throw new Error('Camera not initialized');
    }
  }

  async getAllCameras(): Promise<{ cameras: string[]; }> {
    if (this.camera) {
      let cameras = await this.camera.getAllCameras();
      let labels:string[] = [];
      cameras.forEach(camera => {
        labels.push(camera.label);
      });
      return {cameras:labels};
    }else {
      throw new Error('Camera not initialized');
    }
  }

  async getSelectedCamera(): Promise<{ selectedCamera: string; }> {
    if (this.camera) {
      let cameraInfo = this.camera.getSelectedCamera();
      return {selectedCamera:cameraInfo.label};
    }else {
      throw new Error('Camera not initialized');
    }
  }

  async selectCamera(options: { cameraID: string; }): Promise<void> {
    if (this.camera) {
      let cameras = await this.camera.getAllCameras()
      for (let index = 0; index < cameras.length; index++) {
        const camera = cameras[index];
        if (camera.label === options.cameraID) {
          await this.camera.selectCamera(camera);
          return;
        }
      }
    }else {
      throw new Error('Camera not initialized');
    }
  }

  async setScanRegion(options: { region: ScanRegion; }): Promise<void> {
    if (this.camera){
      this.region = options.region;
      this.applyScanRegion();
    }else {
      throw new Error('Camera not initialized');
    }
  }

  applyScanRegion(){
    if (this.camera && this.region){
      this.camera.setScanRegion({
        regionLeft:this.region.left,
        regionTop:this.region.top,
        regionRight:this.region.right,
        regionBottom:this.region.bottom,
        regionMeasuredByPercentage: this.region.measuredByPercentage
      });
    }
  }

  async setZoom(options: { factor: number; }): Promise<void> {
    if (this.camera) {
      await this.camera.setZoom(options.factor);
      return;
    }else {
      throw new Error('Camera not initialized');
    }
  }

  async setFocus(options: {x: number, y: number}): Promise<void> {
    if (this.camera) {
      this.camera.setFocus({
        mode:"manual",
        area:{
          centerPoint:{x:options.x.toString(),y:options.y.toString()}
        }
      });
    }else{
      throw new Error('Camera not initialized');
    }
  }

  async toggleTorch(options: { on: boolean; }): Promise<void> {
    if (this.camera) {
      try{
        if (options["on"]){
          await this.camera.turnOnTorch();
        }else{
          await this.camera.turnOffTorch();
        }
      } catch (e){
        throw new Error("Torch unsupported");
      }
    }
  }

  async startCamera(): Promise<void> {
    if (this.camera) {
      await this.camera?.open(true);
      if (this.container && this.isSafari() === true) {
        const resetZoom = async () => {
          await this.camera?.setZoom(1.001);
          await this.camera?.setZoom(1.0);
        }
        setTimeout(resetZoom,500);
      }
    }else {
      throw new Error('Camera not initialized');
    }
  }

  isSafari():boolean{
    const u = navigator.userAgent.toLowerCase()
    if (u.indexOf('safari') > -1 && u.indexOf('chrome') === -1) {
      return true;
    }else{
      return false;
    }
  }

  async stopCamera(): Promise<void> {
    if (this.camera) {
      this.camera.close(true);
    }else {
      throw new Error('Camera not initialized');
    }
  }

  async isOpen(): Promise<{isOpen:boolean}> {
    if (this.camera) {
      return {isOpen:this.camera.isOpen()};
    }else{
      throw new Error('Camera not initialized');
    }
  }

  async takeSnapshot(options:{quality?:number, checkBlur?:boolean}): Promise<{ base64: string, isBlur?: boolean }> {
    if (this.camera) {
      let desiredQuality = this.desiredJpegQuality;
      if (options?.quality !== undefined) {
        desiredQuality = Math.max(1, Math.min(100, options.quality)) / 100.0;
      }
      let canvas = this.camera.getFrame().toCanvas();
      let dataURL = canvas.toDataURL('image/jpeg', desiredQuality);
      let base64 = dataURL.replace("data:image/jpeg;base64,","");
      
      // Only perform blur detection if checkBlur option is true
      if (options?.checkBlur === true) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const blurResult = this.detectBlurWeb(imageData, canvas.width, canvas.height);
          return {
            base64: base64,
            isBlur: blurResult.blurScore < 150
          };
        }
      }
      
      return {base64: base64};
    }else{
      throw new Error('Camera not initialized');
    }
  }

  async detectBlur(options: {image: string}): Promise<{isBlur: boolean, blurConfidence: number, sharpConfidence: number}> {
    try {
      // Create image element from base64
      const img = new Image();
      
      return new Promise((resolve, reject) => {
        img.onload = () => {
          try {
            // Create canvas to extract image data
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            if (!ctx) {
              reject(new Error('Cannot create canvas context'));
              return;
            }
            
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            // Get image data
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const blurResult = this.detectBlurWeb(imageData, canvas.width, canvas.height);
            
            // Convert blur score to confidence values
            // Higher blurScore means sharper image
            const normalizedScore = Math.max(0, Math.min(1, blurResult.blurScore / 300)); // Normalize to 0-1
            const sharpConfidence = normalizedScore;
            const blurConfidence = 1 - normalizedScore;
            const isBlur = blurResult.blurScore < 150;
            
            resolve({
              isBlur: isBlur,
              blurConfidence: blurConfidence,
              sharpConfidence: sharpConfidence
            });
          } catch (error) {
            reject(error);
          }
        };
        
        img.onerror = () => {
          reject(new Error('Failed to load image'));
        };
        
        // Handle both data URLs and base64 strings
        if (options.image.startsWith('data:')) {
          img.src = options.image;
        } else {
          img.src = `data:image/jpeg;base64,${options.image}`;
        }
      });
    } catch (error) {
      throw new Error(`Failed to detect blur: ${error}`);
    }
  }

  async takeSnapshot2(options:{canvas:HTMLCanvasElement,maxLength?:number}): Promise<{scaleRatio?:number}> {
    if (this.camera) {
      let canvas = options.canvas;
      let scaleRatio = 1.0;
      if (options && options.maxLength) {
        let res = (await this.getResolution()).resolution;
        let width = parseInt(res.split("x")[0]);
        let height = parseInt(res.split("x")[1]);
        let targetWidth = width;
        let targetHeight = height;
        if (width > options.maxLength || height > options.maxLength) {
          if (width > height) {
            targetWidth = options.maxLength;
            targetHeight = options.maxLength/width*height;
            scaleRatio = options.maxLength/width;
          }else{
            targetHeight = options.maxLength;
            targetWidth = options.maxLength/height*width;
            scaleRatio = options.maxLength/height;
          }
          canvas.width = targetWidth;
          canvas.height = targetHeight;
          let video = this.camera.getUIElement().getElementsByTagName("video")[0];
          let ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
          }
        }else{
          this.drawFullFrame(canvas);
        }
      }else{
        this.drawFullFrame(canvas);
      }
      return {scaleRatio:scaleRatio};
    }else {
      throw new Error('Camera not initialized');
    }
  }

  drawFullFrame(canvas:HTMLCanvasElement):HTMLCanvasElement{
    let video = this.camera?.getUIElement().getElementsByTagName("video")[0];
    if (video) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      let ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
    }
    return canvas;
  }

  async takePhoto(_options:{includeBase64?:boolean}): Promise<{ path?:string, base64?: string, blob?:Blob, blurScore?: number }> {
    if (this.camera) {
      let video = this.camera.getUIElement().getElementsByTagName("video")[0];
      let localStream:MediaStream = video.srcObject as MediaStream;
      if (localStream) {
        if ("ImageCapture" in window) {
          try {
            //@ts-ignore 
            let ImageCapture:any = window["ImageCapture"];
            console.log("ImageCapture supported");
            const track = localStream.getVideoTracks()[0];
            let imageCapture = new ImageCapture(track);
            let blob = await imageCapture.takePhoto();
            return {blob:blob};  
          } catch (error) {
            console.log(error);
          }
        }else{
          console.log("ImageCapture not supported");
        }
      }
      this.camera.setScanRegion(
        {
          regionLeft: 0,
          regionTop: 0,
          regionBottom: 100,
          regionRight: 100,
          regionMeasuredByPercentage: 1
        }
      )
      let canvas = this.camera.getFrame().toCanvas();
      let base64 = this.removeDataURLHead(canvas.toDataURL("image/jpeg", this.desiredJpegQuality));
      this.applyScanRegion();
      
      // Add blur detection if base64 is included
      if (_options?.includeBase64) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const blurResult = this.detectBlurWeb(imageData, canvas.width, canvas.height);
          return {
            base64: base64,
            blurScore: blurResult.blurScore
          };
        }
      }
      
      return {base64:base64};
    }else {
      throw new Error('Camera not initialized');
    }
  }

  removeDataURLHead(dataURL:string){
    return dataURL.substring(dataURL.indexOf(",")+1,dataURL.length);
  }


  async requestCameraPermission(): Promise<void> {
    const constraints = {video: true, audio: false};
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    const tracks = stream.getTracks();
    for (let i=0;i<tracks.length;i++) {
      const track = tracks[i];
      track.stop();  // stop the opened tracks
    }
  }

  async requestMicroPhonePermission(): Promise<void> {
    try {
      const constraints = {video: false, audio: true};
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      const tracks = stream.getTracks();
      for (let i=0;i<tracks.length;i++) {
        const track = tracks[i];
        track.stop();  // stop the opened tracks
      }
      this.hasMicrophone = true;
    } catch (error) {
      this.hasMicrophone = false;
      throw error;
    }
  }

  async startRecording(): Promise<void> {
    if (this.camera) {
      if (this.hasMicrophone) {
        let settings = this.camera.getVideoSettings();
        settings.audio = true;
        await this.camera.updateVideoSettings(settings);
      }
      let video = this.camera.getUIElement().getElementsByTagName("video")[0];
      this.recorder = new RecordRTC(video.srcObject as MediaStream, {
        type: 'video'
      });
      this.recorder.startRecording();
    }else{
      throw new Error("camera not initialized");
    }
  }

  async stopRecording(): Promise<{blob:Blob}> {
    return new Promise<{blob:Blob}>((resolve, reject) => {
      if (this.recorder) {
        const stopRecordingCallback = () => {
          let blob = this.recorder!.getBlob();
          this.recorder!.destroy();
          this.recorder = null;
          resolve({blob})
        }
        this.recorder.stopRecording(stopRecordingCallback);
      }else{
        reject("recorder not initialized");
      }
    })    
  }

  async setLayout(options: {top: string,left:string,width:string, height:string}): Promise<void> {
    if (this.camera) {
      let ele = this.camera.getUIElement();
      if (options.top) {
        ele.style.top = options.top;
      }
      if (options.left) {
        ele.style.left = options.left;
      }
      if (options.width) {
        ele.style.width = options.width;
      }
      if (options.height) {
        ele.style.height = options.height;
      }
      ele.style.position = "absolute";
    }else{
      throw new Error("Camera not initialized");
    }
  }

  /**
   * Detect blur using Laplacian variance algorithm (same as iOS/Android)
   * Higher values indicate sharper images
   */
  private detectBlurWeb(imageData: ImageData, width: number, height: number): { blurScore: number } {
    const data = imageData.data;
    let variance = 0;
    let count = 0;
    
    // Sample every 4th pixel for performance (similar to Android implementation)
    const step = 4;
    for (let y = step; y < height - step; y += step) {
      for (let x = step; x < width - step; x += step) {
        const idx = (y * width + x) * 4;
        
        // Convert to grayscale using same formula as Android
        const gray = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        
        // Calculate neighbors for 3x3 Laplacian kernel
        const neighbors = [
          0.299 * data[((y-1) * width + (x-1)) * 4] + 0.587 * data[((y-1) * width + (x-1)) * 4 + 1] + 0.114 * data[((y-1) * width + (x-1)) * 4 + 2],
          0.299 * data[((y-1) * width + x) * 4] + 0.587 * data[((y-1) * width + x) * 4 + 1] + 0.114 * data[((y-1) * width + x) * 4 + 2],
          0.299 * data[((y-1) * width + (x+1)) * 4] + 0.587 * data[((y-1) * width + (x+1)) * 4 + 1] + 0.114 * data[((y-1) * width + (x+1)) * 4 + 2],
          0.299 * data[(y * width + (x-1)) * 4] + 0.587 * data[(y * width + (x-1)) * 4 + 1] + 0.114 * data[(y * width + (x-1)) * 4 + 2],
          0.299 * data[(y * width + (x+1)) * 4] + 0.587 * data[(y * width + (x+1)) * 4 + 1] + 0.114 * data[(y * width + (x+1)) * 4 + 2],
          0.299 * data[((y+1) * width + (x-1)) * 4] + 0.587 * data[((y+1) * width + (x-1)) * 4 + 1] + 0.114 * data[((y+1) * width + (x-1)) * 4 + 2],
          0.299 * data[((y+1) * width + x) * 4] + 0.587 * data[((y+1) * width + x) * 4 + 1] + 0.114 * data[((y+1) * width + x) * 4 + 2],
          0.299 * data[((y+1) * width + (x+1)) * 4] + 0.587 * data[((y+1) * width + (x+1)) * 4 + 1] + 0.114 * data[((y+1) * width + (x+1)) * 4 + 2]
        ];
        
        // Apply 3x3 Laplacian kernel (matches Android implementation)
        const laplacian = -neighbors[0] - neighbors[1] - neighbors[2] +
                         -neighbors[3] + 8 * gray - neighbors[4] +
                         -neighbors[5] - neighbors[6] - neighbors[7];
        
        variance += laplacian * laplacian;
        count++;
      }
    }
    
    const blurScore = count > 0 ? variance / count : 0;
    
    return { blurScore };
  }
}

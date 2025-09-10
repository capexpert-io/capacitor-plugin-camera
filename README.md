# capacitor-plugin-camera

A capacitor camera plugin.

[Online demo](https://dazzling-cactus-692f14.netlify.app/)

## Supported Platforms

* Android (based on CameraX)
* iOS (based on AVCaptureSession)
* Web (based on getUserMedia with Dynamsoft Camera Enhancer)

## Versions

For Capacitor 5, use versions 1.x.

For Capacitor 6, use versions 2.x.

For Capacitor 7, use versions 3.x.

## Install

```bash
npm install capacitor-plugin-camera
npx cap sync
```
## Get Bitmap/UIImage via Reflection

If you are developing a plugin, you can use reflection to get the camera frames as Bitmap or UIImage on the native side.

Java:

```java
Class cls = Class.forName("com.tonyxlh.capacitor.camera.CameraPreviewPlugin");
Method m = cls.getMethod("getBitmap",null);
Bitmap bitmap = (Bitmap) m.invoke(null, null);
```

Objective-C:

```obj-c
- (UIImage*)getUIImage{
    UIImage *image = ((UIImage* (*)(id, SEL))objc_msgSend)(objc_getClass("CameraPreviewPlugin"), sel_registerName("getBitmap"));
    return image;
}
```

You have to call `saveFrame` beforehand.

## Declare Permissions

To use camera and microphone, we need to declare permissions.

Add the following to Android's `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Add the following to iOS's `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>For camera usage</string>
<key>NSMicrophoneUsageDescription</key>
<string>For video recording</string>
```

## FAQ

Why I cannot see the camera?

For native platforms, the plugin puts the native camera view behind the webview and sets the webview as transparent so that we can display HTML elements above the camera.

You may need to add the style below on your app's HTML or body element to avoid blocking the camera view:

```css
ion-content {
  --background: transparent;
}
```

In dark mode, it is neccessary to set the `--ion-blackground-color` property. You can do this with the following code:

```js
document.documentElement.style.setProperty('--ion-background-color', 'transparent');
```

## API

<docgen-index>

* [`initialize(...)`](#initialize)
* [`getResolution()`](#getresolution)
* [`setResolution(...)`](#setresolution)
* [`getAllCameras()`](#getallcameras)
* [`getSelectedCamera()`](#getselectedcamera)
* [`selectCamera(...)`](#selectcamera)
* [`setScanRegion(...)`](#setscanregion)
* [`setZoom(...)`](#setzoom)
* [`setFocus(...)`](#setfocus)
* [`setDefaultUIElementURL(...)`](#setdefaultuielementurl)
* [`setElement(...)`](#setelement)
* [`startCamera()`](#startcamera)
* [`stopCamera()`](#stopcamera)
* [`takeSnapshot(...)`](#takesnapshot)
* [`detectBlur(...)`](#detectblur)
* [`saveFrame()`](#saveframe)
* [`takeSnapshot2(...)`](#takesnapshot2)
* [`takePhoto(...)`](#takephoto)
* [`toggleTorch(...)`](#toggletorch)
* [`getOrientation()`](#getorientation)
* [`startRecording()`](#startrecording)
* [`stopRecording(...)`](#stoprecording)
* [`setLayout(...)`](#setlayout)
* [`requestCameraPermission()`](#requestcamerapermission)
* [`requestMicroPhonePermission()`](#requestmicrophonepermission)
* [`isOpen()`](#isopen)
* [`addListener('onPlayed', ...)`](#addlisteneronplayed-)
* [`addListener('onOrientationChanged', ...)`](#addlisteneronorientationchanged-)
* [`removeAllListeners()`](#removealllisteners)
* [Interfaces](#interfaces)
* [Type Aliases](#type-aliases)

</docgen-index>

<docgen-api>
<!--Update the source file JSDoc comments and rerun docgen to update the docs below-->

### initialize(...)

```typescript
initialize(options?: { quality?: number | undefined; } | undefined) => Promise<void>
```

| Param         | Type                               |
| ------------- | ---------------------------------- |
| **`options`** | <code>{ quality?: number; }</code> |

--------------------


### getResolution()

```typescript
getResolution() => Promise<{ resolution: string; }>
```

**Returns:** <code>Promise&lt;{ resolution: string; }&gt;</code>

--------------------


### setResolution(...)

```typescript
setResolution(options: { resolution: number; }) => Promise<void>
```

| Param         | Type                                 |
| ------------- | ------------------------------------ |
| **`options`** | <code>{ resolution: number; }</code> |

--------------------


### getAllCameras()

```typescript
getAllCameras() => Promise<{ cameras: string[]; }>
```

**Returns:** <code>Promise&lt;{ cameras: string[]; }&gt;</code>

--------------------


### getSelectedCamera()

```typescript
getSelectedCamera() => Promise<{ selectedCamera: string; }>
```

**Returns:** <code>Promise&lt;{ selectedCamera: string; }&gt;</code>

--------------------


### selectCamera(...)

```typescript
selectCamera(options: { cameraID: string; }) => Promise<void>
```

| Param         | Type                               |
| ------------- | ---------------------------------- |
| **`options`** | <code>{ cameraID: string; }</code> |

--------------------


### setScanRegion(...)

```typescript
setScanRegion(options: { region: ScanRegion; }) => Promise<void>
```

| Param         | Type                                                           |
| ------------- | -------------------------------------------------------------- |
| **`options`** | <code>{ region: <a href="#scanregion">ScanRegion</a>; }</code> |

--------------------


### setZoom(...)

```typescript
setZoom(options: { factor: number; }) => Promise<void>
```

| Param         | Type                             |
| ------------- | -------------------------------- |
| **`options`** | <code>{ factor: number; }</code> |

--------------------


### setFocus(...)

```typescript
setFocus(options: { x: number; y: number; }) => Promise<void>
```

| Param         | Type                                   |
| ------------- | -------------------------------------- |
| **`options`** | <code>{ x: number; y: number; }</code> |

--------------------


### setDefaultUIElementURL(...)

```typescript
setDefaultUIElementURL(url: string) => Promise<void>
```

Web Only

| Param     | Type                |
| --------- | ------------------- |
| **`url`** | <code>string</code> |

--------------------


### setElement(...)

```typescript
setElement(ele: any) => Promise<void>
```

Web Only

| Param     | Type             |
| --------- | ---------------- |
| **`ele`** | <code>any</code> |

--------------------


### startCamera()

```typescript
startCamera() => Promise<void>
```

--------------------


### stopCamera()

```typescript
stopCamera() => Promise<void>
```

--------------------


### takeSnapshot(...)

```typescript
takeSnapshot(options: { quality?: number; checkBlur?: boolean; }) => Promise<{ base64: string; confidence?: number; }>
```

take a snapshot as base64.

| Param         | Type                                                    |
| ------------- | ------------------------------------------------------- |
| **`options`** | <code>{ quality?: number; checkBlur?: boolean; }</code> |

**Returns:** <code>Promise&lt;{ base64: string; confidence?: number; }&gt;</code>

--------------------


### detectBlur(...)

```typescript
detectBlur(options: { image: string; }) => Promise<{ isBlur: boolean; blurConfidence: number; sharpConfidence: number; }>
```

analyze an image for blur detection with detailed confidence scores.

| Param         | Type                            |
| ------------- | ------------------------------- |
| **`options`** | <code>{ image: string; }</code> |

**Returns:** <code>Promise&lt;{ isBlur: boolean; blurConfidence: number; sharpConfidence: number; }&gt;</code>

--------------------


### saveFrame()

```typescript
saveFrame() => Promise<{ success: boolean; }>
```

save a frame internally. Android and iOS only.

**Returns:** <code>Promise&lt;{ success: boolean; }&gt;</code>

--------------------


### takeSnapshot2(...)

```typescript
takeSnapshot2(options: { canvas: HTMLCanvasElement; maxLength?: number; }) => Promise<{ scaleRatio?: number; }>
```

take a snapshot on to a canvas. Web Only

| Param         | Type                                              |
| ------------- | ------------------------------------------------- |
| **`options`** | <code>{ canvas: any; maxLength?: number; }</code> |

**Returns:** <code>Promise&lt;{ scaleRatio?: number; }&gt;</code>

--------------------


### takePhoto(...)

```typescript
takePhoto(options: { pathToSave?: string; includeBase64?: boolean; }) => Promise<{ path?: string; base64?: string; blob?: Blob; confidence?: number; }>
```

| Param         | Type                                                           |
| ------------- | -------------------------------------------------------------- |
| **`options`** | <code>{ pathToSave?: string; includeBase64?: boolean; }</code> |

**Returns:** <code>Promise&lt;{ path?: string; base64?: string; blob?: any; confidence?: number; }&gt;</code>

--------------------


### toggleTorch(...)

```typescript
toggleTorch(options: { on: boolean; }) => Promise<void>
```

| Param         | Type                          |
| ------------- | ----------------------------- |
| **`options`** | <code>{ on: boolean; }</code> |

--------------------


### getOrientation()

```typescript
getOrientation() => Promise<{ "orientation": "PORTRAIT" | "LANDSCAPE"; }>
```

get the orientation of the device.

**Returns:** <code>Promise&lt;{ orientation: 'PORTRAIT' | 'LANDSCAPE'; }&gt;</code>

--------------------


### startRecording()

```typescript
startRecording() => Promise<void>
```

--------------------


### stopRecording(...)

```typescript
stopRecording(options: { includeBase64?: boolean; }) => Promise<{ path?: string; base64?: string; blob?: Blob; }>
```

| Param         | Type                                      |
| ------------- | ----------------------------------------- |
| **`options`** | <code>{ includeBase64?: boolean; }</code> |

**Returns:** <code>Promise&lt;{ path?: string; base64?: string; blob?: any; }&gt;</code>

--------------------


### setLayout(...)

```typescript
setLayout(options: { top: string; left: string; width: string; height: string; }) => Promise<void>
```

| Param         | Type                                                                       |
| ------------- | -------------------------------------------------------------------------- |
| **`options`** | <code>{ top: string; left: string; width: string; height: string; }</code> |

--------------------


### requestCameraPermission()

```typescript
requestCameraPermission() => Promise<void>
```

--------------------


### requestMicroPhonePermission()

```typescript
requestMicroPhonePermission() => Promise<void>
```

--------------------


### isOpen()

```typescript
isOpen() => Promise<{ isOpen: boolean; }>
```

**Returns:** <code>Promise&lt;{ isOpen: boolean; }&gt;</code>

--------------------


### addListener('onPlayed', ...)

```typescript
addListener(eventName: 'onPlayed', listenerFunc: onPlayedListener) => Promise<PluginListenerHandle>
```

| Param              | Type                                                          |
| ------------------ | ------------------------------------------------------------- |
| **`eventName`**    | <code>'onPlayed'</code>                                       |
| **`listenerFunc`** | <code><a href="#onplayedlistener">onPlayedListener</a></code> |

**Returns:** <code>Promise&lt;<a href="#pluginlistenerhandle">PluginListenerHandle</a>&gt;</code>

--------------------


### addListener('onOrientationChanged', ...)

```typescript
addListener(eventName: 'onOrientationChanged', listenerFunc: onOrientationChangedListener) => Promise<PluginListenerHandle>
```

| Param              | Type                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------- |
| **`eventName`**    | <code>'onOrientationChanged'</code>                                                   |
| **`listenerFunc`** | <code><a href="#onorientationchangedlistener">onOrientationChangedListener</a></code> |

**Returns:** <code>Promise&lt;<a href="#pluginlistenerhandle">PluginListenerHandle</a>&gt;</code>

--------------------


### removeAllListeners()

```typescript
removeAllListeners() => Promise<void>
```

--------------------


### Interfaces


#### ScanRegion

measuredByPercentage: 0 in pixel, 1 in percent

| Prop                       | Type                |
| -------------------------- | ------------------- |
| **`left`**                 | <code>number</code> |
| **`top`**                  | <code>number</code> |
| **`right`**                | <code>number</code> |
| **`bottom`**               | <code>number</code> |
| **`measuredByPercentage`** | <code>number</code> |


#### PluginListenerHandle

| Prop         | Type                                      |
| ------------ | ----------------------------------------- |
| **`remove`** | <code>() =&gt; Promise&lt;void&gt;</code> |


### Type Aliases


#### onPlayedListener

<code>(result: { resolution: string; }): void</code>


#### onOrientationChangedListener

<code>(): void</code>

</docgen-api>

## Blur Detection

The plugin includes blur detection capabilities using TensorFlow Lite models with Laplacian variance fallback, providing consistent results across all platforms.

#### Analyze Existing Images

Use the `detectBlur` method to analyze any base64 image with detailed confidence scores:

```typescript
// Analyze an existing image (base64 string or data URL)
const result = await CameraPreview.detectBlur({
  image: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."
  // or just the base64 string: "/9j/4AAQSkZJRgABAQEASABIAAD..."
});

console.log('Is Blurry:', result.isBlur);           // boolean: true/false
console.log('Blur Confidence:', result.blurConfidence);   // 0.0-1.0 (higher = more blurry)  
console.log('Sharp Confidence:', result.sharpConfidence); // 0.0-1.0 (higher = more sharp)

// Use confidence scores for advanced logic
if (result.blurConfidence > 0.7) {
  console.log('High confidence this image is blurry');
} else if (result.sharpConfidence > 0.8) {
  console.log('High confidence this image is sharp');
} else {
  console.log('Uncertain blur status - manual review needed');
}
```

#### Basic Usage (Capture + Detection)

```typescript
// Take a snapshot with blur detection
const result = await CameraPreview.takeSnapshot({
  quality: 85,
  checkBlur: true  // Optional, defaults to false for performance
});

console.log('Base64:', result.base64);
if (result.blurScore !== undefined) {
  console.log('Blur Score:', result.blurScore);
  
  // Implement your own blur threshold logic
  const threshold = 50.0; // Adjust based on your quality requirements
  const isBlurry = result.blurScore < threshold;
  
  if (isBlurry) {
    console.log('Image appears to be blurry');
  } else {
    console.log('Image appears to be sharp');
  }
}
```

#### Performance Control

Blur detection is **disabled by default** for optimal performance. Enable it only when needed:

```typescript
// Blur detection OFF (default) - faster performance
const result = await CameraPreview.takeSnapshot({ quality: 85 });

// Blur detection ON - includes blur analysis
const resultWithBlur = await CameraPreview.takeSnapshot({ 
  quality: 85, 
  checkBlur: true 
});
```

#### Understanding Blur Results

**New `detectBlur` Method (Recommended):**
- Returns standardized **confidence scores** (0.0-1.0 range) across all platforms
- `blurConfidence`: Higher values indicate more blur (>0.7 = likely blurry)
- `sharpConfidence`: Higher values indicate more sharpness (>0.8 = likely sharp)  
- `isBlur`: Simple boolean result based on confidence thresholds

**Legacy `takeSnapshot` Method:**
- **Higher values** = Sharper images
- **Lower values** = Blurrier images
- **Threshold guidelines**: 
  - iOS: Consider values below `0.001` as blurry
  - Android/Web: Consider values below `50-100` as blurry
  - Adjust thresholds based on your specific quality requirements

#### When to Use Which Method

**Use `detectBlur` for:**
- Analyzing already captured images
- Batch processing multiple images  
- Getting detailed confidence scores for advanced decision logic
- Post-processing workflows
- When you need consistent confidence values across platforms

**Use `takeSnapshot` with `checkBlur: true` for:**
- Real-time capture with immediate blur feedback
- Simple blur detection during image capture
- When you only need a basic blur/sharp indication

#### Performance Impact

| Platform | Without Blur Detection | With Blur Detection | Overhead |
|----------|----------------------|-------------------|----------|
| iOS | 100-120ms | 120-145ms | ~20% |
| Android | 80-120ms | 100-145ms | ~21% |
| Web | 60-100ms | 85-140ms | ~40% |

#### Implementation Notes

- **TensorFlow Lite models** for advanced blur detection with high accuracy
- **Laplacian variance fallback** when TFLite models unavailable
- Pixel sampling for performance optimization  
- Hardware acceleration on iOS with Core Image
- Client-side threshold logic for maximum flexibility
- Cross-platform algorithm consistency
- **Dual API**: `takeSnapshot` for capture+detection, `detectBlur` for analyzing existing images

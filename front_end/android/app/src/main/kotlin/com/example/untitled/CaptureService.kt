package com.example.untitled

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.util.DisplayMetrics
import android.util.Size
import android.view.WindowManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import java.io.ByteArrayOutputStream
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/**
 * Foreground service that captures frames from either the device camera
 * (Camera2) or the device screen (MediaProjection), throttles them to a target
 * FPS, encodes each to JPEG and POSTs it to the configured server endpoint.
 *
 * Because all capture and networking happen inside this service (not a Flutter
 * isolate), it keeps running while the app is minimized.
 */
class CaptureService : Service() {

    companion object {
        const val ACTION_START = "com.example.untitled.capture.START"
        const val ACTION_STOP = "com.example.untitled.capture.STOP"
        const val ACTION_SET_FPS = "com.example.untitled.capture.SET_FPS"

        const val EXTRA_SOURCE = "source"        // "camera" | "screen"
        const val EXTRA_FPS = "fps"              // Int
        const val EXTRA_URL = "url"              // String, e.g. http://ip:8000/capture/frame
        const val EXTRA_LENS = "lens"            // "back" | "front"

        private const val CHANNEL_ID = "frame_capture"
        private const val NOTIFICATION_ID = 42

        @Volatile var isRunning: Boolean = false
            private set
    }

    // --- Config -------------------------------------------------------------
    @Volatile private var source: String = "camera"
    @Volatile private var targetFps: Int = 5
    @Volatile private var endpointUrl: String = ""
    @Volatile private var lens: String = "back"

    private val frameCount = AtomicLong(0)
    @Volatile private var lastSentMs: Long = 0
    @Volatile private var lastError: String? = null

    // --- Threads ------------------------------------------------------------
    private var captureThread: HandlerThread? = null
    private var captureHandler: Handler? = null

    // Single-slot poster: only the latest frame matters; drop older ones.
    private val posterQueue = LinkedBlockingQueue<ByteArray>(1)
    private val poster = ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, LinkedBlockingQueue())
    @Volatile private var posterRunning = false

    // --- Camera2 ------------------------------------------------------------
    private var cameraDevice: CameraDevice? = null
    private var cameraSession: CameraCaptureSession? = null
    private var cameraReader: ImageReader? = null
    private var cameraSensorOrientation: Int = 0

    // --- Screen -------------------------------------------------------------
    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var screenReader: ImageReader? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopCapture()
                stopSelf()
                return START_NOT_STICKY
            }
            ACTION_SET_FPS -> {
                targetFps = (intent.getIntExtra(EXTRA_FPS, targetFps)).coerceIn(1, 60)
                emit()
                return START_STICKY
            }
            ACTION_START -> {
                source = intent.getStringExtra(EXTRA_SOURCE) ?: "camera"
                targetFps = intent.getIntExtra(EXTRA_FPS, 5).coerceIn(1, 60)
                endpointUrl = intent.getStringExtra(EXTRA_URL) ?: ""
                lens = intent.getStringExtra(EXTRA_LENS) ?: "back"
                startCapture()
                return START_STICKY
            }
        }
        return START_NOT_STICKY
    }

    // --- Lifecycle ----------------------------------------------------------
    private fun startCapture() {
        if (isRunning) stopCapture()
        lastError = null
        frameCount.set(0)
        startForegroundWithType()

        captureThread = HandlerThread("capture").also { it.start() }
        captureHandler = Handler(captureThread!!.looper)

        posterRunning = true
        poster.execute { posterLoop() }

        isRunning = true
        try {
            when (source) {
                "screen" -> startScreen()
                else -> startCamera()
            }
        } catch (t: Throwable) {
            lastError = t.message ?: t.toString()
            emit()
        }
        emit()
    }

    private fun stopCapture() {
        isRunning = false
        posterRunning = false
        posterQueue.clear()

        try { cameraSession?.close() } catch (_: Throwable) {}
        try { cameraDevice?.close() } catch (_: Throwable) {}
        try { cameraReader?.close() } catch (_: Throwable) {}
        cameraSession = null; cameraDevice = null; cameraReader = null

        try { virtualDisplay?.release() } catch (_: Throwable) {}
        try { screenReader?.close() } catch (_: Throwable) {}
        try { mediaProjection?.stop() } catch (_: Throwable) {}
        virtualDisplay = null; screenReader = null; mediaProjection = null

        captureThread?.quitSafely()
        captureThread = null; captureHandler = null

        emit()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
    }

    override fun onDestroy() {
        stopCapture()
        poster.shutdownNow()
        super.onDestroy()
    }

    // --- Camera2 engine -----------------------------------------------------
    @Suppress("MissingPermission")
    private fun startCamera() {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val wantFront = lens == "front"
        var chosenId: String? = null
        for (id in manager.cameraIdList) {
            val ch = manager.getCameraCharacteristics(id)
            val facing = ch.get(CameraCharacteristics.LENS_FACING)
            val isFront = facing == CameraCharacteristics.LENS_FACING_FRONT
            if (isFront == wantFront) { chosenId = id; break }
            if (chosenId == null) chosenId = id
        }
        val cameraId = chosenId ?: throw IllegalStateException("No camera available")
        val ch = manager.getCameraCharacteristics(cameraId)
        cameraSensorOrientation = ch.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0

        val size = pickCameraSize(ch)
        cameraReader = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, 2).apply {
            setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    if (shouldSend()) {
                        val jpeg = yuv420ToJpeg(image, cameraSensorOrientation)
                        if (jpeg != null) enqueue(jpeg)
                    }
                } catch (t: Throwable) {
                    lastError = t.message
                } finally {
                    image.close()
                }
            }, captureHandler)
        }

        manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                cameraDevice = device
                val surface = cameraReader!!.surface
                device.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        cameraSession = session
                        val req = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                            addTarget(surface)
                            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                        }
                        try {
                            session.setRepeatingRequest(req.build(), null, captureHandler)
                        } catch (t: Throwable) {
                            lastError = t.message; emit()
                        }
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        lastError = "Camera session configure failed"; emit()
                    }
                }, captureHandler)
            }
            override fun onDisconnected(device: CameraDevice) { device.close(); cameraDevice = null }
            override fun onError(device: CameraDevice, error: Int) {
                lastError = "Camera error $error"; device.close(); cameraDevice = null; emit()
            }
        }, captureHandler)
    }

    private fun pickCameraSize(ch: CameraCharacteristics): Size {
        val map = ch.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val sizes = map?.getOutputSizes(ImageFormat.YUV_420_888) ?: return Size(640, 480)
        // Prefer a moderate resolution around 720p to keep JPEGs small.
        return sizes.filter { it.width <= 1280 && it.height <= 1280 }
            .maxByOrNull { it.width.toLong() * it.height } ?: sizes.minByOrNull { it.width.toLong() * it.height } ?: Size(640, 480)
    }

    // --- Screen engine ------------------------------------------------------
    private fun startScreen() {
        if (!CaptureBridge.hasProjectionConsent()) {
            throw IllegalStateException("Screen capture consent not granted")
        }
        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        val projection = mpm.getMediaProjection(
            CaptureBridge.projectionResultCode,
            CaptureBridge.projectionResultData!!,
        ) ?: throw IllegalStateException("Failed to obtain MediaProjection")
        mediaProjection = projection

        projection.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                lastError = "Screen capture stopped"
                stopCapture()
            }
        }, captureHandler)

        val metrics = DisplayMetrics()
        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)

        // Downscale large screens so JPEGs stay reasonable.
        var w = metrics.widthPixels
        var h = metrics.heightPixels
        val maxDim = 1280
        val scale = maxOf(w, h).let { if (it > maxDim) maxDim.toFloat() / it else 1f }
        w = (w * scale).toInt().coerceAtLeast(2)
        h = (h * scale).toInt().coerceAtLeast(2)
        // Round to even dimensions.
        if (w % 2 == 1) w -= 1
        if (h % 2 == 1) h -= 1

        screenReader = ImageReader.newInstance(w, h, android.graphics.PixelFormat.RGBA_8888, 2).apply {
            setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    if (shouldSend()) {
                        val jpeg = rgbaToJpeg(image)
                        if (jpeg != null) enqueue(jpeg)
                    }
                } catch (t: Throwable) {
                    lastError = t.message
                } finally {
                    image.close()
                }
            }, captureHandler)
        }

        virtualDisplay = projection.createVirtualDisplay(
            "capture",
            w, h, metrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            screenReader!!.surface,
            null,
            captureHandler,
        )
    }

    // --- FPS throttle & poster ---------------------------------------------
    private fun shouldSend(): Boolean {
        val now = System.currentTimeMillis()
        val intervalMs = 1000L / targetFps.coerceIn(1, 60)
        if (now - lastSentMs < intervalMs) return false
        lastSentMs = now
        return true
    }

    private fun enqueue(jpeg: ByteArray) {
        // Keep only the newest frame.
        posterQueue.poll()
        posterQueue.offer(jpeg)
    }

    private fun posterLoop() {
        while (posterRunning) {
            val jpeg = try {
                posterQueue.poll(500, TimeUnit.MILLISECONDS) ?: continue
            } catch (_: InterruptedException) {
                break
            }
            postFrame(jpeg)
        }
    }

    private fun postFrame(jpeg: ByteArray) {
        if (endpointUrl.isBlank()) return
        var conn: HttpURLConnection? = null
        try {
            val seq = frameCount.get()
            val url = URL("$endpointUrl?source=$source&seq=$seq")
            conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                doOutput = true
                connectTimeout = 4000
                readTimeout = 8000
                setRequestProperty("Content-Type", "image/jpeg")
                setRequestProperty("X-Source", source)
                setFixedLengthStreamingMode(jpeg.size)
            }
            val out: OutputStream = conn.outputStream
            out.write(jpeg)
            out.flush()
            out.close()
            val code = conn.responseCode
            conn.inputStream.use { it.readBytes() }
            if (code in 200..299) {
                val n = frameCount.incrementAndGet()
                if (n % targetFps.coerceAtLeast(1) == 0L) emit() // ~once per second
            } else {
                lastError = "Server responded $code"; emit()
            }
        } catch (t: Throwable) {
            lastError = t.message ?: t.toString()
            emit()
        } finally {
            conn?.disconnect()
        }
    }

    private fun emit() {
        CaptureBridge.emitStatus(isRunning, if (isRunning) source else null, targetFps, frameCount.get(), lastError)
    }

    // --- Encoding helpers ---------------------------------------------------
    /** Convert a YUV_420_888 [Image] to a JPEG, rotating by [sensorOrientation]. */
    private fun yuv420ToJpeg(image: Image, sensorOrientation: Int): ByteArray? {
        val nv21 = yuv420ToNv21(image)
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val jpeg = ByteArrayOutputStream()
        if (!yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 70, jpeg)) return null
        val bytes = jpeg.toByteArray()
        // Rotate to upright if needed.
        return if (sensorOrientation % 360 != 0) rotateJpeg(bytes, sensorOrientation, lens == "front") else bytes
    }

    private fun yuv420ToNv21(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val nv21 = ByteArray(ySize + ySize / 2)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        // Y
        val yBuffer = yPlane.buffer
        val yRowStride = yPlane.rowStride
        var pos = 0
        if (yRowStride == width) {
            yBuffer.get(nv21, 0, ySize)
            pos = ySize
        } else {
            for (row in 0 until height) {
                yBuffer.position(row * yRowStride)
                yBuffer.get(nv21, pos, width)
                pos += width
            }
        }

        // Interleave V and U as VU (NV21).
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        for (row in 0 until chromaHeight) {
            var uvIndex = row * uvRowStride
            for (col in 0 until chromaWidth) {
                val vuPos = ySize + row * width + col * 2
                nv21[vuPos] = vBuffer.get(uvIndex)
                nv21[vuPos + 1] = uBuffer.get(uvIndex)
                uvIndex += uvPixelStride
            }
        }
        return nv21
    }

    private fun rgbaToJpeg(image: Image): ByteArray? {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * image.width
        val bmpWidth = image.width + rowPadding / pixelStride
        val bitmap = Bitmap.createBitmap(bmpWidth, image.height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)
        val cropped = if (rowPadding == 0) bitmap
            else Bitmap.createBitmap(bitmap, 0, 0, image.width, image.height)
        val out = ByteArrayOutputStream()
        cropped.compress(Bitmap.CompressFormat.JPEG, 60, out)
        if (cropped != bitmap) cropped.recycle()
        bitmap.recycle()
        return out.toByteArray()
    }

    private fun rotateJpeg(jpeg: ByteArray, degrees: Int, mirror: Boolean): ByteArray {
        val src = android.graphics.BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: return jpeg
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees.toFloat())
        if (mirror) matrix.postScale(-1f, 1f)
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
        val out = ByteArrayOutputStream()
        rotated.compress(Bitmap.CompressFormat.JPEG, 70, out)
        if (rotated != src) rotated.recycle()
        src.recycle()
        return out.toByteArray()
    }

    // --- Foreground notification -------------------------------------------
    private fun startForegroundWithType() {
        createChannel()
        val notification: Notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Frame capture running")
            .setContentText("Streaming $source frames")
            .setSmallIcon(android.R.drawable.presence_video_online)
            .setOngoing(true)
            .build()

        // FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION exists since Q (29);
        // FOREGROUND_SERVICE_TYPE_CAMERA since R (30). Only pass a type when the
        // running platform actually recognizes it, otherwise start untyped.
        val type: Int = when {
            source == "screen" && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q ->
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            source != "screen" && Build.VERSION.SDK_INT >= Build.VERSION_CODES.R ->
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
            else -> 0
        }
        if (type != 0) {
            startForeground(NOTIFICATION_ID, notification, type)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mgr = getSystemService(NotificationManager::class.java)
            if (mgr.getNotificationChannel(CHANNEL_ID) == null) {
                val channel = NotificationChannel(
                    CHANNEL_ID, "Frame capture", NotificationManager.IMPORTANCE_LOW,
                )
                channel.description = "Background frame capture and streaming"
                mgr.createNotificationChannel(channel)
            }
        }
    }
}

package com.example.untitled

import android.app.Activity
import android.content.Intent
import android.media.projection.MediaProjectionManager
import androidx.core.content.ContextCompat
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    private val methodChannelName = "com.example.untitled/capture"
    private val notificationChannelName = "com.example.untitled/notifications"
    private val eventChannelName = "com.example.untitled/capture_events"
    private val screenRequestCode = 9001

    private var pendingScreenResult: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, methodChannelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "requestScreenPermission" -> requestScreenPermission(result)
                    "hasScreenPermission" -> result.success(CaptureBridge.hasProjectionConsent())
                    "start" -> {
                        val source = call.argument<String>("source") ?: "camera"
                        val fps = call.argument<Int>("fps") ?: 5
                        val url = call.argument<String>("url") ?: ""
                        val lens = call.argument<String>("lens") ?: "back"
                        if (source == "screen" && !CaptureBridge.hasProjectionConsent()) {
                            result.error("NO_CONSENT", "Screen capture not authorized", null)
                        } else {
                            startCapture(source, fps, url, lens)
                            result.success(true)
                        }
                    }
                    "stop" -> {
                        val intent = Intent(this, CaptureService::class.java).apply {
                            action = CaptureService.ACTION_STOP
                        }
                        startService(intent)
                        result.success(true)
                    }
                    "setFps" -> {
                        val fps = call.argument<Int>("fps") ?: 5
                        val intent = Intent(this, CaptureService::class.java).apply {
                            action = CaptureService.ACTION_SET_FPS
                            putExtra(CaptureService.EXTRA_FPS, fps)
                        }
                        startService(intent)
                        result.success(true)
                    }
                    "isRunning" -> result.success(CaptureService.isRunning)
                    else -> result.notImplemented()
                }
            }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, notificationChannelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "show" -> {
                        AlertNotifier.show(
                            this,
                            call.argument<Int>("id") ?: 0,
                            call.argument<String>("title") ?: "HomeMind",
                            call.argument<String>("body") ?: "",
                            call.argument<String>("severity") ?: "important",
                        )
                        result.success(true)
                    }
                    "startMonitoring" -> {
                        val apiBase = call.argument<String>("apiBase") ?: ""
                        val eventNotifications =
                            call.argument<Boolean>("eventNotifications") ?: true
                        val proactiveNotifications =
                            call.argument<Boolean>("proactiveNotifications") ?: false
                        val intent = Intent(this, AlertPollingService::class.java).apply {
                            putExtra(AlertPollingService.EXTRA_API_BASE, apiBase)
                            putExtra(
                                AlertPollingService.EXTRA_EVENT_NOTIFICATIONS,
                                eventNotifications,
                            )
                            putExtra(
                                AlertPollingService.EXTRA_PROACTIVE_NOTIFICATIONS,
                                proactiveNotifications,
                            )
                        }
                        ContextCompat.startForegroundService(this, intent)
                        result.success(true)
                    }
                    "stopMonitoring" -> {
                        stopService(Intent(this, AlertPollingService::class.java))
                        result.success(true)
                    }
                    else -> result.notImplemented()
                }
            }

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, eventChannelName)
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    CaptureBridge.setEventSink(events)
                }
                override fun onCancel(arguments: Any?) {
                    CaptureBridge.setEventSink(null)
                }
            })
    }

    private fun startCapture(source: String, fps: Int, url: String, lens: String) {
        val intent = Intent(this, CaptureService::class.java).apply {
            action = CaptureService.ACTION_START
            putExtra(CaptureService.EXTRA_SOURCE, source)
            putExtra(CaptureService.EXTRA_FPS, fps)
            putExtra(CaptureService.EXTRA_URL, url)
            putExtra(CaptureService.EXTRA_LENS, lens)
        }
        ContextCompat.startForegroundService(this, intent)
    }

    private fun requestScreenPermission(result: MethodChannel.Result) {
        if (CaptureBridge.hasProjectionConsent()) {
            result.success(true)
            return
        }
        pendingScreenResult = result
        val mpm = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        startActivityForResult(mpm.createScreenCaptureIntent(), screenRequestCode)
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == screenRequestCode) {
            val granted = resultCode == Activity.RESULT_OK && data != null
            if (granted) {
                CaptureBridge.projectionResultCode = resultCode
                CaptureBridge.projectionResultData = data
            } else {
                CaptureBridge.clearProjectionConsent()
            }
            pendingScreenResult?.success(granted)
            pendingScreenResult = null
        }
    }
}

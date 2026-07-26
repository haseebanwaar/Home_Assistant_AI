package com.example.untitled

import android.content.Intent
import android.os.Handler
import android.os.Looper
import io.flutter.plugin.common.EventChannel

/**
 * Process-wide bridge shared between [MainActivity] and [CaptureService].
 *
 * Holds the MediaProjection consent result (obtained by the Activity) so the
 * service can build the projection, and forwards status/error events from the
 * service back up to Flutter through an [EventChannel].
 */
object CaptureBridge {
    // MediaProjection consent, captured by MainActivity via startActivityForResult.
    @Volatile var projectionResultCode: Int = 0
    @Volatile var projectionResultData: Intent? = null

    fun hasProjectionConsent(): Boolean = projectionResultData != null

    fun clearProjectionConsent() {
        projectionResultCode = 0
        projectionResultData = null
    }

    // --- Event forwarding to Flutter ---------------------------------------
    @Volatile private var eventSink: EventChannel.EventSink? = null
    private val main = Handler(Looper.getMainLooper())

    fun setEventSink(sink: EventChannel.EventSink?) {
        eventSink = sink
    }

    /** Emit a status snapshot to Flutter on the main thread. */
    fun emitStatus(running: Boolean, source: String?, fps: Int, frames: Long, error: String?) {
        val sink = eventSink ?: return
        val payload = HashMap<String, Any?>()
        payload["type"] = "status"
        payload["running"] = running
        payload["source"] = source
        payload["fps"] = fps
        payload["frames"] = frames
        payload["error"] = error
        main.post { eventSink?.success(payload) }
    }
}

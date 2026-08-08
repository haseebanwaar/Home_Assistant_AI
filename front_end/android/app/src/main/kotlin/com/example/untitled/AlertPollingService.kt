package com.example.untitled

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.speech.tts.TextToSpeech
import androidx.core.app.NotificationCompat
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

class AlertPollingService : Service() {
    @Volatile private var running = false
    @Volatile private var apiBase = ""
    @Volatile private var eventNotifications = true
    @Volatile private var proactiveNotifications = false
    private var worker: Thread? = null
    private var textToSpeech: TextToSpeech? = null
    @Volatile private var ttsReady = false

    override fun onCreate() {
        super.onCreate()
        textToSpeech = TextToSpeech(this) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
        }
        createMonitorChannel()
        startForeground(MONITOR_NOTIFICATION_ID, monitorNotification())
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        val supplied = intent?.getStringExtra(EXTRA_API_BASE)?.trimEnd('/')
        if (!supplied.isNullOrBlank()) {
            apiBase = supplied
            prefs.edit().putString(KEY_API_BASE, supplied).apply()
        } else if (apiBase.isBlank()) {
            apiBase = prefs.getString(KEY_API_BASE, "") ?: ""
        }
        if (intent?.hasExtra(EXTRA_EVENT_NOTIFICATIONS) == true) {
            eventNotifications =
                intent.getBooleanExtra(EXTRA_EVENT_NOTIFICATIONS, true)
            prefs.edit()
                .putBoolean(KEY_EVENT_NOTIFICATIONS, eventNotifications).apply()
        } else {
            eventNotifications = prefs.getBoolean(KEY_EVENT_NOTIFICATIONS, true)
        }
        if (intent?.hasExtra(EXTRA_PROACTIVE_NOTIFICATIONS) == true) {
            proactiveNotifications =
                intent.getBooleanExtra(EXTRA_PROACTIVE_NOTIFICATIONS, false)
            prefs.edit()
                .putBoolean(KEY_PROACTIVE_NOTIFICATIONS, proactiveNotifications)
                .apply()
        } else {
            proactiveNotifications =
                prefs.getBoolean(KEY_PROACTIVE_NOTIFICATIONS, false)
        }
        if (!eventNotifications && !proactiveNotifications) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (worker?.isAlive != true) {
            running = true
            worker = thread(name = "homemind-alert-poller", isDaemon = true) {
                while (running) {
                    try {
                        pollEnabledChannels()
                    } catch (_: Exception) {
                        // Tailscale/home server can be temporarily unavailable.
                    }
                    try {
                        Thread.sleep(POLL_INTERVAL_MS)
                    } catch (_: InterruptedException) {
                        break
                    }
                }
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        running = false
        worker?.interrupt()
        worker = null
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        ttsReady = false
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun pollEnabledChannels() {
        if (eventNotifications) pollEvents()
        if (proactiveNotifications) pollProactive()
    }

    private fun pollEvents() {
        val base = apiBase
        if (base.isBlank()) return
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        val sequenceKey = "$KEY_SEQUENCE:$base"
        val hasBaseline = prefs.contains(sequenceKey)
        val since = prefs.getInt(sequenceKey, 0)
        val connection = URL("$base/notifications?since=$since&limit=50")
            .openConnection() as HttpURLConnection
        connection.connectTimeout = 5000
        connection.readTimeout = 5000
        connection.requestMethod = "GET"
        try {
            if (connection.responseCode != 200) return
            val payload = connection.inputStream.bufferedReader().use { it.readText() }
            val root = JSONObject(payload)
            val latest = root.optInt("latest_sequence", since)
            if (!hasBaseline) {
                prefs.edit().putInt(sequenceKey, latest).apply()
                return
            }
            val items = root.optJSONArray("notifications") ?: return
            // API returns newest first; notify oldest first so the newest alert
            // remains visually on top.
            for (index in items.length() - 1 downTo 0) {
                val item = items.getJSONObject(index)
                val sequence = item.optInt("sequence", 0)
                if (sequence <= since) continue
                AlertNotifier.show(
                    this,
                    9000 + sequence,
                    item.optString("title", "HomeMind"),
                    item.optString("body", ""),
                    item.optString("severity", "important"),
                )
                if (item.optBoolean("speak", false) && ttsReady) {
                    val speech = item.optString("body", item.optString("title", ""))
                    textToSpeech?.speak(
                        speech,
                        TextToSpeech.QUEUE_ADD,
                        null,
                        "task-reminder-$sequence",
                    )
                }
            }
            prefs.edit().putInt(sequenceKey, latest).apply()
        } finally {
            connection.disconnect()
        }
    }

    private fun pollProactive() {
        val base = apiBase
        if (base.isBlank()) return
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        val sequenceKey = "$KEY_PROACTIVE_SEQUENCE:$base"
        val hasBaseline = prefs.contains(sequenceKey)
        val since = prefs.getInt(sequenceKey, 0)
        val connection = URL("$base/proactive?since=$since")
            .openConnection() as HttpURLConnection
        connection.connectTimeout = 5000
        connection.readTimeout = 5000
        connection.requestMethod = "GET"
        try {
            if (connection.responseCode != 200) return
            val payload = connection.inputStream.bufferedReader().use { it.readText() }
            val root = JSONObject(payload)
            val latest = root.optInt("latest_id", since)
            if (!hasBaseline) {
                prefs.edit().putInt(sequenceKey, latest).apply()
                return
            }
            val items = root.optJSONArray("insights") ?: return
            for (index in items.length() - 1 downTo 0) {
                val item = items.getJSONObject(index)
                val id = item.optInt("id", 0)
                if (id <= since) continue
                val source = item.optString("source", "")
                    .removePrefix("camera:")
                    .replace('_', ' ')
                val title = if (source.isBlank()) {
                    "HomeMind insight"
                } else {
                    "HomeMind · $source"
                }
                AlertNotifier.show(
                    this,
                    19000 + id,
                    title,
                    item.optString("text", ""),
                    "proactive",
                )
            }
            prefs.edit().putInt(sequenceKey, latest).apply()
        } finally {
            connection.disconnect()
        }
    }

    private fun monitorNotification() =
        NotificationCompat.Builder(this, MONITOR_CHANNEL)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("HomeMind monitoring")
            .setContentText("Delivering enabled alerts and proactive insights")
            .setOngoing(true)
            .setSilent(true)
            .setContentIntent(
                PendingIntent.getActivity(
                    this,
                    MONITOR_NOTIFICATION_ID,
                    Intent(this, MainActivity::class.java),
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
                )
            )
            .build()

    private fun createMonitorChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(MONITOR_CHANNEL) == null) {
            manager.createNotificationChannel(
                NotificationChannel(
                    MONITOR_CHANNEL,
                    "HomeMind monitoring",
                    NotificationManager.IMPORTANCE_LOW,
                ).apply {
                    description = "Keeps remote home alerts active in the background"
                }
            )
        }
    }

    companion object {
        const val EXTRA_API_BASE = "api_base"
        const val EXTRA_EVENT_NOTIFICATIONS = "event_notifications"
        const val EXTRA_PROACTIVE_NOTIFICATIONS = "proactive_notifications"
        private const val PREFS = "homemind_alerts"
        private const val KEY_API_BASE = "api_base"
        private const val KEY_SEQUENCE = "last_sequence"
        private const val KEY_PROACTIVE_SEQUENCE = "last_proactive_sequence"
        private const val KEY_EVENT_NOTIFICATIONS = "event_notifications"
        private const val KEY_PROACTIVE_NOTIFICATIONS = "proactive_notifications"
        private const val MONITOR_CHANNEL = "homemind_monitor"
        private const val MONITOR_NOTIFICATION_ID = 8200
        private const val POLL_INTERVAL_MS = 15_000L
    }
}

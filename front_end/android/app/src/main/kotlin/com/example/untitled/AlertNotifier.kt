package com.example.untitled

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat

object AlertNotifier {
    private const val CRITICAL_CHANNEL = "homemind_critical"
    private const val IMPORTANT_CHANNEL = "homemind_important"

    fun show(
        context: Context,
        id: Int,
        title: String,
        body: String,
        severity: String,
    ) {
        createChannels(context)
        val critical = severity == "critical"
        val channel = if (critical) CRITICAL_CHANNEL else IMPORTANT_CHANNEL
        val openIntent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
            putExtra("open_notifications", true)
        }
        val pendingIntent = PendingIntent.getActivity(
            context,
            id,
            openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(context, channel)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .setCategory(
                if (critical) NotificationCompat.CATEGORY_ALARM
                else NotificationCompat.CATEGORY_RECOMMENDATION
            )
            .setPriority(
                if (critical) NotificationCompat.PRIORITY_MAX
                else NotificationCompat.PRIORITY_DEFAULT
            )
            .build()
        context.getSystemService(NotificationManager::class.java)
            .notify(id, notification)
    }

    private fun createChannels(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = context.getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CRITICAL_CHANNEL) == null) {
            manager.createNotificationChannel(
                NotificationChannel(
                    CRITICAL_CHANNEL,
                    "Critical home alerts",
                    NotificationManager.IMPORTANCE_HIGH,
                ).apply {
                    description = "Urgent safety and security events"
                    enableVibration(true)
                }
            )
        }
        if (manager.getNotificationChannel(IMPORTANT_CHANNEL) == null) {
            manager.createNotificationChannel(
                NotificationChannel(
                    IMPORTANT_CHANNEL,
                    "Important activity",
                    NotificationManager.IMPORTANCE_DEFAULT,
                ).apply {
                    description = "Noteworthy home and productivity events"
                }
            )
        }
    }
}

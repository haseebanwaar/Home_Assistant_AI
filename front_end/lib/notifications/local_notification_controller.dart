import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';

class LocalNotificationController {
  static const _channel =
      MethodChannel('com.example.untitled/notifications');

  bool get isSupported =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android;
  String? _monitoringSignature;
  bool _monitoringStateKnown = false;

  Future<void> initialize() async {
    if (!isSupported) return;
    await Permission.notification.request();
  }

  Future<void> show(Map<String, dynamic> notification) async {
    if (!isSupported) return;
    await _channel.invokeMethod('show', {
      'id': (notification['sequence'] as num?)?.toInt() ?? 0,
      'title': (notification['title'] ?? 'HomeMind').toString(),
      'body': (notification['body'] ?? '').toString(),
      'severity': (notification['severity'] ?? 'important').toString(),
    });
  }

  Future<void> startMonitoring(
    String apiBase, {
    required bool eventNotifications,
    required bool proactiveNotifications,
  }) async {
    final signature =
        '$apiBase|events=$eventNotifications|proactive=$proactiveNotifications';
    if (!isSupported || apiBase.isEmpty || _monitoringSignature == signature) {
      return;
    }
    await initialize();
    await _channel.invokeMethod('startMonitoring', {
      'apiBase': apiBase,
      'eventNotifications': eventNotifications,
      'proactiveNotifications': proactiveNotifications,
    });
    _monitoringSignature = signature;
    _monitoringStateKnown = true;
  }

  Future<void> stopMonitoring() async {
    if (!isSupported ||
        (_monitoringStateKnown && _monitoringSignature == null)) {
      return;
    }
    await _channel.invokeMethod('stopMonitoring');
    _monitoringSignature = null;
    _monitoringStateKnown = true;
  }
}

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';

/// Which device source to capture frames from.
enum CaptureSource { camera, screen }

/// Immutable snapshot of the native capture service status.
@immutable
class CaptureStatus {
  final bool running;
  final String? source; // "camera" | "screen" | null
  final int fps;
  final int frames; // frames successfully POSTed
  final int resizeFactor;
  final int width;
  final int height;
  final String? error;

  const CaptureStatus({
    required this.running,
    required this.source,
    required this.fps,
    required this.frames,
    required this.resizeFactor,
    required this.width,
    required this.height,
    required this.error,
  });

  static const idle = CaptureStatus(
    running: false,
    source: null,
    fps: 0,
    frames: 0,
    resizeFactor: 1,
    width: 0,
    height: 0,
    error: null,
  );

  factory CaptureStatus.fromMap(Map<dynamic, dynamic> m) => CaptureStatus(
    running: m['running'] == true,
    source: m['source'] as String?,
    fps: (m['fps'] as num?)?.toInt() ?? 0,
    frames: (m['frames'] as num?)?.toInt() ?? 0,
    resizeFactor: (m['resizeFactor'] as num?)?.toInt() ?? 1,
    width: (m['width'] as num?)?.toInt() ?? 0,
    height: (m['height'] as num?)?.toInt() ?? 0,
    error: m['error'] as String?,
  );
}

/// Dart-side wrapper around the native Android capture foreground service.
///
/// Talks to `MainActivity`/`CaptureService` over a [MethodChannel] and receives
/// status updates over an [EventChannel]. All heavy work (capture, JPEG
/// encoding, HTTP POST) happens natively so it survives app minimization.
class FrameCaptureController {
  static const _method = MethodChannel('com.example.untitled/capture');
  static const _events = EventChannel('com.example.untitled/capture_events');

  /// The native capture service only exists on Android. Everywhere else
  /// (web, desktop, iOS) the platform channels have no handler, so we must
  /// not touch them — doing so throws [MissingPluginException].
  static final bool _supported =
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  final _statusController = StreamController<CaptureStatus>.broadcast();
  StreamSubscription? _eventSub;
  CaptureStatus _last = CaptureStatus.idle;

  Stream<CaptureStatus> get status => _statusController.stream;
  CaptureStatus get lastStatus => _last;

  /// Whether native frame capture is available on this platform.
  bool get isSupported => _supported;

  FrameCaptureController() {
    if (!_supported) return;
    _eventSub = _events.receiveBroadcastStream().listen(
      (event) {
        if (event is Map && event['type'] == 'status') {
          _last = CaptureStatus.fromMap(event);
          _statusController.add(_last);
        }
      },
      onError: (e) {
        if (kDebugMode) print('capture event error: $e');
      },
    );
  }

  /// Full endpoint the native side POSTs each JPEG frame to.
  static String endpointFor(String apiBase) =>
      '${apiBase.replaceFirst(RegExp(r'/+$'), '')}/capture/frame';

  /// Ensure the runtime permissions needed for [source] are granted.
  /// Returns true if capture can proceed.
  Future<bool> ensurePermissions(CaptureSource source) async {
    if (!_supported) return false;
    // Notifications are needed for the foreground service on Android 13+.
    await Permission.notification.request();

    if (source == CaptureSource.camera) {
      final cam = await Permission.camera.request();
      if (!cam.isGranted) return false;
      return true;
    } else {
      // Screen capture consent is handled by the native MediaProjection dialog.
      final granted = await _method.invokeMethod<bool>(
        'requestScreenPermission',
      );
      return granted == true;
    }
  }

  /// Start capturing from [source] at [fps], POSTing frames to [apiBase].
  Future<void> start({
    required CaptureSource source,
    required int fps,
    required String apiBase,
    bool frontCamera = false,
    int resizeFactor = 1,
  }) async {
    if (!_supported) return;
    await _method.invokeMethod('start', {
      'source': source == CaptureSource.screen ? 'screen' : 'camera',
      'fps': fps.clamp(1, 60),
      'url': endpointFor(apiBase),
      'lens': frontCamera ? 'front' : 'back',
      'resizeFactor':
          const [1, 2, 3, 5].contains(resizeFactor) ? resizeFactor : 1,
    });
  }

  Future<void> setFps(int fps) async {
    if (!_supported) return;
    await _method.invokeMethod('setFps', {'fps': fps.clamp(1, 60)});
  }

  Future<void> stop() async {
    if (!_supported) return;
    await _method.invokeMethod('stop');
  }

  Future<bool> isRunning() async {
    if (!_supported) return false;
    return (await _method.invokeMethod<bool>('isRunning')) ?? false;
  }

  void dispose() {
    _eventSub?.cancel();
    _statusController.close();
  }
}

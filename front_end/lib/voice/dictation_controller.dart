import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';

/// Records a clip and turns it into text via the backend's `/transcribe`.
///
/// This is dictation, not conversation: the caller decides what the words
/// become. The Home page instead sends whole turns to `/chat/audio`, which
/// answers as it transcribes — a room composer needs the text back so the user
/// can edit it and choose between filing a note and asking the agent.
class DictationController {
  DictationController();

  final AudioRecorder _recorder = AudioRecorder();
  final BytesBuilder _buffer = BytesBuilder();

  bool _recording = false;
  bool get isRecording => _recording;

  Future<void> dispose() => _recorder.dispose();

  /// Begins buffering microphone audio. Returns false when the microphone is
  /// unavailable or permission was refused.
  Future<bool> start() async {
    if (_recording) return true;
    try {
      if (!await _recorder.hasPermission()) return false;
      _buffer.clear();
      final stream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          numChannels: 1,
          sampleRate: 16000,
        ),
      );
      stream.listen(_buffer.add, onError: (error, stack) {
        debugPrint('dictation stream error: $error');
      });
      _recording = true;
      return true;
    } catch (error) {
      debugPrint('dictation failed to start: $error');
      _recording = false;
      return false;
    }
  }

  /// Stops recording and resolves to the transcript.
  ///
  /// Throws with a readable message on failure so the caller can surface it —
  /// silence is worse than an error here, since the words are already spoken.
  Future<String> stopAndTranscribe(String apiBase) async {
    if (!_recording) return '';
    await _recorder.stop();
    _recording = false;
    final audio = _buffer.toBytes();
    _buffer.clear();

    // 16-bit mono at 16 kHz: under ~0.25 s there is nothing worth sending.
    if (audio.length < 8000) return '';
    if (apiBase.isEmpty) throw 'No home hub address configured';

    final response = await http
        .post(
          Uri.parse('$apiBase/transcribe'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({'data': audio}),
        )
        .timeout(const Duration(seconds: 45));
    final body = json.decode(response.body);
    if (response.statusCode != 200) {
      final detail = body is Map ? body['error'] : null;
      throw detail?.toString() ?? 'transcription failed (HTTP ${response.statusCode})';
    }
    return ((body as Map)['text'] ?? '').toString().trim();
  }

  /// Drops a recording without transcribing it.
  Future<void> cancel() async {
    if (!_recording) return;
    try {
      await _recorder.stop();
    } catch (error) {
      debugPrint('dictation failed to cancel cleanly: $error');
    }
    _recording = false;
    _buffer.clear();
  }
}

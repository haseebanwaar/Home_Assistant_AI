import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
// C:\Users\haseeb\AppData\Local\Android\Sdk\platform-tools/adb pair 192.168.1.17:38535
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:image_picker/image_picker.dart';
import 'package:audioplayers/audioplayers.dart';
import 'capture/frame_capture_controller.dart';
import 'memory/timeline_screen.dart';
import 'rooms/rooms_screen.dart';

void main() => runApp(const HomeMindApp());

class HomeMindApp extends StatelessWidget {
  const HomeMindApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF6EE7D8);
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'HomeMind',
      theme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: seed,
          brightness: Brightness.dark,
          surface: const Color(0xFF111827),
        ),
        scaffoldBackgroundColor: const Color(0xFF070B14),
        fontFamily: 'Segoe UI',
        useMaterial3: true,
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: const Color(0xFF111827),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: BorderSide.none,
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: Color(0xFF263246)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: seed, width: 1.5),
          ),
        ),
      ),
      home: const MyApp(),
    );
  }
}

enum MessageSender { user, assistant }

class ChatMessage {
  final MessageSender sender;
  String text;
  Uint8List? fullAudio; // Used to store the complete, replayable audio

  ChatMessage({
    required this.sender,
    required this.text,
    this.fullAudio,
  });
}

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  static const _ink = Color(0xFF070B14);
  static const _panel = Color(0xFF111827);
  static const _panelRaised = Color(0xFF182235);
  static const _line = Color(0xFF263246);
  static const _mint = Color(0xFF6EE7D8);
  static const _violet = Color(0xFF9B8AFB);
  static const _muted = Color(0xFF91A0B8);

  final TextEditingController _ipTextController = TextEditingController();

  Uint8List? _fileImage;
  final ImagePicker _picker = ImagePicker();
  final AudioPlayer _audioPlayer = AudioPlayer();
  final AudioRecorder _audioRecorder = AudioRecorder();
  final _audioBuffer = BytesBuilder();
  bool _isRecording = false;
  bool _isProcessing = false;
  final List<ChatMessage> _chatHistory = [];
  bool _isTalking = false;
  bool _isLive = false; // Add this line
  bool _useMemory = false; // Add this line
  bool _backendConnected = false;
  String _backendActivity = 'Connecting...';
  Map<String, dynamic> _backendStatus = const {};
  Timer? _statusTimer;

  // Proactive insights: id of the last one we've shown/played, plus a one-time
  // sync flag so we adopt the backend's latest id on connect without replaying
  // a backlog of stale insights.
  int _lastProactiveId = 0;
  bool _proactiveSynced = false;

  // Add these lines for the context selection
  final List<String> _contextOptions = ['talker', 'screen', 'camera'];
  final List<bool> _selectedContexts = [true, false, false]; // 'talker' is selected by default
  String _currentContext = 'talker';

  // For sequential audio playback
  final Queue<Uint8List> _audioQueue = Queue<Uint8List>();
  StreamSubscription? _playerStateSubscription;
  bool _isAudioPlaying = false;

  // --- Frame capture (camera / screen) source ---
  final FrameCaptureController _capture = FrameCaptureController();
  StreamSubscription<CaptureStatus>? _captureSub;
  CaptureStatus _captureStatus = CaptureStatus.idle;
  CaptureSource _captureSource = CaptureSource.camera;
  bool _frontCamera = false;
  final TextEditingController _fpsController = TextEditingController(text: '5');

  @override
  void initState() {
    super.initState();
    _ipTextController.text = '192.168.1.20'; // Set a default/last known IP
    _setupAudioPlayerListener();
    _startService();
    _captureSub = _capture.status.listen((s) {
      if (mounted) setState(() => _captureStatus = s);
    });
    _pollBackendStatus();
    _statusTimer = Timer.periodic(const Duration(seconds: 2), (_) {
      _pollBackendStatus();
      _fetchProactiveInsights();
    });
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    _audioPlayer.dispose();
    _playerStateSubscription?.cancel();
    _ipTextController.dispose();
    _captureSub?.cancel();
    _capture.dispose();
    _fpsController.dispose();
    _statusTimer?.cancel();
    super.dispose();
  }

  void _setupAudioPlayerListener() {
    _playerStateSubscription =
        _audioPlayer.onPlayerStateChanged.listen((state) {
      if (state == PlayerState.completed) {
        _isAudioPlaying = false;
        // When one audio chunk finishes, play the next one in the queue
        _playNextInQueue();
      }
    });
  }

  void _playNextInQueue() async {
    if (_audioQueue.isNotEmpty && !_isAudioPlaying) {
      _isAudioPlaying = true;
      // The play call is asynchronous, so we don't need to await it here.
      // The onPlayerStateChanged listener will handle the next steps.
      _audioPlayer.play(BytesSource(_audioQueue.removeFirst()));
    }
  }

  Future<void> _startService() async {
    try {
      if (await _audioRecorder.hasPermission()) {
        const encoder = AudioEncoder.pcm16bits;
        final isSupported = await _audioRecorder.isEncoderSupported(encoder);
        debugPrint('${encoder.name} supported: $isSupported');
        final config = RecordConfig(
          encoder: encoder,
          numChannels: 1,
          sampleRate: 16000,
        );
        debugPrint('$config');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error starting audio service: $e');
      }
    }
  }

  Future<void> _start() async {
    try {
      debugPrint('in start: ');
      final stream = await _audioRecorder.startStream(
        RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          numChannels: 1,
          sampleRate: 16000,
        ),
      );
      stream.listen(
        (data) {
          _audioBuffer.add(data);
        },
        onError: (o, s) {
          print('Error in audio stream: $o, stack: $s');
        },
      );
      setState(() {
        _isRecording = true;
      });
    } catch (e) {
      if (kDebugMode) {
        print('Error starting audio recording: $e');
      }
    }
  }

  Future<void> _stop() async {
    try {
      await _audioRecorder.stop();
      final audioData = _audioBuffer.toBytes();
      _audioBuffer.clear();

      setState(() {
        _isRecording = false;
        _isProcessing = true;
      });
      await _processAudio(audioData, true);
      setState(() {
        _isProcessing = false;
      });
    } catch (e) {
      if (kDebugMode) {
        print('Error stopping audio recording: $e');
        setState(() {
          _isRecording = false;
          _isProcessing = false;
        });
      }
    }
  }

  /// Merges multiple WAV file bytes into a single WAV file byte array.
  /// It assumes all WAV files have the same format (sample rate, channels, etc.).
  Uint8List _mergeWavBytes(List<Uint8List> wavChunks) {
    if (wavChunks.isEmpty) {
      return Uint8List(0);
    }
    if (wavChunks.length == 1) {
      return wavChunks.first;
    }

    // Use the header from the first chunk (typically 44 bytes for PCM)
    final header = wavChunks.first.sublist(0, 44);
    final mergedData = BytesBuilder();

    for (final chunk in wavChunks) {
      // Add the audio data part of each chunk, skipping the header
      if (chunk.length > 44) {
        mergedData.add(chunk.sublist(44));
      }
    }

    final fullAudioData = mergedData.toBytes();
    final headerView = ByteData.view(header.buffer);
    // Update RIFF chunk size (overall file size - 8)
    headerView.setUint32(4, 36 + fullAudioData.length, Endian.little);
    // Update data sub-chunk size (just the audio data size)
    headerView.setUint32(40, fullAudioData.length, Endian.little);

    return (BytesBuilder()..add(header)..add(fullAudioData)).toBytes();
  }

  Future<void> _processAudio(Uint8List buff, bool isEnd) async {
    try {
      if (mounted) setState(() => _backendActivity = 'Uploading audio');
      // Clear the audio queue for the new response
      // and stop any ongoing playback from a previous turn.
      _isAudioPlaying = false;
      await _audioPlayer.stop();
      _audioQueue.clear();

      final endPoint = 'http://${_ipTextController.text}:8000/chat/audio';
      final url = Uri.parse(endPoint);

      final Map<String, dynamic> requestBody = {
        'data': buff,
        'image': _fileImage != null ? base64.encode(_fileImage!) : null,
        'talking': _isTalking,
        'context': _currentContext, // Add the new context value
        'live': _isLive, // Add this line
        'memory': _useMemory, // Add this line
      };

      final request =
          http.Request('POST', url)
            ..headers['Content-Type'] = 'application/json'
            ..body = json.encode(requestBody);

      final streamedResponse = await request.send();

      if (streamedResponse.statusCode != 200) {
        print('Request failed with status: ${streamedResponse.statusCode}');
        final body = await streamedResponse.stream.bytesToString();
        print('Response body: $body');
        if (mounted) setState(() =>
            _backendActivity = 'Backend error (${streamedResponse.statusCode})');
        return;
      }

      final stream = streamedResponse.stream
          .transform(utf8.decoder)
          .transform(const LineSplitter());

      final List<Uint8List> assistantAudioChunks = [];
      ChatMessage? currentAssistantMessage;

      await for (final line in stream) {
        if (line.isEmpty) continue;

        try {
          final jsonResponse = json.decode(line);
          final type = jsonResponse['type'];

          if (type == 'query') {
            if (mounted) setState(() => _backendActivity = 'Generating response');
            final queryText = jsonResponse['text'];
            if (mounted) {
              setState(() {
                _chatHistory.add(ChatMessage(
                    sender: MessageSender.user, text: "User: $queryText"));
                // Add a placeholder for the assistant's response
                currentAssistantMessage = ChatMessage(
                    sender: MessageSender.assistant, text: "Assistant: ");
                _chatHistory.add(currentAssistantMessage!);
              });
            }
          } else if (type == 'vlm_text') {
            final vlmText = jsonResponse['text'];
            if (mounted && currentAssistantMessage != null) {
              setState(() {
                // Append streaming text to the last assistant message
                currentAssistantMessage!.text += vlmText;
              });
            }
          } else if (type == 'audio') {
            if (mounted) setState(() => _backendActivity = 'Streaming speech');
            final audioData = base64.decode(jsonResponse['data']);
            assistantAudioChunks.add(audioData);
            _audioQueue.add(audioData);
            // If the player is not already playing, start the queue.
            if (!_isAudioPlaying) {
              _playNextInQueue();
            }
          } else if (type == 'debug') {
            if (mounted) setState(() =>
                _backendActivity = 'Backend: ${jsonResponse['stage']}');
          } else if (type == 'error') {
            if (mounted) setState(() =>
                _backendActivity = 'Error: ${jsonResponse['message']}');
          } else if (type == 'done') {
            if (mounted) setState(() =>
                _backendActivity = 'Ready (${jsonResponse['total_ms']} ms)');
          }
        } catch (e) {
          print("Error processing stream line: $e. Line: '$line'");
        }
      }
      // Once the stream is finished, save the complete audio to the message
      if (currentAssistantMessage != null) {
        currentAssistantMessage!.fullAudio = _mergeWavBytes(assistantAudioChunks);
      }
    } catch (error) {
      if (kDebugMode) {
        print('Error processing audio: $error');
      }
      if (mounted) setState(() => _backendActivity = 'Connection failed');
    }
    }

  String get _apiBase => 'http://${_ipTextController.text.trim()}:8000';

  Future<void> _pollBackendStatus() async {
    if (_ipTextController.text.trim().isEmpty) return;
    try {
      final responses = await Future.wait([
        http.get(Uri.parse('$_apiBase/status')),
        http.get(Uri.parse('$_apiBase/ready')),
      ]).timeout(const Duration(seconds: 3));
      if (responses.any((r) => r.statusCode != 200)) {
        throw Exception('backend health request failed');
      }
      final value = json.decode(responses[0].body) as Map<String, dynamic>;
      value['readiness'] = json.decode(responses[1].body);
      if (mounted) setState(() {
        _backendConnected = true;
        _backendStatus = value;
        final pipeline = value['pipeline'] as Map?;
        if (pipeline != null) {
          final stage = '${pipeline['stage'] ?? 'ready'}'.replaceAll('_', ' ');
          _backendActivity = pipeline['active'] == true ? 'Backend: $stage' : stage;
        }
      });
    } catch (_) {
      if (mounted) setState(() {
        _backendConnected = false;
        if (!_isProcessing) _backendActivity = 'Backend offline';
      });
      // Re-sync proactive ids on the next successful connect (the server may
      // have restarted and reset its counter).
      _proactiveSynced = false;
    }
  }

  Future<void> _clearChatHistory() async {
    try {
      final response = await http.post(Uri.parse('$_apiBase/history/clear'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) throw Exception(response.body);
      if (mounted) setState(() => _chatHistory.clear());
      _showSnack('Conversation history cleared');
      await _pollBackendStatus();
    } catch (e) {
      _showSnack('Could not clear conversation history: $e');
    }
  }

  Future<void> _clearMemory() async {
    try {
      final response = await http.post(Uri.parse('$_apiBase/memory/clear'))
          .timeout(const Duration(seconds: 15));
      if (response.statusCode != 200) throw Exception(response.body);
      final result = json.decode(response.body) as Map<String, dynamic>;
      if (result['cleared'] != true) throw Exception(result['error'] ?? 'unknown error');
      _showSnack('Long-term activity memory cleared');
    } catch (e) {
      _showSnack('Could not clear activity memory: $e');
    }
  }

  /// Poll for unprompted proactive insights and play their speech on THIS
  /// device. New insights also land in the chat list (replayable via 💡).
  Future<void> _fetchProactiveInsights() async {
    if (_ipTextController.text.trim().isEmpty) return;
    try {
      final response = await http
          .get(Uri.parse('$_apiBase/proactive?since=$_lastProactiveId'))
          .timeout(const Duration(seconds: 3));
      if (response.statusCode != 200) return;
      final data = json.decode(response.body) as Map<String, dynamic>;

      // First poll after (re)connect: adopt the latest id without replaying
      // insights that were generated before the app was listening.
      if (!_proactiveSynced) {
        _lastProactiveId = (data['latest_id'] as num?)?.toInt() ?? 0;
        _proactiveSynced = true;
        return;
      }

      final insights = (data['insights'] as List?) ?? const [];
      for (final item in insights) {
        final map = item as Map<String, dynamic>;
        final id = (map['id'] as num?)?.toInt() ?? _lastProactiveId;
        if (id <= _lastProactiveId) continue;
        _lastProactiveId = id;

        Uint8List? audio;
        final audioB64 = map['audio'];
        if (audioB64 is String && audioB64.isNotEmpty) {
          audio = base64.decode(audioB64);
        }

        if (mounted) {
          setState(() {
            _chatHistory.add(ChatMessage(
              sender: MessageSender.assistant,
              text: '💡 ${map['text'] ?? ''}',
              fullAudio: audio,
            ));
          });
        }

        // Play on this device by enqueueing into the shared audio queue.
        if (audio != null) {
          _audioQueue.add(audio);
          if (!_isAudioPlaying) _playNextInQueue();
        }
      }
    } catch (_) {
      // Transient/offline — connectivity is surfaced by the status poll.
    }
  }

  Future<void> _setBackendCapture(bool start) async {
    final response = await http.post(
      Uri.parse('$_apiBase/capture/control'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'action': start ? 'start' : 'stop',
                         'source': _captureSource.name}),
    ).timeout(const Duration(seconds: 5));
    if (response.statusCode != 200) {
      throw Exception('backend returned ${response.statusCode}: ${response.body}');
    }
    await _pollBackendStatus();
  }

  int get _fps {
    final v = int.tryParse(_fpsController.text.trim()) ?? 5;
    return v.clamp(1, 60);
  }

  Future<void> _startCapture() async {
    final ip = _ipTextController.text.trim();
    if (ip.isEmpty) {
      _showSnack('Enter the server IP first');
      return;
    }
    final ok = await _capture.ensurePermissions(_captureSource);
    if (!ok) {
      _showSnack('Permission denied for ${_captureSource.name} capture');
      return;
    }
    try {
      await _setBackendCapture(true);
      await _capture.start(
        source: _captureSource,
        fps: _fps,
        ip: ip,
        frontCamera: _frontCamera,
      );
      final index = _captureSource == CaptureSource.camera ? 2 : 1;
      if (mounted) setState(() {
        _currentContext = _contextOptions[index];
        for (var i = 0; i < _selectedContexts.length; i++) {
          _selectedContexts[i] = i == index;
        }
        _isLive = true;
        _backendActivity = 'Waiting for ${_captureSource.name} frames';
      });
    } catch (e) {
      try { await _setBackendCapture(false); } catch (_) {}
      _showSnack('Failed to start capture: $e');
    }
  }

  Future<void> _stopCapture() async {
    await _capture.stop();
    try {
      await _setBackendCapture(false);
      if (mounted) setState(() {
        _isLive = false;
        _backendActivity = 'Capture stopped';
      });
    } catch (e) {
      _showSnack('Capture stopped locally, but backend stop failed: $e');
    }
  }

  Future<void> _applyFps() async {
    if (_captureStatus.running) {
      await _capture.setFps(_fps);
    }
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  Widget _buildBackendIndicators() {
    final mobile = (_backendStatus['mobile_capture'] as Map?) ?? const {};
    final active = mobile['active'] == true;
    final healthy = mobile['healthy'] == true;
    final frames = mobile['frames_received'] ?? 0;
    final asr = (_backendStatus['asr'] as Map?) ?? const {};
    final asrReady = asr['ready'] == true;
    Widget indicator(Color color, String text) => Padding(
      padding: const EdgeInsets.only(right: 12, bottom: 4),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(Icons.circle, size: 11, color: color),
        const SizedBox(width: 5),
        Text(text, style: const TextStyle(fontSize: 12)),
      ]),
    );
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: _panel.withOpacity(.7),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _line),
      ),
      child: Wrap(children: [
        indicator(_backendConnected ? Colors.green : Colors.red,
            _backendConnected ? 'Backend connected' : 'Backend offline'),
        indicator(asrReady ? Colors.green : Colors.red,
            asrReady ? 'Parakeet ready' : 'Parakeet unavailable'),
        indicator(active ? (healthy ? Colors.green : Colors.orange) : Colors.grey,
            active ? '${mobile['source']} active ($frames frames)' : 'Vision stopped'),
        indicator(_isProcessing ? Colors.blue : Colors.grey, _backendActivity),
      ]),
    );
  }

  Widget _buildCapturePanel() {
    final running = _captureStatus.running;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _panelRaised.withOpacity(.55),
        border: Border.all(color: _line),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text('Frame source: ', style: TextStyle(fontSize: 14)),
              ChoiceChip(
                label: const Text('Camera'),
                selected: _captureSource == CaptureSource.camera,
                onSelected: running
                    ? null
                    : (_) => setState(() => _captureSource = CaptureSource.camera),
              ),
              const SizedBox(width: 6),
              ChoiceChip(
                label: const Text('Screen'),
                selected: _captureSource == CaptureSource.screen,
                onSelected: running
                    ? null
                    : (_) => setState(() => _captureSource = CaptureSource.screen),
              ),
            ],
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(
                width: 70,
                child: TextField(
                  controller: _fpsController,
                  keyboardType: TextInputType.number,
                  textAlign: TextAlign.center,
                  decoration: const InputDecoration(labelText: 'FPS'),
                  onSubmitted: (_) => _applyFps(),
                ),
              ),
              const SizedBox(width: 8),
              if (running)
                _buttonsProcessing('Set FPS', 60, _applyFps),
              if (_captureSource == CaptureSource.camera && !running) ...[
                const SizedBox(width: 12),
                const Text('Front', style: TextStyle(fontSize: 14)),
                Switch(
                  value: _frontCamera,
                  onChanged: (v) => setState(() => _frontCamera = v),
                ),
              ],
            ],
          ),
          const SizedBox(height: 4),
          GestureDetector(
            onTap: running ? _stopCapture : _startCapture,
            child: Container(
              height: 42,
              width: 200,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: running ? const Color(0xFFFF607C) : _violet,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                running ? 'Stop Capture' : 'Start Capture',
                style: const TextStyle(color: Colors.white, fontSize: 15),
              ),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            running
                ? 'Streaming ${_captureStatus.source} @ ${_captureStatus.fps} fps · ${_captureStatus.frames} frames'
                : 'Idle (runs in background when minimized)',
            style: const TextStyle(fontSize: 11, color: _muted),
          ),
          if (_captureStatus.error != null)
            Padding(
              padding: const EdgeInsets.only(top: 2.0),
              child: Text(
                _captureStatus.error!,
                style: const TextStyle(fontSize: 11, color: Colors.red),
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _pickImageFile() async {
    final XFile? result = await _picker.pickImage(source: ImageSource.gallery);
    if (result != null) {
      _fileImage = await result.readAsBytes();
      setState(() => _isLive = false);
    }
  }

  Future<void> _unpickImageFile() async {
    _fileImage = null;
    setState(() {});
  }

  Widget _buttonsProcessing(String txt, double w, VoidCallback tap) {
    return OutlinedButton(
      onPressed: tap,
      style: OutlinedButton.styleFrom(
        minimumSize: Size(w, 44),
        foregroundColor: _mint,
        side: const BorderSide(color: _line),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      ),
      child: Text(txt, textAlign: TextAlign.center),
    );
  }

  Widget _bodyTextarea(Size s) {
    return SizedBox(
      width: 220,
      child: TextField(
        controller: _ipTextController,
        maxLines: 1,
        style: const TextStyle(fontSize: 13),
        decoration: InputDecoration(
          labelText: 'Home hub',
          hintText: '192.168.1.20',
          prefixIcon: const Icon(Icons.router_outlined, size: 19),
          suffixIcon: IconButton(
            tooltip: 'Reconnect',
            onPressed: _pollBackendStatus,
            icon: const Icon(Icons.refresh_rounded, size: 19),
          ),
        ),
        onSubmitted: (_) => _pollBackendStatus(),
      ),
    );
  }

  Widget _buildTapToSpeakButton() {
    Color color =
        _isProcessing
            ? _muted
            : (_isRecording ? const Color(0xFFFF607C) : _mint);

    return GestureDetector(
      onTapDown: _isProcessing ? null : (_) => _start(),
      onTapUp: _isProcessing ? null : (_) => _stop(),
      onTapCancel: _isProcessing ? null : () => _stop(),
      child: Container(
        width: 72,
        height: 72,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: LinearGradient(
            colors: _isRecording
                ? [const Color(0xFFFF607C), const Color(0xFFFF8A68)]
                : [color, const Color(0xFF42C7D6)],
          ),
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(.32),
              blurRadius: 28,
              spreadRadius: _isRecording ? 5 : 1,
            ),
          ],
        ),
        child: Icon(
          _isRecording ? Icons.stop : Icons.mic,
          color: _ink,
          size: 31,
        ),
      ),
    );
  }

  Widget _buildTranscriptionList() {
    return Expanded(
      child: ListView.builder(
        padding: const EdgeInsets.fromLTRB(18, 8, 18, 18),
        itemCount: _chatHistory.length,
        itemBuilder: (context, index) {
          final message = _chatHistory[index];
          return MessageBubble(message: message, audioPlayer: _audioPlayer);
        },
      ),
    );
  }

  Widget _buildHeader(Size size) {
    final brand = Row(
      children: [
        Container(
          width: 46,
          height: 46,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [_mint, _violet],
            ),
            borderRadius: BorderRadius.circular(15),
          ),
          child: const Icon(Icons.auto_awesome, color: _ink, size: 23),
        ),
        const SizedBox(width: 13),
        const Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('HomeMind',
                  style: TextStyle(
                      fontSize: 21,
                      fontWeight: FontWeight.w800,
                      letterSpacing: -.5)),
              Text('AMBIENT HOME INTELLIGENCE',
                  style: TextStyle(
                      color: _muted,
                      fontSize: 9,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.7)),
            ],
          ),
        ),
        IconButton(
          tooltip: 'Rooms',
          icon: const Icon(Icons.dashboard_customize, color: _mint),
          onPressed: () => Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => RoomsListScreen(apiBase: _apiBase),
            ),
          ),
        ),
        IconButton(
          tooltip: 'Memory timeline',
          icon: const Icon(Icons.timeline, color: _mint),
          onPressed: () => Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => MemoryTimelineScreen(apiBase: _apiBase),
            ),
          ),
        ),
      ],
    );
    if (size.width < 600) {
      return Column(
        children: [
          brand,
          const SizedBox(height: 12),
          SizedBox(width: double.infinity, child: _bodyTextarea(size)),
        ],
      );
    }
    return Row(
      children: [
        Expanded(child: brand),
        const SizedBox(width: 20),
        _bodyTextarea(size),
      ],
    );
  }

  void _showConnectionSheet() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: _panel,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      builder: (sheetContext) => Padding(
        padding: EdgeInsets.fromLTRB(
          22,
          12,
          22,
          22 + MediaQuery.viewInsetsOf(sheetContext).bottom,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 38,
                height: 4,
                decoration: BoxDecoration(
                  color: _line,
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
            ),
            const SizedBox(height: 22),
            const Text('Home hub',
                style: TextStyle(fontSize: 19, fontWeight: FontWeight.w700)),
            const SizedBox(height: 5),
            const Text('Connect this device to your local assistant.',
                style: TextStyle(color: _muted, fontSize: 12)),
            const SizedBox(height: 18),
            SizedBox(width: double.infinity, child: _bodyTextarea(MediaQuery.sizeOf(context))),
            const SizedBox(height: 14),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: () {
                  Navigator.pop(sheetContext);
                  _pollBackendStatus();
                },
                style: FilledButton.styleFrom(
                  backgroundColor: _mint,
                  foregroundColor: _ink,
                  padding: const EdgeInsets.symmetric(vertical: 15),
                ),
                child: const Text('Connect'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMobileHeader() {
    return Row(
      children: [
        Container(
          width: 38,
          height: 38,
          decoration: BoxDecoration(
            color: _mint,
            borderRadius: BorderRadius.circular(12),
          ),
          child: const Icon(Icons.auto_awesome, color: _ink, size: 19),
        ),
        const SizedBox(width: 11),
        const Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('HomeMind',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
              Text('Your home, in sync',
                  style: TextStyle(color: _muted, fontSize: 10)),
            ],
          ),
        ),
        IconButton(
          tooltip: 'Home hub settings',
          onPressed: _showConnectionSheet,
          style: IconButton.styleFrom(backgroundColor: _panel),
          icon: const Icon(Icons.tune_rounded, size: 20),
        ),
      ],
    );
  }

  Widget _buildMobileStatus() {
    final ready = _backendConnected;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        color: _panel,
        borderRadius: BorderRadius.circular(13),
        border: Border.all(color: _line),
      ),
      child: Row(
        children: [
          Icon(Icons.circle,
              size: 8, color: ready ? _mint : const Color(0xFFFF718B)),
          const SizedBox(width: 8),
          Text(ready ? 'Home hub online' : 'Home hub offline',
              style:
                  const TextStyle(fontSize: 11, fontWeight: FontWeight.w600)),
          const Spacer(),
          Flexible(
            child: Text(
              _backendActivity,
              overflow: TextOverflow.ellipsis,
              textAlign: TextAlign.right,
              style: const TextStyle(color: _muted, fontSize: 10),
            ),
          ),
        ],
      ),
    );
  }

  void _showCaptureSheet() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: _panel,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      builder: (sheetContext) => SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(16, 10, 16, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 38,
                height: 4,
                decoration: BoxDecoration(
                  color: _line,
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              const SizedBox(height: 16),
              _buildCapturePanel(),
              const SizedBox(height: 8),
              SizedBox(
                width: double.infinity,
                child: TextButton.icon(
                  onPressed: _clearMemory,
                  icon: const Icon(Icons.layers_clear_outlined, size: 17),
                  label: const Text('Clear long-term memory'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildConversationCard() {
    return Container(
      decoration: BoxDecoration(
        color: _panel,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: _line),
        boxShadow: const [
          BoxShadow(color: Color(0x33000000), blurRadius: 30, offset: Offset(0, 14)),
        ],
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(18, 16, 12, 12),
            child: Row(
              children: [
                const Icon(Icons.forum_outlined, color: _mint, size: 19),
                const SizedBox(width: 9),
                const Text('Conversation',
                    style:
                        TextStyle(fontSize: 14, fontWeight: FontWeight.w700)),
                const Spacer(),
                IconButton(
                  tooltip: 'Clear conversation',
                  onPressed: _clearChatHistory,
                  icon: const Icon(Icons.delete_sweep_outlined,
                      color: _muted, size: 20),
                ),
              ],
            ),
          ),
          const Divider(height: 1, color: _line),
          if (_chatHistory.isEmpty)
            const Expanded(
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.waves_rounded, color: _mint, size: 42),
                    SizedBox(height: 14),
                    Text('Your home is listening',
                        style: TextStyle(
                            fontSize: 17, fontWeight: FontWeight.w700)),
                    SizedBox(height: 6),
                    Text('Hold the microphone and start a conversation',
                        style: TextStyle(color: _muted, fontSize: 12)),
                  ],
                ),
              ),
            )
          else
            _buildTranscriptionList(),
          if (_fileImage != null)
            Padding(
              padding: const EdgeInsets.all(12),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.memory(_fileImage!, height: 100),
              ),
            ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
            decoration: const BoxDecoration(
              color: _panelRaised,
              borderRadius: BorderRadius.vertical(bottom: Radius.circular(24)),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(_isRecording ? 'Listening…' : 'Hold to speak',
                          style: const TextStyle(
                              fontSize: 13, fontWeight: FontWeight.w700)),
                      Text(
                          _isProcessing
                              ? 'HomeMind is thinking'
                              : 'Release when you are finished',
                          style:
                              const TextStyle(color: _muted, fontSize: 10)),
                    ],
                  ),
                ),
                _buildTapToSpeakButton(),
                const Spacer(),
                IconButton.filledTonal(
                  tooltip: 'Attach image',
                  onPressed: _pickImageFile,
                  icon: const Icon(Icons.add_photo_alternate_outlined),
                ),
                if (_fileImage != null)
                  IconButton(
                    tooltip: 'Remove image',
                    onPressed: _unpickImageFile,
                    icon: const Icon(Icons.close, color: _muted),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildContextSelector() {
    const icons = [Icons.mic_none, Icons.monitor_outlined, Icons.camera_alt_outlined];
    return Container(
      padding: const EdgeInsets.all(5),
      decoration: BoxDecoration(
          color: _ink, borderRadius: BorderRadius.circular(15)),
      child: Row(
        children: List.generate(_contextOptions.length, (index) {
          final selected = _selectedContexts[index];
          return Expanded(
            child: InkWell(
              borderRadius: BorderRadius.circular(11),
              onTap: () {
                setState(() {
                  for (var i = 0; i < _selectedContexts.length; i++) {
                    _selectedContexts[i] = i == index;
                  }
                  _currentContext = _contextOptions[index];
                  _isLive = index != 0;
                });
              },
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                padding: const EdgeInsets.symmetric(vertical: 10),
                decoration: BoxDecoration(
                  color: selected ? _panelRaised : Colors.transparent,
                  borderRadius: BorderRadius.circular(11),
                ),
                child: Column(
                  children: [
                    Icon(icons[index],
                        size: 18, color: selected ? _mint : _muted),
                    const SizedBox(height: 4),
                    Text(_contextOptions[index].toUpperCase(),
                        style: TextStyle(
                            color: selected ? Colors.white : _muted,
                            fontSize: 9,
                            fontWeight: FontWeight.w700,
                            letterSpacing: .6)),
                  ],
                ),
              ),
            ),
          );
        }),
      ),
    );
  }

  Widget _buildControlPanel() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: _panel,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: _line),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('PERCEPTION MODE',
              style: TextStyle(
                  color: _muted,
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.4)),
          const SizedBox(height: 10),
          _buildContextSelector(),
          const SizedBox(height: 16),
          _buildCapturePanel(),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _buildToggleSwitch('Conversation', _isTalking,
                  (v) => setState(() => _isTalking = v)),
              _buildToggleSwitch('Live', _isLive, (v) {
                if (v && _currentContext == 'talker') {
                  _showSnack('Choose Screen or Camera before enabling Live');
                  return;
                }
                setState(() => _isLive = v);
              }),
              _buildToggleSwitch('Memory', _useMemory,
                  (v) => setState(() => _useMemory = v)),
            ],
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: _clearMemory,
              icon: const Icon(Icons.layers_clear_outlined, size: 18),
              label: const Text('Clear long-term memory'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMobileControls() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: _panel,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _line),
      ),
      child: Column(
        children: [
          _buildContextSelector(),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _buildToggleSwitch('Talk', _isTalking,
                    (v) => setState(() => _isTalking = v)),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: _buildToggleSwitch('Live', _isLive, (v) {
                  if (v && _currentContext == 'talker') {
                    _showSnack('Choose Screen or Camera first');
                    return;
                  }
                  setState(() => _isLive = v);
                }),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: _buildToggleSwitch('Memory', _useMemory,
                    (v) => setState(() => _useMemory = v)),
              ),
            ],
          ),
          ListTile(
            dense: true,
            contentPadding: const EdgeInsets.fromLTRB(4, 4, 4, 0),
            onTap: _showCaptureSheet,
            leading:
                const Icon(Icons.center_focus_strong, color: _muted, size: 19),
            title: const Text('Capture & privacy',
                style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
            subtitle: const Text('Camera, screen and stored memory',
                style: TextStyle(color: _muted, fontSize: 10)),
            trailing:
                const Icon(Icons.chevron_right_rounded, color: _muted, size: 20),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final Size s = MediaQuery.of(context).size;
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: RadialGradient(
            center: Alignment(-.8, -1),
            radius: 1.2,
            colors: [Color(0xFF132235), _ink],
          ),
        ),
        child: SafeArea(
          child: LayoutBuilder(builder: (context, constraints) {
            final desktop = constraints.maxWidth >= 900;
            final desktopContent = Column(
              children: [
                _buildHeader(s),
                const SizedBox(height: 14),
                Align(
                    alignment: Alignment.centerLeft,
                    child: _buildBackendIndicators()),
                const SizedBox(height: 14),
                if (desktop)
                  Expanded(
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Expanded(child: _buildConversationCard()),
                        const SizedBox(width: 14),
                        SizedBox(
                          width: 370,
                          child: SingleChildScrollView(
                              child: _buildControlPanel()),
                        ),
                      ],
                    ),
                  )
              ],
            );
            return Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1240),
                child: Padding(
                  padding: EdgeInsets.fromLTRB(
                      desktop ? 16 : 14, 12, desktop ? 16 : 14, 12),
                  child: desktop
                      ? desktopContent
                      : Column(
                          children: [
                            _buildMobileHeader(),
                            const SizedBox(height: 10),
                            _buildMobileStatus(),
                            const SizedBox(height: 10),
                            Expanded(child: _buildConversationCard()),
                            const SizedBox(height: 10),
                            _buildMobileControls(),
                          ],
                        ),
                ),
              ),
            );
          }),
        ),
      ),
    );
  }

  // Helper method to reduce code duplication for switches
  Widget _buildToggleSwitch(
      String title, bool value, ValueChanged<bool> onChanged) {
    return FilterChip(
      selected: value,
      onSelected: onChanged,
      showCheckmark: false,
      avatar: Icon(
        value ? Icons.check_circle : Icons.circle_outlined,
        size: 16,
        color: value ? _mint : _muted,
      ),
      label: Text(title),
      side: const BorderSide(color: _line),
      selectedColor: _mint.withOpacity(.12),
      backgroundColor: _panelRaised,
    );
  }
}

class MessageBubble extends StatelessWidget {
  final ChatMessage message;
  final AudioPlayer audioPlayer;

  const MessageBubble(
      {Key? key, required this.message, required this.audioPlayer})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    final isUserMessage = message.sender == MessageSender.user;
    final alignment =
        isUserMessage ? Alignment.centerRight : Alignment.centerLeft;
    const mint = Color(0xFF6EE7D8);
    const violet = Color(0xFF9B8AFB);
    final color =
        isUserMessage ? violet.withOpacity(.18) : const Color(0xFF182235);
    final accent = isUserMessage ? violet : mint;

    return Align(
      alignment: alignment,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 560),
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.fromLTRB(14, 11, 10, 11),
        decoration: BoxDecoration(
          color: color,
          border: Border.all(color: accent.withOpacity(.22)),
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(18),
            topRight: const Radius.circular(18),
            bottomLeft: Radius.circular(isUserMessage ? 18 : 4),
            bottomRight: Radius.circular(isUserMessage ? 4 : 18),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Flexible(
              child: Text(
                message.text,
                style: const TextStyle(
                    fontSize: 14, color: Color(0xFFE9EEF7), height: 1.45),
              ),
            ),
            if (!isUserMessage && message.fullAudio != null && message.fullAudio!.isNotEmpty)
              _buildReplayButton(),
          ],
        ),
      ),
    );
  }

  Widget _buildReplayButton() {
    return IconButton(
      icon: const Icon(Icons.replay_rounded, color: Color(0xFF6EE7D8)),
      onPressed: () {
        if (message.fullAudio != null) {
          audioPlayer.play(BytesSource(message.fullAudio!));
        }
      },
    );
  }
}

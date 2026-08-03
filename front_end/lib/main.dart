import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
// C:\Users\haseeb\AppData\Local\Android\Sdk\platform-tools/adb pair 192.168.1.17:38535
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
// Only the initialiser is needed here; the full export collides with
// audioplayers' PlayerState.
import 'package:media_kit/media_kit.dart' show MediaKit;
import 'package:record/record.dart';
import 'package:image_picker/image_picker.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'capture/frame_capture_controller.dart';
import 'memory/timeline_screen.dart';
import 'rooms/rooms_screen.dart';
import 'assistant/assistant_screen.dart';
import 'notifications/desktop_alert.dart';
import 'notifications/local_notification_controller.dart';
import 'notifications/notifications_screen.dart';
import 'clips/clip_viewer.dart';
import 'network/http_json.dart';
import 'settings/global_hotkey_service.dart';
import 'settings/settings_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Must run before any Player is constructed, so clip playback works on the
  // desktop build.
  MediaKit.ensureInitialized();
  await setupDesktopAlerts();
  runApp(const HomeMindApp());
}

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
        fontFamily: 'NotoSans',
        useMaterial3: true,
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: const Color(0xFF111827),
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 16,
            vertical: 14,
          ),
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
  // Footage an unprompted insight was made from, when the server kept one. The
  // remark is a claim about something the user was not watching, so the bubble
  // carries the way to check it.
  final String? clipId;
  final double? clipCoversSeconds;
  final double? clipPlaysSeconds;

  ChatMessage({
    required this.sender,
    required this.text,
    this.fullAudio,
    this.clipId,
    this.clipCoversSeconds,
    this.clipPlaysSeconds,
  });
}

class _ReflectionSourceOption {
  const _ReflectionSourceOption({
    required this.id,
    required this.label,
    required this.context,
    required this.available,
    required this.detail,
  });

  final String id;
  final String label;
  final String context;
  final bool available;
  final String detail;

  factory _ReflectionSourceOption.fromJson(Map<String, dynamic> data) {
    return _ReflectionSourceOption(
      id: '${data['id'] ?? ''}',
      label: '${data['label'] ?? data['id'] ?? 'Source'}',
      context: '${data['context'] ?? 'screen'}',
      available: data['available'] == true,
      detail: '${data['detail'] ?? ''}',
    );
  }
}

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  static const _homeHubPreferenceKey = 'home_hub_url';
  static const _reflectShortcutPreferenceKey = 'reflect_shortcut';
  static const _sourceReflectShortcutPreferenceKey = 'source_reflect_shortcut';
  static const _clipboardAnswerShortcutPreferenceKey =
      'clipboard_answer_shortcut';
  static const _clipboardAnswerPromptPreferenceKey = 'clipboard_answer_prompt';
  static const _promptShortcutPreferencePrefix = 'reflection_prompt_shortcut_';
  static const _promptTextPreferencePrefix = 'reflection_prompt_text_';
  static const _shortcutThinkingPreferencePrefix = 'shortcut_thinking_';
  static const _disabledShortcutValue = 'disabled';
  static const _defaultHomeHub = String.fromEnvironment(
    'HOME_HUB_URL',
    defaultValue: '192.168.1.37',
  );
  static const _ink = Color(0xFF070B14);
  static const _panel = Color(0xFF111827);
  static const _panelRaised = Color(0xFF182235);
  static const _line = Color(0xFF263246);
  static const _mint = Color(0xFF6EE7D8);
  static const _violet = Color(0xFF9B8AFB);
  static const _muted = Color(0xFF91A0B8);

  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  int _workspaceIndex = 0;
  final TextEditingController _ipTextController = TextEditingController();

  Uint8List? _fileImage;
  // The typed alternative to holding the mic. Same turn pipeline either way.
  final TextEditingController _composerController = TextEditingController();
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
  bool _conversationThinking = false;
  bool _backendConnected = false;
  String _backendActivity = 'Connecting...';
  Map<String, dynamic> _backendStatus = const {};
  Timer? _statusTimer;
  bool _statusPollInFlight = false;
  int _consecutiveStatusFailures = 0;

  // Proactive insights: id of the last one we've shown/played, plus a one-time
  // sync flag so we adopt the backend's latest id on connect without replaying
  // a backlog of stale insights.
  int _lastProactiveId = 0;
  bool _proactiveSynced = false;
  bool _proactiveEnabled = true;
  bool _proactiveVoiceEnabled = true;
  bool _proactiveFeedEnabled = true;
  bool _proactiveNotificationsEnabled = false;
  bool _eventNotificationsEnabled = true;
  bool _notificationsMuted = false;
  bool _deliveryPreferencesLoaded = false;
  String _kokoroVoice = 'bf_lily';
  List<KokoroVoiceOption> _kokoroVoices = const [];
  bool _kokoroVoiceLoading = false;
  bool _kokoroVoiceSaving = false;
  String? _kokoroVoiceError;
  List<CaptureSourceSetting> _captureSources = const [];
  bool _captureSettingsLoading = false;
  String? _captureSettingsSavingSource;
  String? _captureSettingsError;
  final LocalNotificationController _notificationController =
      LocalNotificationController();
  int _lastNotificationSequence = 0;
  int _unreadNotifications = 0;

  // On-demand reflection (Alt+Shift+W by default). Ten frames at the 1fps the
  // backend buffers is the last ten seconds of whatever is being captured.
  static const int _reflectFrames = 10;
  bool _reflecting = false;
  bool _choosingReflectionSource = false;
  AppShortcutBinding? _reflectShortcut = AppShortcutBinding.reflectionDefault;
  AppShortcutBinding? _sourceReflectShortcut =
      AppShortcutBinding.sourceReflectionDefault;
  AppShortcutBinding? _clipboardAnswerShortcut =
      AppShortcutBinding.clipboardAnswerDefault;
  String _clipboardAnswerPrompt = defaultClipboardAnswerPrompt;
  Map<String, AppShortcutBinding?> _promptShortcuts = {
    for (final preset in reflectionPromptPresets)
      preset.id: preset.defaultBinding,
  };
  // Only presets the user actually reworded appear here; everything else falls
  // back to the shipped prompt, so edits survive changes to the defaults.
  Map<String, String> _promptTexts = {};
  Map<String, bool> _shortcutThinking = {
    'reflect_now': true,
    'reflect_from_source': true,
    clipboardAnswerActionId: false,
    for (final preset in reflectionPromptPresets) preset.id: true,
  };
  final GlobalHotkeyService _globalHotkeys = createGlobalHotkeyService();
  String? _globalHotkeyError;

  // Add these lines for the context selection
  final List<String> _contextOptions = ['talker', 'screen', 'camera'];
  final List<bool> _selectedContexts = [
    true,
    false,
    false,
  ]; // 'talker' is selected by default
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
    _ipTextController.text = _defaultHomeHub;
    _setupAudioPlayerListener();
    _startService();
    _loadDeliveryPreferences();
    _loadShortcutPreferences();
    _loadPromptTexts();
    _loadClipboardAnswerPrompt();
    _captureSub = _capture.status.listen((s) {
      if (mounted) setState(() => _captureStatus = s);
    });
    _loadHomeHub();
    _statusTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      _pollBackendStatus();
      _loadCaptureSettings();
      _fetchProactiveInsights();
      _fetchNotifications();
    });
  }

  Future<void> _loadHomeHub() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_homeHubPreferenceKey)?.trim();
    if (!mounted) return;
    if (saved != null && saved.isNotEmpty) {
      _ipTextController.text = saved;
    }
    await _pollBackendStatus();
    await _loadTtsSettings();
    await _loadCaptureSettings();
  }

  Future<void> _connectToHomeHub() async {
    final value = _ipTextController.text.trim();
    if (value.isEmpty) {
      _showSnack('Enter the PC Wi-Fi address first');
      return;
    }
    if (mounted) {
      setState(() {
        _backendConnected = false;
        _backendActivity = 'Connecting...';
      });
    }
    _consecutiveStatusFailures = 0;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_homeHubPreferenceKey, value);
    await _pollBackendStatus();
    await _loadTtsSettings();
    await _loadCaptureSettings();
  }

  Future<void> _loadTtsSettings() async {
    final apiBase = _apiBase;
    if (apiBase.isEmpty || _kokoroVoiceLoading) return;
    _kokoroVoiceLoading = true;
    try {
      final response = await http
          .get(Uri.parse('$apiBase/settings/tts'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) {
        throw Exception('TTS settings request returned ${response.statusCode}');
      }
      final data = decodeJsonResponse(response) as Map<String, dynamic>;
      final voices = (data['voices'] as List? ?? const [])
          .whereType<Map>()
          .map(
            (voice) => KokoroVoiceOption(
              id: '${voice['id'] ?? ''}',
              label: '${voice['label'] ?? voice['id'] ?? ''}',
              name: '${voice['name'] ?? ''}',
              accent: '${voice['accent'] ?? ''}',
              gender: '${voice['gender'] ?? ''}',
            ),
          )
          .where((voice) => voice.id.isNotEmpty)
          .toList(growable: false);
      final selected = '${data['voice'] ?? ''}';
      if (!mounted || _apiBase != apiBase) return;
      setState(() {
        _kokoroVoices = voices;
        if (voices.any((voice) => voice.id == selected)) {
          _kokoroVoice = selected;
        }
        _kokoroVoiceError = null;
      });
    } catch (error) {
      if (!mounted || _apiBase != apiBase) return;
      setState(() {
        _kokoroVoices = const [];
        _kokoroVoiceError = 'Could not load Kokoro voices from the Home Hub.';
      });
    } finally {
      _kokoroVoiceLoading = false;
    }
  }

  Future<void> _updateKokoroVoice(String voice) async {
    if (_kokoroVoiceSaving || voice == _kokoroVoice) return;
    final apiBase = _apiBase;
    if (apiBase.isEmpty) {
      _showSnack('Set the home hub address first');
      return;
    }
    final previous = _kokoroVoice;
    setState(() {
      _kokoroVoice = voice;
      _kokoroVoiceSaving = true;
      _kokoroVoiceError = null;
    });
    try {
      final response = await http
          .put(
            Uri.parse('$apiBase/settings/tts'),
            headers: const {'Content-Type': 'application/json'},
            body: json.encode({'voice': voice}),
          )
          .timeout(const Duration(seconds: 5));
      final data = decodeJsonResponse(response) as Map<String, dynamic>;
      if (response.statusCode != 200) {
        throw Exception('${data['error'] ?? 'Unable to save speaker voice'}');
      }
      final saved = '${data['voice'] ?? voice}';
      if (!mounted) return;
      if (_apiBase != apiBase) {
        setState(() => _kokoroVoiceSaving = false);
        return;
      }
      setState(() {
        _kokoroVoice = saved;
        _kokoroVoiceSaving = false;
        _kokoroVoiceError = null;
      });
      _showSnack('Kokoro speaker changed');
    } catch (error) {
      if (!mounted) return;
      if (_apiBase != apiBase) {
        setState(() => _kokoroVoiceSaving = false);
        return;
      }
      setState(() {
        _kokoroVoice = previous;
        _kokoroVoiceSaving = false;
        _kokoroVoiceError = 'Could not save the Kokoro speaker.';
      });
      _showSnack('Could not change the Kokoro speaker');
    }
  }

  List<CaptureSourceSetting> _decodeCaptureSources(Map<String, dynamic> data) {
    return (data['sources'] as List? ?? const [])
        .whereType<Map>()
        .map(
          (source) => CaptureSourceSetting(
            id: '${source['id'] ?? ''}',
            label: '${source['label'] ?? source['id'] ?? ''}',
            kind: '${source['kind'] ?? ''}',
            sampleFps: (source['sample_fps'] as num?)?.toDouble() ?? 1.0,
            inferenceIntervalSeconds:
                (source['inference_interval_seconds'] as num?)?.toInt() ?? 60,
            expectedFrames: (source['expected_frames'] as num?)?.toInt() ?? 0,
            bufferedFrames: (source['buffered_frames'] as num?)?.toInt() ?? 0,
            available: source['available'] == true,
            thinking: source['thinking'] == true,
          ),
        )
        .where((source) => source.id.isNotEmpty)
        .toList(growable: false);
  }

  Future<void> _loadCaptureSettings() async {
    final apiBase = _apiBase;
    if (apiBase.isEmpty || _captureSettingsSavingSource != null) return;
    if (_captureSources.isEmpty && mounted) {
      setState(() => _captureSettingsLoading = true);
    }
    try {
      final response = await http
          .get(Uri.parse('$apiBase/settings/capture'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) {
        throw Exception(
          'Capture settings request returned ${response.statusCode}',
        );
      }
      final data = decodeJsonResponse(response) as Map<String, dynamic>;
      if (!mounted || _apiBase != apiBase) return;
      setState(() {
        _captureSources = _decodeCaptureSources(data);
        _captureSettingsLoading = false;
        _captureSettingsError = null;
      });
    } catch (_) {
      if (!mounted || _apiBase != apiBase) return;
      setState(() {
        _captureSettingsLoading = false;
        _captureSettingsError =
            'Could not load automatic capture settings from the Home Hub.';
      });
    }
  }

  Future<void> _updateCaptureSource(
    String sourceId,
    double sampleFps,
    int inferenceIntervalSeconds,
    bool thinking,
  ) async {
    final apiBase = _apiBase;
    if (apiBase.isEmpty || _captureSettingsSavingSource != null) return;
    setState(() {
      _captureSettingsSavingSource = sourceId;
      _captureSettingsError = null;
    });
    try {
      final response = await http
          .put(
            Uri.parse('$apiBase/settings/capture'),
            headers: const {'Content-Type': 'application/json'},
            body: json.encode({
              'source_id': sourceId,
              'sample_fps': sampleFps,
              'inference_interval_seconds': inferenceIntervalSeconds,
              'thinking': thinking,
            }),
          )
          .timeout(const Duration(seconds: 8));
      final data = decodeJsonResponse(response) as Map<String, dynamic>;
      if (response.statusCode != 200) {
        throw Exception('${data['error'] ?? 'Unable to save capture profile'}');
      }
      if (!mounted || _apiBase != apiBase) return;
      setState(() {
        _captureSources = _decodeCaptureSources(data);
        _captureSettingsSavingSource = null;
        _captureSettingsError = null;
      });
      _showSnack('Capture profile saved; current frame window was reset');
    } catch (error) {
      if (!mounted || _apiBase != apiBase) return;
      setState(() {
        _captureSettingsSavingSource = null;
        _captureSettingsError = 'Could not save the capture profile.';
      });
      _showSnack('Could not save capture profile');
    }
  }

  Future<void> _loadDeliveryPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _proactiveEnabled = prefs.getBool('proactive_enabled') ?? true;
      _proactiveVoiceEnabled = prefs.getBool('proactive_voice_enabled') ?? true;
      _proactiveFeedEnabled = prefs.getBool('proactive_feed_enabled') ?? true;
      _proactiveNotificationsEnabled =
          prefs.getBool('proactive_notifications_enabled') ?? false;
      _eventNotificationsEnabled =
          prefs.getBool('event_notifications_enabled') ?? true;
      _notificationsMuted = prefs.getBool('notifications_muted') ?? false;
      _deliveryPreferencesLoaded = true;
    });
    await _syncNotificationMonitoring();
  }

  Future<void> _loadShortcutPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    final shortcut = _shortcutFromPreference(
      prefs.getString(_reflectShortcutPreferenceKey),
      AppShortcutBinding.reflectionDefault,
    );
    final usedBindings = <AppShortcutBinding>{if (shortcut != null) shortcut};
    var sourceShortcut = _shortcutFromPreference(
      prefs.getString(_sourceReflectShortcutPreferenceKey),
      AppShortcutBinding.sourceReflectionDefault,
    );
    if (sourceShortcut != null && usedBindings.contains(sourceShortcut)) {
      sourceShortcut =
          usedBindings.contains(AppShortcutBinding.sourceReflectionDefault)
              ? null
              : AppShortcutBinding.sourceReflectionDefault;
    }
    if (sourceShortcut != null) usedBindings.add(sourceShortcut);

    final promptShortcuts = <String, AppShortcutBinding?>{};
    for (final preset in reflectionPromptPresets) {
      var binding = _shortcutFromPreference(
        prefs.getString(_promptShortcutPreferenceKey(preset.id)),
        preset.defaultBinding,
      );
      if (binding != null && usedBindings.contains(binding)) {
        binding =
            usedBindings.contains(preset.defaultBinding)
                ? null
                : preset.defaultBinding;
      }
      promptShortcuts[preset.id] = binding;
      if (binding != null) usedBindings.add(binding);
    }

    // Existing user-configured prompt shortcuts win if one already uses the
    // new action's default. In that rare case the clipboard action starts
    // disabled instead of silently taking another action's binding.
    var clipboardShortcut = _shortcutFromPreference(
      prefs.getString(_clipboardAnswerShortcutPreferenceKey),
      AppShortcutBinding.clipboardAnswerDefault,
    );
    if (clipboardShortcut != null && usedBindings.contains(clipboardShortcut)) {
      clipboardShortcut = null;
    }
    if (!mounted) return;
    final shortcutThinking = <String, bool>{
      for (final entry in _shortcutThinking.entries)
        entry.key:
            prefs.getBool('$_shortcutThinkingPreferencePrefix${entry.key}') ??
            entry.value,
    };
    setState(() {
      _reflectShortcut = shortcut;
      _sourceReflectShortcut = sourceShortcut;
      _clipboardAnswerShortcut = clipboardShortcut;
      _promptShortcuts = promptShortcuts;
      _shortcutThinking = shortcutThinking;
    });
    await _syncGlobalHotkeys();
  }

  String _promptShortcutPreferenceKey(String presetId) =>
      '$_promptShortcutPreferencePrefix$presetId';

  String _promptTextPreferenceKey(String presetId) =>
      '$_promptTextPreferencePrefix$presetId';

  bool _thinkingForShortcut(String actionId) =>
      _shortcutThinking[actionId] ?? false;

  void _updateShortcutThinking(String actionId, bool enabled) {
    setState(() {
      _shortcutThinking = Map<String, bool>.from(_shortcutThinking)
        ..[actionId] = enabled;
    });
    unawaited(_persistShortcutThinking(actionId, enabled));
  }

  Future<void> _persistShortcutThinking(String actionId, bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('$_shortcutThinkingPreferencePrefix$actionId', enabled);
  }

  Future<void> _loadPromptTexts() async {
    final prefs = await SharedPreferences.getInstance();
    final texts = <String, String>{};
    for (final preset in reflectionPromptPresets) {
      final saved =
          prefs.getString(_promptTextPreferenceKey(preset.id))?.trim();
      if (saved != null && saved.isNotEmpty && saved != preset.prompt) {
        texts[preset.id] = saved;
      }
    }
    if (!mounted) return;
    setState(() => _promptTexts = texts);
  }

  Future<void> _loadClipboardAnswerPrompt() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_clipboardAnswerPromptPreferenceKey)?.trim();
    if (!mounted) return;
    setState(() {
      _clipboardAnswerPrompt =
          saved == null || saved.isEmpty ? defaultClipboardAnswerPrompt : saved;
    });
  }

  /// The wording a preset sends: the user's edit when there is one, else the
  /// prompt it shipped with.
  String _promptTextFor(ReflectionPromptPreset preset) {
    final custom = _promptTexts[preset.id]?.trim();
    return (custom == null || custom.isEmpty) ? preset.prompt : custom;
  }

  void _updatePromptText(String presetId, String? prompt) {
    final trimmed = prompt?.trim();
    setState(() {
      final next = Map<String, String>.from(_promptTexts);
      if (trimmed == null || trimmed.isEmpty) {
        next.remove(presetId);
      } else {
        next[presetId] = trimmed;
      }
      _promptTexts = next;
    });
    unawaited(_persistPromptText(presetId, trimmed));
  }

  Future<void> _persistPromptText(String presetId, String? prompt) async {
    final prefs = await SharedPreferences.getInstance();
    final key = _promptTextPreferenceKey(presetId);
    // Removing the key rather than storing the default keeps a preset tracking
    // the shipped wording if that wording later changes.
    if (prompt == null || prompt.isEmpty) {
      await prefs.remove(key);
    } else {
      await prefs.setString(key, prompt);
    }
  }

  AppShortcutBinding? _shortcutFromPreference(
    String? saved,
    AppShortcutBinding fallback,
  ) {
    if (saved == _disabledShortcutValue) return null;
    return AppShortcutBinding.tryDecode(saved) ?? fallback;
  }

  void _updateReflectShortcut(AppShortcutBinding? shortcut) {
    setState(() => _reflectShortcut = shortcut);
    unawaited(
      _persistShortcutPreference(_reflectShortcutPreferenceKey, shortcut),
    );
  }

  void _updateSourceReflectShortcut(AppShortcutBinding? shortcut) {
    setState(() => _sourceReflectShortcut = shortcut);
    unawaited(
      _persistShortcutPreference(_sourceReflectShortcutPreferenceKey, shortcut),
    );
  }

  void _updateClipboardAnswerShortcut(AppShortcutBinding? shortcut) {
    setState(() => _clipboardAnswerShortcut = shortcut);
    unawaited(
      _persistShortcutPreference(
        _clipboardAnswerShortcutPreferenceKey,
        shortcut,
      ),
    );
  }

  void _updateClipboardAnswerPrompt(String? prompt) {
    final trimmed = prompt?.trim();
    setState(() {
      _clipboardAnswerPrompt =
          trimmed == null || trimmed.isEmpty
              ? defaultClipboardAnswerPrompt
              : trimmed;
    });
    unawaited(_persistClipboardAnswerPrompt(trimmed));
  }

  Future<void> _persistClipboardAnswerPrompt(String? prompt) async {
    final prefs = await SharedPreferences.getInstance();
    if (prompt == null ||
        prompt.isEmpty ||
        prompt == defaultClipboardAnswerPrompt) {
      await prefs.remove(_clipboardAnswerPromptPreferenceKey);
    } else {
      await prefs.setString(_clipboardAnswerPromptPreferenceKey, prompt);
    }
  }

  void _updatePromptShortcut(String presetId, AppShortcutBinding? shortcut) {
    setState(() {
      _promptShortcuts = Map<String, AppShortcutBinding?>.from(_promptShortcuts)
        ..[presetId] = shortcut;
    });
    unawaited(
      _persistShortcutPreference(
        _promptShortcutPreferenceKey(presetId),
        shortcut,
      ),
    );
  }

  Future<void> _persistShortcutPreference(
    String key,
    AppShortcutBinding? shortcut,
  ) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(key, shortcut?.encode() ?? _disabledShortcutValue);
    await _syncGlobalHotkeys();
  }

  Future<void> _syncGlobalHotkeys() async {
    final registrations = <GlobalHotkeyRegistration>[
      if (_reflectShortcut case final shortcut?)
        GlobalHotkeyRegistration(
          name: 'Reflect now',
          binding: shortcut,
          onPressed: () => unawaited(_runGlobalReflection()),
        ),
      if (_sourceReflectShortcut case final shortcut?)
        GlobalHotkeyRegistration(
          name: 'Reflect from source',
          binding: shortcut,
          onPressed: () => unawaited(_runGlobalReflection(chooseSource: true)),
        ),
      if (_clipboardAnswerShortcut case final shortcut?)
        GlobalHotkeyRegistration(
          name: 'Answer clipboard',
          binding: shortcut,
          onPressed: () => unawaited(_answerClipboard(background: true)),
        ),
      for (final preset in reflectionPromptPresets)
        if (_promptShortcuts[preset.id] case final shortcut?)
          GlobalHotkeyRegistration(
            name: preset.title,
            binding: shortcut,
            onPressed:
                () => unawaited(_runGlobalReflection(promptPreset: preset)),
          ),
    ];
    final error = await _globalHotkeys.sync(registrations);
    if (mounted) setState(() => _globalHotkeyError = error);
  }

  Future<void> _persistDeliveryPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    await Future.wait([
      prefs.setBool('proactive_enabled', _proactiveEnabled),
      prefs.setBool('proactive_voice_enabled', _proactiveVoiceEnabled),
      prefs.setBool('proactive_feed_enabled', _proactiveFeedEnabled),
      prefs.setBool(
        'proactive_notifications_enabled',
        _proactiveNotificationsEnabled,
      ),
      prefs.setBool('event_notifications_enabled', _eventNotificationsEnabled),
      prefs.setBool('notifications_muted', _notificationsMuted),
    ]);
    await _syncNotificationMonitoring();
  }

  Future<void> _syncNotificationMonitoring() async {
    if (!_deliveryPreferencesLoaded) return;
    final events = !_notificationsMuted && _eventNotificationsEnabled;
    final proactive =
        !_notificationsMuted &&
        _proactiveEnabled &&
        _proactiveNotificationsEnabled;
    if (_apiBase.isEmpty || (!events && !proactive)) {
      await _notificationController.stopMonitoring();
      return;
    }
    await _notificationController.startMonitoring(
      _apiBase,
      eventNotifications: events,
      proactiveNotifications: proactive,
    );
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    _audioPlayer.dispose();
    _playerStateSubscription?.cancel();
    _ipTextController.dispose();
    _composerController.dispose();
    _captureSub?.cancel();
    _capture.dispose();
    _fpsController.dispose();
    _statusTimer?.cancel();
    unawaited(_globalHotkeys.dispose());
    super.dispose();
  }

  void _setupAudioPlayerListener() {
    _playerStateSubscription = _audioPlayer.onPlayerStateChanged.listen((
      state,
    ) {
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
      await _sendTurn(audio: audioData);
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

    return (BytesBuilder()
          ..add(header)
          ..add(fullAudioData))
        .toBytes();
  }

  /// Send one conversation turn — spoken (`audio`) or typed (`text`).
  ///
  /// Both go to `/chat/audio`, which skips ASR when text is supplied, so live
  /// frames, memory tools and spoken replies work the same either way. The
  /// user's words come back on the `query` line, so neither path echoes locally.
  Future<void> _sendTurn({
    Uint8List? audio,
    String? text,
    String? contextOverride,
    bool includeImage = true,
    bool? thinkingOverride,
  }) async {
    try {
      if (mounted) {
        setState(
          () =>
              _backendActivity =
                  text != null ? 'Sending message' : 'Uploading audio',
        );
      }
      // Clear the audio queue for the new response
      // and stop any ongoing playback from a previous turn.
      _isAudioPlaying = false;
      await _audioPlayer.stop();
      _audioQueue.clear();

      final url = Uri.parse('$_apiBase/chat/audio');

      final Map<String, dynamic> requestBody = {
        'data': audio,
        'text': text,
        'image':
            includeImage && _fileImage != null
                ? base64.encode(_fileImage!)
                : null,
        'talking': _isTalking,
        'context': contextOverride ?? _currentContext,
        'live': _isLive, // Add this line
        'memory': _useMemory, // Add this line
        'thinking': thinkingOverride ?? _conversationThinking,
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
        if (mounted)
          setState(
            () =>
                _backendActivity =
                    'Backend error (${streamedResponse.statusCode})',
          );
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
            if (mounted)
              setState(() => _backendActivity = 'Generating response');
            final queryText = jsonResponse['text'];
            if (mounted) {
              setState(() {
                _chatHistory.add(
                  ChatMessage(
                    sender: MessageSender.user,
                    text: "User: $queryText",
                  ),
                );
                // Add a placeholder for the assistant's response
                currentAssistantMessage = ChatMessage(
                  sender: MessageSender.assistant,
                  text: "Assistant: ",
                );
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
            if (mounted)
              setState(
                () => _backendActivity = 'Backend: ${jsonResponse['stage']}',
              );
          } else if (type == 'error') {
            if (mounted)
              setState(
                () => _backendActivity = 'Error: ${jsonResponse['message']}',
              );
          } else if (type == 'done') {
            if (mounted)
              setState(
                () =>
                    _backendActivity = 'Ready (${jsonResponse['total_ms']} ms)',
              );
          }
        } catch (e) {
          print("Error processing stream line: $e. Line: '$line'");
        }
      }
      // Once the stream is finished, save the complete audio to the message
      if (currentAssistantMessage != null) {
        currentAssistantMessage!.fullAudio = _mergeWavBytes(
          assistantAudioChunks,
        );
      }
    } catch (error) {
      if (kDebugMode) {
        print('Error processing audio: $error');
      }
      if (mounted) setState(() => _backendActivity = 'Connection failed');
    }
  }

  Future<void> _sendTypedMessage() async {
    final text = _composerController.text.trim();
    if (text.isEmpty || _isProcessing || _isRecording) return;
    if (_apiBase.isEmpty) {
      _showSnack('Set the home hub address first');
      return;
    }
    _composerController.clear();
    setState(() => _isProcessing = true);
    try {
      await _sendTurn(text: text);
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Future<void> _answerClipboard({bool background = false}) async {
    Future<void> report(String message) async {
      _showSnack(message);
      if (background) await showDesktopAlert('Answer clipboard', message);
    }

    if (_isProcessing || _isRecording) {
      await report('HomeMind is already handling another request');
      return;
    }
    if (_apiBase.isEmpty) {
      await report('Connect to the home hub first');
      return;
    }

    final clipboard = await Clipboard.getData(Clipboard.kTextPlain);
    final clipboardText = clipboard?.text;
    if (clipboardText == null || clipboardText.trim().isEmpty) {
      await report('The clipboard does not contain any text');
      return;
    }

    if (mounted) {
      setState(() {
        _isProcessing = true;
        _workspaceIndex = 0;
      });
    }
    try {
      await _sendTurn(
        text: buildClipboardAnswerRequest(
          _clipboardAnswerPrompt,
          clipboardText,
        ),
        contextOverride: 'talker',
        includeImage: false,
        thinkingOverride: _thinkingForShortcut(clipboardAnswerActionId),
      );
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  String get _apiBase {
    final value = _ipTextController.text.trim();
    if (value.isEmpty) return '';

    final hasScheme =
        value.startsWith('http://') || value.startsWith('https://');
    var uri = Uri.parse(hasScheme ? value : 'http://$value');

    // Plain hostnames/IPs use the FastAPI development port. An HTTPS URL is
    // left on its standard port so Tailscale Serve/Cloudflare also work.
    if (!uri.hasPort && uri.scheme == 'http') {
      uri = uri.replace(port: 8000);
    }
    return uri.toString().replaceFirst(RegExp(r'/$'), '');
  }

  Future<void> _pollBackendStatus() async {
    final apiBase = _apiBase;
    if (apiBase.isEmpty || _statusPollInFlight) return;
    _statusPollInFlight = true;
    try {
      // Keep this probe single-flight. The old two-second timer could start a
      // new probe before the previous three-second timeout had completed,
      // which made a healthy Tailscale connection flicker offline.
      final response = await http
          .get(Uri.parse('$apiBase/status'))
          .timeout(const Duration(seconds: 4));
      if (response.statusCode != 200) {
        throw Exception(
          'backend health request returned ${response.statusCode}',
        );
      }
      final value = decodeJsonResponse(response) as Map<String, dynamic>;
      if (_apiBase != apiBase) return;
      _consecutiveStatusFailures = 0;
      if (mounted)
        setState(() {
          _backendConnected = true;
          _backendStatus = value;
          final pipeline = value['pipeline'] as Map?;
          if (pipeline != null) {
            final stage = '${pipeline['stage'] ?? 'ready'}'.replaceAll(
              '_',
              ' ',
            );
            _backendActivity =
                pipeline['active'] == true ? 'Backend: $stage' : stage;
          }
        });
      await _syncNotificationMonitoring();
      if (_kokoroVoices.isEmpty) {
        unawaited(_loadTtsSettings());
      }
    } catch (error) {
      if (_apiBase == apiBase) {
        _consecutiveStatusFailures++;
        // Tailscale can briefly pause while Android changes radio or network.
        // Require three failed probes before declaring the hub offline.
        if (_consecutiveStatusFailures >= 3) {
          if (mounted)
            setState(() {
              _backendConnected = false;
              if (!_isProcessing) _backendActivity = 'Backend offline';
            });
          // Re-sync proactive ids on the next successful connect (the server
          // may have restarted and reset its counter).
          _proactiveSynced = false;
        }
        if (kDebugMode) {
          print(
            'Home hub status probe failed '
            '($_consecutiveStatusFailures/3): $error',
          );
        }
      }
    } finally {
      _statusPollInFlight = false;
    }
  }

  Future<void> _clearChatHistory() async {
    try {
      final response = await http
          .post(Uri.parse('$_apiBase/history/clear'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) {
        throw Exception(decodeUtf8Response(response));
      }
      if (mounted) setState(() => _chatHistory.clear());
      _showSnack('Conversation history cleared');
      await _pollBackendStatus();
    } catch (e) {
      _showSnack('Could not clear conversation history: $e');
    }
  }

  Future<void> _clearMemory() async {
    try {
      final response = await http
          .post(Uri.parse('$_apiBase/memory/clear'))
          .timeout(const Duration(seconds: 15));
      if (response.statusCode != 200) {
        throw Exception(decodeUtf8Response(response));
      }
      final result = decodeJsonResponse(response) as Map<String, dynamic>;
      if (result['cleared'] != true)
        throw Exception(result['error'] ?? 'unknown error');
      _showSnack('Long-term activity memory cleared');
    } catch (e) {
      _showSnack('Could not clear activity memory: $e');
    }
  }

  /// Poll for unprompted proactive insights and play their speech on THIS
  /// device. New insights also land in the chat list for replay.
  Future<void> _fetchProactiveInsights() async {
    if (_ipTextController.text.trim().isEmpty) return;
    try {
      final response = await http
          .get(Uri.parse('$_apiBase/proactive?since=$_lastProactiveId'))
          .timeout(const Duration(seconds: 3));
      if (response.statusCode != 200) return;
      final data = decodeJsonResponse(response) as Map<String, dynamic>;

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
        if (!_proactiveEnabled) continue;

        Uint8List? audio;
        final audioB64 = map['audio'];
        if ((_proactiveVoiceEnabled || _proactiveFeedEnabled) &&
            audioB64 is String &&
            audioB64.isNotEmpty) {
          audio = base64.decode(audioB64);
        }

        final clip = map['clip'] as Map<String, dynamic>?;
        if (mounted && _proactiveFeedEnabled) {
          setState(() {
            _chatHistory.add(
              ChatMessage(
                sender: MessageSender.assistant,
                text: 'Insight: ${map['text'] ?? ''}',
                fullAudio: audio,
                clipId:
                    map['can_ask'] == true ? map['clip_id']?.toString() : null,
                clipCoversSeconds:
                    (clip?['covers_seconds'] as num?)?.toDouble(),
                clipPlaysSeconds: (clip?['plays_seconds'] as num?)?.toDouble(),
              ),
            );
          });
        }

        // Play on this device by enqueueing into the shared audio queue.
        if (_proactiveVoiceEnabled && audio != null) {
          _audioQueue.add(audio);
          if (!_isAudioPlaying) _playNextInQueue();
        }
      }
    } catch (_) {
      // Transient/offline — connectivity is surfaced by the status poll.
    }
  }

  Future<void> _runGlobalReflection({
    bool chooseSource = false,
    ReflectionPromptPreset? promptPreset,
  }) async {
    // Stealing focus defeats the point: the shortcut is pressed while working
    // in another window, and the answer arrives as speech plus a message
    // waiting in Home. The one exception is picking a source, which cannot be
    // done without a visible window.
    if (chooseSource) {
      try {
        await _globalHotkeys.bringAppToFront();
      } catch (error) {
        if (mounted) {
          setState(
            () =>
                _globalHotkeyError =
                    'The shortcut fired, but HomeMind could not come forward: $error',
          );
        }
      }
    }
    if (!mounted) return;
    if (chooseSource) {
      await _showReflectionSourcePicker(
        thinking: _thinkingForShortcut('reflect_from_source'),
      );
    } else {
      final actionId = promptPreset?.id ?? 'reflect_now';
      await _reflectOnScreen(
        question: promptPreset == null ? null : _promptTextFor(promptPreset),
        actionLabel: promptPreset?.title,
        background: true,
        thinking: _thinkingForShortcut(actionId),
      );
    }
  }

  Future<List<_ReflectionSourceOption>> _reflectionSources() async {
    final response = await http
        .get(Uri.parse('$_apiBase/reflect/sources'))
        .timeout(const Duration(seconds: 5));
    if (response.statusCode != 200) {
      throw Exception('backend returned ${response.statusCode}');
    }
    final data = decodeJsonResponse(response) as Map<String, dynamic>;
    return ((data['sources'] as List?) ?? const [])
        .whereType<Map>()
        .map(
          (item) =>
              _ReflectionSourceOption.fromJson(Map<String, dynamic>.from(item)),
        )
        .where((source) => source.id.isNotEmpty)
        .toList();
  }

  Future<void> _showReflectionSourcePicker({bool? thinking}) async {
    if (_reflecting) {
      _showSnack('A reflection is already running');
      return;
    }
    if (_choosingReflectionSource) return;
    if (_apiBase.isEmpty) {
      _showSnack('Connect to the home hub first');
      return;
    }
    _choosingReflectionSource = true;
    _showSnack('Loading live reflection sources…');
    List<_ReflectionSourceOption> sources;
    try {
      sources = await _reflectionSources();
    } catch (error) {
      _showSnack('Could not load reflection sources: $error');
      _choosingReflectionSource = false;
      return;
    }
    if (!mounted) {
      _choosingReflectionSource = false;
      return;
    }
    _ReflectionSourceOption? selected;
    try {
      selected = await showModalBottomSheet<_ReflectionSourceOption>(
        context: context,
        isScrollControlled: true,
        backgroundColor: _panel,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        ),
        builder:
            (sheetContext) => SafeArea(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(18, 10, 18, 24),
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
                    const SizedBox(height: 18),
                    const Text(
                      'Reflect from source',
                      style: TextStyle(
                        fontSize: 19,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 5),
                    const Text(
                      'Choose the exact live frames HomeMind should attach.',
                      style: TextStyle(color: _muted, fontSize: 11.5),
                    ),
                    const SizedBox(height: 14),
                    if (sources.isEmpty)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 22),
                        child: Center(
                          child: Text(
                            'No reflection sources are configured.',
                            style: TextStyle(color: _muted),
                          ),
                        ),
                      )
                    else
                      ...sources.map((source) {
                        final isMobile = source.id.startsWith('mobile_');
                        final icon =
                            source.id == 'pc_screen'
                                ? Icons.desktop_windows_outlined
                                : isMobile
                                ? (source.context == 'screen'
                                    ? Icons.phone_android_outlined
                                    : Icons.phone_iphone_outlined)
                                : Icons.videocam_outlined;
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Material(
                            color: _panelRaised,
                            borderRadius: BorderRadius.circular(14),
                            child: ListTile(
                              enabled: source.available,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(14),
                              ),
                              leading: Icon(
                                icon,
                                color: source.available ? _mint : _muted,
                              ),
                              title: Text(
                                source.label,
                                style: const TextStyle(
                                  fontSize: 13,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                              subtitle: Text(
                                source.detail,
                                style: const TextStyle(
                                  color: _muted,
                                  fontSize: 10,
                                ),
                              ),
                              trailing:
                                  source.available
                                      ? const Icon(
                                        Icons.chevron_right_rounded,
                                        color: _muted,
                                      )
                                      : const Text(
                                        'Unavailable',
                                        style: TextStyle(
                                          color: _muted,
                                          fontSize: 9.5,
                                        ),
                                      ),
                              onTap:
                                  source.available
                                      ? () =>
                                          Navigator.pop(sheetContext, source)
                                      : null,
                            ),
                          ),
                        );
                      }),
                  ],
                ),
              ),
            ),
      );
    } finally {
      _choosingReflectionSource = false;
    }
    if (selected != null && mounted) {
      await _reflectOnScreen(requestedSource: selected, thinking: thinking);
    }
  }

  void _runPromptPreset(ReflectionPromptPreset preset) {
    unawaited(
      _reflectOnScreen(
        question: _promptTextFor(preset),
        actionLabel: preset.title,
        thinking: _thinkingForShortcut(preset.id),
      ),
    );
  }

  IconData _promptPresetIcon(ReflectionPromptKind kind) {
    return switch (kind) {
      ReflectionPromptKind.reading => Icons.menu_book_outlined,
      ReflectionPromptKind.code => Icons.code_rounded,
      ReflectionPromptKind.guidance => Icons.route_outlined,
    };
  }

  Color _promptPresetColor(ReflectionPromptKind kind) {
    return switch (kind) {
      ReflectionPromptKind.reading => _mint,
      ReflectionPromptKind.code => const Color(0xFF62B5FF),
      ReflectionPromptKind.guidance => const Color(0xFFFFC857),
    };
  }

  Future<void> _showPromptActionPicker() async {
    if (_reflecting) {
      _showSnack('A reflection is already running');
      return;
    }
    final selected = await showModalBottomSheet<ReflectionPromptPreset>(
      context: context,
      isScrollControlled: true,
      backgroundColor: _panel,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      builder:
          (sheetContext) => SafeArea(
            child: SizedBox(
              height: MediaQuery.sizeOf(sheetContext).height * .76,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.fromLTRB(18, 10, 18, 0),
                    child: Column(
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
                        const SizedBox(height: 18),
                        const Text(
                          'Guided reflection',
                          style: TextStyle(
                            fontSize: 19,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        const SizedBox(height: 5),
                        const Text(
                          'Run a general prompt on the current Screen or Camera context.',
                          style: TextStyle(color: _muted, fontSize: 11.5),
                        ),
                        const SizedBox(height: 14),
                      ],
                    ),
                  ),
                  Expanded(
                    child: ListView(
                      key: const Key('prompt-action-list'),
                      padding: const EdgeInsets.fromLTRB(18, 0, 18, 24),
                      children: [
                        for (final preset in reflectionPromptPresets)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 8),
                            child: Material(
                              color: _panelRaised,
                              borderRadius: BorderRadius.circular(14),
                              child: ListTile(
                                key: ValueKey('prompt-action-${preset.id}'),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(14),
                                ),
                                leading: Icon(
                                  _promptPresetIcon(preset.kind),
                                  color: _promptPresetColor(preset.kind),
                                ),
                                title: Text(
                                  preset.title,
                                  style: const TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                                subtitle: Text(
                                  '${_promptShortcuts[preset.id]?.label ?? 'Shortcut disabled'}'
                                  ' • ${preset.description}',
                                  style: const TextStyle(
                                    color: _muted,
                                    fontSize: 10,
                                  ),
                                ),
                                trailing: const Icon(
                                  Icons.chevron_right_rounded,
                                  color: _muted,
                                ),
                                onTap:
                                    () => Navigator.pop(sheetContext, preset),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
    );
    if (selected != null && mounted) {
      await _reflectOnScreen(
        question: selected.prompt,
        actionLabel: selected.title,
        thinking: _thinkingForShortcut(selected.id),
      );
    }
  }

  /// Ask the backend what it makes of the last few seconds of live frames.
  ///
  /// Bound to the configurable reflection shortcut (Alt+Shift+W by default):
  /// the point is to get an opinion on whatever is on screen *right now* (the
  /// page being read, the error in the terminal, the thing just highlighted)
  /// without breaking off to type a question.
  Future<void> _reflectOnScreen({
    _ReflectionSourceOption? requestedSource,
    String? question,
    String? actionLabel,
    // Set when a global shortcut started this and the window stayed back, so
    // failures are reported by the OS instead of to an unwatched SnackBar.
    bool background = false,
    bool? thinking,
  }) async {
    final label = actionLabel ?? 'Reflection';
    Future<void> report(String message) async {
      _showSnack(message);
      if (background) await showDesktopAlert(label, message);
    }

    if (_reflecting) {
      await report('A reflection is already running');
      return;
    }
    if (_apiBase.isEmpty) {
      await report('Connect to the home hub first');
      return;
    }
    if (Navigator.of(context).canPop()) {
      Navigator.of(context).popUntil((route) => route.isFirst);
    }
    setState(() {
      _reflecting = true;
      // The answer lands in the Home conversation, so go there to see it.
      _workspaceIndex = 0;
    });
    _showSnack(
      actionLabel == null
          ? 'Looking at the last ${_reflectFrames}s…'
          : '$actionLabel • looking at the last ${_reflectFrames}s…',
    );
    try {
      final response = await http
          .post(
            Uri.parse('$_apiBase/reflect'),
            headers: const {'Content-Type': 'application/json'},
            body: json.encode({
              'context':
                  requestedSource?.context ??
                  (_currentContext == 'camera' ? 'camera' : 'screen'),
              if (requestedSource != null) 'source': requestedSource.id,
              'frames': _reflectFrames,
              'speak': _proactiveVoiceEnabled,
              'thinking':
                  thinking ??
                  ((requestedSource?.context ?? _currentContext) == 'screen'),
              if (question != null && question.trim().isNotEmpty)
                'question': question.trim(),
            }),
          )
          // The VLM needs real time on a cold model; a short timeout here just
          // throws away an answer the server is still producing.
          .timeout(const Duration(seconds: 180));
      final data = decodeJsonResponse(response) as Map<String, dynamic>;
      if (response.statusCode != 200) {
        await report(
          'Reflection failed: ${data['error'] ?? response.statusCode}',
        );
        return;
      }
      final audioB64 = data['audio'];
      final audio =
          audioB64 is String && audioB64.isNotEmpty
              ? base64.decode(audioB64)
              : null;
      final clip = data['clip'] as Map<String, dynamic>?;
      if (!mounted) return;
      setState(() {
        _chatHistory.add(
          ChatMessage(
            sender: MessageSender.assistant,
            text: 'Reflection: ${data['text'] ?? ''}',
            fullAudio: audio,
            clipId: data['clip_id']?.toString(),
            clipCoversSeconds: (clip?['covers_seconds'] as num?)?.toDouble(),
            clipPlaysSeconds: (clip?['plays_seconds'] as num?)?.toDouble(),
          ),
        );
      });
      if (_proactiveVoiceEnabled && audio != null) {
        _audioQueue.add(audio);
        if (!_isAudioPlaying) _playNextInQueue();
      }
    } catch (e) {
      await report('Could not reflect on the screen: $e');
    } finally {
      if (mounted) setState(() => _reflecting = false);
    }
  }

  Future<void> _fetchNotifications() async {
    if (_apiBase.isEmpty) return;
    try {
      final response = await http
          .get(
            Uri.parse(
              '$_apiBase/notifications?since=$_lastNotificationSequence&limit=50',
            ),
          )
          .timeout(const Duration(seconds: 3));
      if (response.statusCode != 200) return;
      final data = decodeJsonResponse(response) as Map<String, dynamic>;
      final latest = (data['latest_sequence'] as num?)?.toInt() ?? 0;
      if (mounted) {
        setState(() {
          _unreadNotifications = (data['unread_count'] as num?)?.toInt() ?? 0;
        });
      }
      // The native foreground monitor owns system notifications. Flutter only
      // keeps the unread badge and inbox in sync.
      _lastNotificationSequence = latest;
    } catch (_) {
      // The connection indicator already reports backend availability.
    }
  }

  void _openNotifications() {
    _openWorkspace(
      4,
      NotificationsScreen(
        apiBase: _apiBase,
        onUnreadChanged: (count) {
          if (mounted) setState(() => _unreadNotifications = count);
        },
      ),
    );
  }

  Widget _notificationButton() {
    return Stack(
      clipBehavior: Clip.none,
      children: [
        IconButton(
          tooltip:
              _notificationsMuted ? 'Notifications muted' : 'Notifications',
          onPressed: _openNotifications,
          style: IconButton.styleFrom(backgroundColor: _panel),
          icon: Icon(
            _notificationsMuted
                ? Icons.notifications_off_outlined
                : _unreadNotifications > 0
                ? Icons.notifications_active_outlined
                : Icons.notifications_none_outlined,
            size: 20,
          ),
        ),
        if (_unreadNotifications > 0)
          Positioned(
            right: -2,
            top: -3,
            child: Container(
              constraints: const BoxConstraints(minWidth: 17, minHeight: 17),
              padding: const EdgeInsets.symmetric(horizontal: 4),
              decoration: const BoxDecoration(
                color: Color(0xFFFF607C),
                shape: BoxShape.circle,
              ),
              alignment: Alignment.center,
              child: Text(
                _unreadNotifications > 99 ? '99+' : '$_unreadNotifications',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 8,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
          ),
      ],
    );
  }

  Future<void> _setBackendCapture(bool start) async {
    final response = await http
        .post(
          Uri.parse('$_apiBase/capture/control'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({
            'action': start ? 'start' : 'stop',
            'source': _captureSource.name,
          }),
        )
        .timeout(const Duration(seconds: 5));
    if (response.statusCode != 200) {
      throw Exception(
        'backend returned ${response.statusCode}: '
        '${decodeUtf8Response(response)}',
      );
    }
    await _pollBackendStatus();
  }

  /// Pause/resume the desktop screen capture on the backend.
  Future<void> _toggleScreen(bool pause) async {
    try {
      final response = await http
          .post(
            Uri.parse('$_apiBase/screen/control'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'action': pause ? 'pause' : 'resume'}),
          )
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) {
        throw Exception(decodeUtf8Response(response));
      }
      await _pollBackendStatus();
    } catch (e) {
      _showSnack('Failed to ${pause ? 'pause' : 'resume'} screen: $e');
    }
  }

  /// Pause/resume a single camera worker on the backend.
  Future<void> _toggleCamera(String cameraId, bool pause) async {
    try {
      final response = await http
          .post(
            Uri.parse('$_apiBase/cameras/$cameraId/control'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'action': pause ? 'pause' : 'resume'}),
          )
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) {
        throw Exception(decodeUtf8Response(response));
      }
      await _pollBackendStatus();
    } catch (e) {
      _showSnack('Failed to ${pause ? 'pause' : 'resume'} camera: $e');
    }
  }

  int get _fps {
    final v = int.tryParse(_fpsController.text.trim()) ?? 5;
    return v.clamp(1, 60);
  }

  Future<void> _startCapture() async {
    if (_apiBase.isEmpty) {
      _showSnack('Enter the home hub address first');
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
        apiBase: _apiBase,
        frontCamera: _frontCamera,
      );
      final index = _captureSource == CaptureSource.camera ? 2 : 1;
      if (mounted)
        setState(() {
          _currentContext = _contextOptions[index];
          _conversationThinking = index == 1;
          for (var i = 0; i < _selectedContexts.length; i++) {
            _selectedContexts[i] = i == index;
          }
          _isLive = true;
          _backendActivity = 'Waiting for ${_captureSource.name} frames';
        });
    } catch (e) {
      try {
        await _setBackendCapture(false);
      } catch (_) {}
      _showSnack('Failed to start capture: $e');
    }
  }

  Future<void> _stopCapture() async {
    await _capture.stop();
    try {
      await _setBackendCapture(false);
      if (mounted)
        setState(() {
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
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.circle, size: 11, color: color),
          const SizedBox(width: 5),
          Text(text, style: const TextStyle(fontSize: 12)),
        ],
      ),
    );
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: _panel.withOpacity(.7),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _line),
      ),
      child: Wrap(
        children: [
          indicator(
            _backendConnected ? Colors.green : Colors.red,
            _backendConnected ? 'Backend connected' : 'Backend offline',
          ),
          indicator(
            asrReady ? Colors.green : Colors.red,
            asrReady ? 'Parakeet ready' : 'Parakeet unavailable',
          ),
          indicator(
            active ? (healthy ? Colors.green : Colors.orange) : Colors.grey,
            active
                ? '${mobile['source']} active ($frames frames)'
                : 'Vision stopped',
          ),
          indicator(
            _isProcessing ? Colors.blue : Colors.grey,
            _backendActivity,
          ),
        ],
      ),
    );
  }

  /// A row for one capture source: status dot + label + pause/resume toggle.
  Widget _captureSourceRow({
    required IconData icon,
    required String label,
    required String statusText,
    required Color dotColor,
    required bool paused,
    required bool available,
    required VoidCallback? onToggle,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          Icon(Icons.circle, size: 10, color: dotColor),
          const SizedBox(width: 8),
          Icon(icon, size: 16, color: Colors.white70),
          const SizedBox(width: 6),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  statusText,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(fontSize: 11, color: Colors.white54),
                ),
              ],
            ),
          ),
          if (available)
            IconButton(
              tooltip: paused ? 'Resume' : 'Pause',
              visualDensity: VisualDensity.compact,
              iconSize: 20,
              icon: Icon(
                paused ? Icons.play_circle_fill : Icons.pause_circle_filled,
                color: paused ? Colors.greenAccent : Colors.amber,
              ),
              onPressed: onToggle,
            )
          else
            const SizedBox(width: 40),
        ],
      ),
    );
  }

  /// Live capture sources — desktop screen + each discovered camera — with an
  /// activity indicator and a pause/resume control for each.
  Widget _buildCaptureSourcesPanel() {
    final screen = (_backendStatus['screen_stream'] as Map?) ?? const {};
    final screenConfigured = screen['configured'] == true;
    final screenPaused = screen['paused'] == true;
    final screenHealthy = screen['healthy'] == true;
    final cameras = (_backendStatus['cameras'] as List?) ?? const [];

    final rows = <Widget>[];
    if (screenConfigured) {
      rows.add(
        _captureSourceRow(
          icon: Icons.desktop_windows,
          label: 'Desktop screen',
          statusText:
              screenPaused
                  ? 'Paused'
                  : (screenHealthy
                      ? 'Recording · ${screen['frames'] ?? 0} frames'
                      : 'Starting…'),
          dotColor:
              screenPaused
                  ? Colors.amber
                  : (screenHealthy ? Colors.green : Colors.grey),
          paused: screenPaused,
          available: true,
          onToggle: () => _toggleScreen(!screenPaused),
        ),
      );
    }
    for (final c in cameras) {
      final cam = (c as Map);
      final id = '${cam['camera_id']}';
      final paused = cam['paused'] == true;
      final connected = cam['connected'] == true;
      final events = cam['events_logged'] ?? 0;
      final summary = '${cam['last_summary'] ?? ''}';
      final motion = (cam['last_motion'] as Map?);
      final idle =
          motion != null &&
          motion['warming'] != true &&
          (motion['motion_frames'] ?? 0) == 0;
      rows.add(
        _captureSourceRow(
          icon: Icons.videocam,
          label: '${cam['name'] ?? id}',
          statusText:
              !connected
                  ? 'Offline${cam['error'] != null ? ' · ${cam['error']}' : ''}'
                  : paused
                  ? 'Paused · $events events'
                  : idle
                  ? 'Idle · watching for motion · $events events'
                  : (summary.isNotEmpty
                      ? summary
                      : 'Watching · $events events'),
          dotColor:
              !connected ? Colors.red : (paused ? Colors.amber : Colors.green),
          paused: paused,
          available: connected,
          onToggle: () => _toggleCamera(id, !paused),
        ),
      );
    }
    if (rows.isEmpty) {
      rows.add(
        const Padding(
          padding: EdgeInsets.symmetric(vertical: 6),
          child: Text(
            'No capture sources active',
            style: TextStyle(fontSize: 12, color: Colors.white54),
          ),
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: _panel.withOpacity(.7),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _line),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Padding(
            padding: EdgeInsets.only(bottom: 4),
            child: Text(
              'CAPTURE SOURCES',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.2,
                color: Colors.white60,
              ),
            ),
          ),
          ...rows,
        ],
      ),
    );
  }

  Widget _buildCapturePanel() {
    // Native frame capture is Android-only; hide the controls elsewhere.
    if (!_capture.isSupported) return const SizedBox.shrink();
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
          Wrap(
            alignment: WrapAlignment.center,
            crossAxisAlignment: WrapCrossAlignment.center,
            spacing: 6,
            runSpacing: 6,
            children: [
              const Text('Frame source: ', style: TextStyle(fontSize: 14)),
              ChoiceChip(
                label: const Text('Camera'),
                selected: _captureSource == CaptureSource.camera,
                onSelected:
                    running
                        ? null
                        : (_) => setState(
                          () => _captureSource = CaptureSource.camera,
                        ),
              ),
              ChoiceChip(
                label: const Text('Screen'),
                selected: _captureSource == CaptureSource.screen,
                onSelected:
                    running
                        ? null
                        : (_) => setState(
                          () => _captureSource = CaptureSource.screen,
                        ),
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
              if (running) _buttonsProcessing('Set FPS', 60, _applyFps),
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
          hintText: '100.x.y.z or https://hub.example.com',
          prefixIcon: const Icon(Icons.router_outlined, size: 19),
          suffixIcon: IconButton(
            tooltip: 'Reconnect',
            onPressed: _connectToHomeHub,
            icon: const Icon(Icons.refresh_rounded, size: 19),
          ),
          helperText: 'Use the PC Wi-Fi IP; localhost points to this phone',
        ),
        onSubmitted: (_) => _connectToHomeHub(),
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
            colors:
                _isRecording
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

  /// Typing is the equal of the mic: the same turn, minus speech recognition.
  /// Useful when dictating is awkward, and when ASR is unavailable.
  Widget _buildTypedComposer() {
    return ValueListenableBuilder<TextEditingValue>(
      valueListenable: _composerController,
      builder: (context, value, _) {
        final canSend =
            value.text.trim().isNotEmpty && !_isProcessing && !_isRecording;
        return Row(
          children: [
            Expanded(
              child: TextField(
                controller: _composerController,
                enabled: !_isRecording,
                minLines: 1,
                maxLines: 4,
                textInputAction: TextInputAction.send,
                style: const TextStyle(fontSize: 13.5),
                decoration: InputDecoration(
                  isDense: true,
                  filled: true,
                  fillColor: _ink,
                  hintText:
                      _isRecording
                          ? 'Listening…'
                          : 'Type a message to HomeMind',
                  hintStyle: const TextStyle(color: _muted, fontSize: 13),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 12,
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(14),
                    borderSide: const BorderSide(color: _line),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(14),
                    borderSide: const BorderSide(color: _line),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(14),
                    borderSide: const BorderSide(color: _mint),
                  ),
                ),
                onSubmitted: (_) => _sendTypedMessage(),
              ),
            ),
            const SizedBox(width: 9),
            IconButton(
              tooltip: 'Send message',
              onPressed: canSend ? _sendTypedMessage : null,
              style: IconButton.styleFrom(
                backgroundColor: canSend ? _mint : _panel,
                minimumSize: const Size(46, 46),
              ),
              icon:
                  _isProcessing && !_isRecording
                      ? const SizedBox(
                        width: 18,
                        height: 18,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: _mint,
                        ),
                      )
                      : Icon(
                        Icons.send_rounded,
                        size: 20,
                        color: canSend ? _ink : _muted,
                      ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildTranscriptionList() {
    return Expanded(
      child: ListView.builder(
        padding: const EdgeInsets.fromLTRB(18, 8, 18, 18),
        itemCount: _chatHistory.length,
        itemBuilder: (context, index) {
          final message = _chatHistory[index];
          return MessageBubble(
            message: message,
            audioPlayer: _audioPlayer,
            apiBase: _apiBase,
          );
        },
      ),
    );
  }

  void _openWorkspace(int index, Widget screen) {
    if (MediaQuery.sizeOf(context).width >= 960) {
      setState(() => _workspaceIndex = index);
      return;
    }
    if (_scaffoldKey.currentState?.isDrawerOpen == true) {
      Navigator.of(context).pop();
    }
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => screen));
  }

  Widget _settingsScreen() {
    return SettingsScreen(
      reflectionShortcut: _reflectShortcut,
      sourceReflectionShortcut: _sourceReflectShortcut,
      clipboardAnswerShortcut: _clipboardAnswerShortcut,
      reflectFrames: _reflectFrames,
      onReflectionShortcutChanged: _updateReflectShortcut,
      onSourceReflectionShortcutChanged: _updateSourceReflectShortcut,
      onClipboardAnswerShortcutChanged: _updateClipboardAnswerShortcut,
      onReflectNow:
          () => _reflectOnScreen(thinking: _thinkingForShortcut('reflect_now')),
      onChooseReflectionSource:
          () => _showReflectionSourcePicker(
            thinking: _thinkingForShortcut('reflect_from_source'),
          ),
      onAnswerClipboard: _answerClipboard,
      promptShortcuts: _promptShortcuts,
      onPromptShortcutChanged: _updatePromptShortcut,
      promptTexts: _promptTexts,
      onPromptTextChanged: _updatePromptText,
      onRunPrompt: _runPromptPreset,
      clipboardAnswerPrompt: _clipboardAnswerPrompt,
      onClipboardAnswerPromptChanged: _updateClipboardAnswerPrompt,
      globalHotkeysSupported: _globalHotkeys.isSupported,
      globalHotkeyError: _globalHotkeyError,
      kokoroVoice: _kokoroVoice,
      kokoroVoices: _kokoroVoices,
      onKokoroVoiceChanged: (voice) {
        unawaited(_updateKokoroVoice(voice));
      },
      kokoroVoiceSaving: _kokoroVoiceSaving,
      kokoroVoiceError: _kokoroVoiceError,
      captureSources: _captureSources,
      captureSettingsLoading: _captureSettingsLoading,
      captureSettingsSavingSource: _captureSettingsSavingSource,
      captureSettingsError: _captureSettingsError,
      shortcutThinking: _shortcutThinking,
      onShortcutThinkingChanged: _updateShortcutThinking,
      onCaptureSourceChanged: (sourceId, fps, interval, thinking) {
        unawaited(_updateCaptureSource(sourceId, fps, interval, thinking));
      },
    );
  }

  Widget _workspaceNavItem({
    required IconData icon,
    required String label,
    VoidCallback? onTap,
    bool selected = false,
    String? badge,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
      child: Material(
        color: selected ? _mint.withValues(alpha: .13) : Colors.transparent,
        borderRadius: BorderRadius.circular(10),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(10),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 10),
            child: Row(
              children: [
                Icon(icon, size: 19, color: selected ? _mint : _muted),
                const SizedBox(width: 11),
                Expanded(
                  child: Text(
                    label,
                    style: TextStyle(
                      color: selected ? Colors.white : Colors.white70,
                      fontSize: 13,
                      fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                    ),
                  ),
                ),
                if (badge != null)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 7,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: _panelRaised,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      badge,
                      style: const TextStyle(color: _muted, fontSize: 10),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildWorkspaceSidebar({bool drawer = false}) {
    return Container(
      width: drawer ? null : 238,
      color: const Color(0xFF0B111E),
      child: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 14, 18),
              child: Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(colors: [_mint, _violet]),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(
                      Icons.auto_awesome,
                      color: _ink,
                      size: 20,
                    ),
                  ),
                  const SizedBox(width: 11),
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'HomeMind',
                          style: TextStyle(
                            fontSize: 17,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                        Text(
                          'Personal workspace',
                          style: TextStyle(color: _muted, fontSize: 10),
                        ),
                      ],
                    ),
                  ),
                  if (drawer)
                    IconButton(
                      icon: const Icon(Icons.close, color: _muted, size: 20),
                      onPressed: () => Navigator.pop(context),
                    ),
                ],
              ),
            ),
            _workspaceNavItem(
              icon: Icons.home_rounded,
              label: 'Home',
              selected: _workspaceIndex == 0,
              onTap: () {
                if (drawer) {
                  Navigator.pop(context);
                } else {
                  setState(() => _workspaceIndex = 0);
                }
              },
            ),
            _workspaceNavItem(
              icon: Icons.auto_awesome,
              label: 'Assistant',
              selected: _workspaceIndex == 1,
              onTap:
                  () => _openWorkspace(1, AssistantScreen(apiBase: _apiBase)),
            ),
            _workspaceNavItem(
              icon: Icons.forum_outlined,
              label: 'Rooms',
              selected: _workspaceIndex == 2,
              onTap:
                  () => _openWorkspace(2, RoomsListScreen(apiBase: _apiBase)),
            ),
            _workspaceNavItem(
              icon: Icons.manage_search,
              label: 'Memory',
              selected: _workspaceIndex == 3,
              onTap:
                  () => _openWorkspace(
                    3,
                    MemoryTimelineScreen(apiBase: _apiBase),
                  ),
            ),
            _workspaceNavItem(
              icon: Icons.notifications_none_outlined,
              label: 'Notifications',
              badge: _unreadNotifications > 0 ? '$_unreadNotifications' : null,
              selected: _workspaceIndex == 4,
              onTap: _openNotifications,
            ),
            const Padding(
              padding: EdgeInsets.fromLTRB(21, 22, 20, 7),
              child: Text(
                'SYSTEM',
                style: TextStyle(
                  color: _muted,
                  fontSize: 9,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.4,
                ),
              ),
            ),
            _workspaceNavItem(
              icon:
                  _notificationsMuted
                      ? Icons.notifications_off_outlined
                      : Icons.record_voice_over_outlined,
              label: 'Initiative & alerts',
              badge: _notificationsMuted ? 'MUTED' : null,
              onTap: _showInitiativeSheet,
            ),
            _workspaceNavItem(
              icon: Icons.center_focus_strong,
              label: 'Capture & privacy',
              onTap: _showCaptureSheet,
            ),
            _workspaceNavItem(
              icon: Icons.tune_rounded,
              label: 'Settings',
              selected: _workspaceIndex == 5,
              onTap: () => _openWorkspace(5, _settingsScreen()),
            ),
            _workspaceNavItem(
              icon: Icons.hub_outlined,
              label: 'Home hub',
              onTap: _showConnectionSheet,
            ),
            const Spacer(),
            Container(
              margin: const EdgeInsets.all(12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _panel,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: _line),
              ),
              child: Row(
                children: [
                  Container(
                    width: 9,
                    height: 9,
                    decoration: BoxDecoration(
                      color:
                          _backendConnected ? _mint : const Color(0xFFFF718B),
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 9),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _backendConnected
                              ? 'Home hub online'
                              : 'Home hub offline',
                          style: const TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        Text(
                          _backendActivity,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(color: _muted, fontSize: 9),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(Size size) {
    return Row(
      children: [
        const Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Home',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                  letterSpacing: -.5,
                ),
              ),
              Text(
                'Your activity, conversation, and home context',
                style: TextStyle(color: _muted, fontSize: 11),
              ),
            ],
          ),
        ),
        _notificationButton(),
        const SizedBox(width: 8),
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
      builder:
          (sheetContext) => Padding(
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
                const Text(
                  'Home hub',
                  style: TextStyle(fontSize: 19, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 5),
                const Text(
                  'Connect this device to your local assistant.',
                  style: TextStyle(color: _muted, fontSize: 12),
                ),
                const SizedBox(height: 18),
                SizedBox(
                  width: double.infinity,
                  child: _bodyTextarea(MediaQuery.sizeOf(context)),
                ),
                const SizedBox(height: 14),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton(
                    onPressed: () {
                      Navigator.pop(sheetContext);
                      _connectToHomeHub();
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

  void _showInitiativeSheet() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: _panel,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      builder:
          (sheetContext) => StatefulBuilder(
            builder: (context, setSheetState) {
              void update(VoidCallback change) {
                setState(change);
                setSheetState(() {});
                _persistDeliveryPreferences();
              }

              Widget toggle({
                required IconData icon,
                required String title,
                required String subtitle,
                required bool value,
                required ValueChanged<bool>? onChanged,
              }) {
                return SwitchListTile.adaptive(
                  contentPadding: const EdgeInsets.symmetric(horizontal: 4),
                  secondary: Icon(
                    icon,
                    color:
                        onChanged == null
                            ? _muted.withValues(alpha: .45)
                            : _mint,
                  ),
                  title: Text(
                    title,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: onChanged == null ? _muted : Colors.white,
                    ),
                  ),
                  subtitle: Text(
                    subtitle,
                    style: const TextStyle(color: _muted, fontSize: 10.5),
                  ),
                  value: value,
                  activeTrackColor: _mint,
                  onChanged: onChanged,
                );
              }

              final proactiveControlsEnabled = _proactiveEnabled;
              final notificationControlsEnabled = !_notificationsMuted;
              return SafeArea(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(18, 10, 18, 24),
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
                      const SizedBox(height: 18),
                      const Text(
                        'Initiative & alerts',
                        style: TextStyle(
                          fontSize: 19,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                      const SizedBox(height: 5),
                      const Text(
                        'Choose how HomeMind reaches you. These are delivery controls; '
                        'the backend can continue observing and remembering.',
                        style: TextStyle(color: _muted, fontSize: 11.5),
                      ),
                      const SizedBox(height: 16),
                      Container(
                        decoration: BoxDecoration(
                          color: _panelRaised,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: _line),
                        ),
                        child: Column(
                          children: [
                            toggle(
                              icon: Icons.auto_awesome,
                              title: 'Proactive assistant',
                              subtitle:
                                  'Allow unprompted insights from screen, mobile, camera and memory context.',
                              value: _proactiveEnabled,
                              onChanged:
                                  (value) =>
                                      update(() => _proactiveEnabled = value),
                            ),
                            const Divider(height: 1, color: _line),
                            toggle(
                              icon: Icons.volume_up_outlined,
                              title: 'Speak insights',
                              subtitle:
                                  'Play proactive insights aloud using TTS.',
                              value: _proactiveVoiceEnabled,
                              onChanged:
                                  proactiveControlsEnabled
                                      ? (value) => update(
                                        () => _proactiveVoiceEnabled = value,
                                      )
                                      : null,
                            ),
                            toggle(
                              icon: Icons.chat_bubble_outline,
                              title: 'Show in conversation',
                              subtitle:
                                  'Add proactive insights to the Home conversation feed.',
                              value: _proactiveFeedEnabled,
                              onChanged:
                                  proactiveControlsEnabled
                                      ? (value) => update(
                                        () => _proactiveFeedEnabled = value,
                                      )
                                      : null,
                            ),
                            toggle(
                              icon: Icons.notification_add_outlined,
                              title: 'Notify proactive insights',
                              subtitle:
                                  'Deliver insights as Android system notifications, including in the background.',
                              value: _proactiveNotificationsEnabled,
                              onChanged:
                                  proactiveControlsEnabled &&
                                          notificationControlsEnabled
                                      ? (value) => update(
                                        () =>
                                            _proactiveNotificationsEnabled =
                                                value,
                                      )
                                      : null,
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 14),
                      const Padding(
                        padding: EdgeInsets.only(left: 4, bottom: 7),
                        child: Text(
                          'NOTIFICATIONS',
                          style: TextStyle(
                            color: _muted,
                            fontSize: 9,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.3,
                          ),
                        ),
                      ),
                      Container(
                        decoration: BoxDecoration(
                          color: _panelRaised,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: _line),
                        ),
                        child: Column(
                          children: [
                            toggle(
                              icon: Icons.notifications_active_outlined,
                              title: 'Home event alerts',
                              subtitle:
                                  'Notify critical and important safety, security and activity events.',
                              value: _eventNotificationsEnabled,
                              onChanged:
                                  notificationControlsEnabled
                                      ? (value) => update(
                                        () =>
                                            _eventNotificationsEnabled = value,
                                      )
                                      : null,
                            ),
                            const Divider(height: 1, color: _line),
                            toggle(
                              icon: Icons.notifications_off_outlined,
                              title: 'Mute all notifications',
                              subtitle:
                                  'Stops system notifications. Voice and the in-app feed remain available.',
                              value: _notificationsMuted,
                              onChanged:
                                  (value) =>
                                      update(() => _notificationsMuted = value),
                            ),
                          ],
                        ),
                      ),
                      if (!_notificationController.isSupported) ...[
                        const SizedBox(height: 12),
                        const Text(
                          'System notification delivery is currently available on Android. '
                          'Voice and the in-app feed work on other platforms.',
                          style: TextStyle(color: _muted, fontSize: 10.5),
                        ),
                      ],
                    ],
                  ),
                ),
              );
            },
          ),
    );
  }

  Widget _buildMobileHeader() {
    return Row(
      children: [
        IconButton(
          tooltip: 'Open workspace',
          onPressed: () => _scaffoldKey.currentState?.openDrawer(),
          style: IconButton.styleFrom(backgroundColor: _panel),
          icon: const Icon(Icons.menu_rounded, size: 21),
        ),
        const SizedBox(width: 8),
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
              Text(
                'HomeMind',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800),
              ),
              Text(
                'Your home, in sync',
                style: TextStyle(color: _muted, fontSize: 10),
              ),
            ],
          ),
        ),
        _notificationButton(),
        const SizedBox(width: 5),
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
          Icon(
            Icons.circle,
            size: 8,
            color: ready ? _mint : const Color(0xFFFF718B),
          ),
          const SizedBox(width: 8),
          Text(
            ready ? 'Home hub online' : 'Home hub offline',
            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600),
          ),
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
      builder:
          (sheetContext) => SafeArea(
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
                  _buildCaptureSourcesPanel(),
                  const SizedBox(height: 12),
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
          BoxShadow(
            color: Color(0x33000000),
            blurRadius: 30,
            offset: Offset(0, 14),
          ),
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
                const Text(
                  'Conversation',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
                ),
                const Spacer(),
                IconButton(
                  tooltip: 'Clear conversation',
                  onPressed: _clearChatHistory,
                  icon: const Icon(
                    Icons.delete_sweep_outlined,
                    color: _muted,
                    size: 20,
                  ),
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
                    Text(
                      'Your home is listening',
                      style: TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    SizedBox(height: 6),
                    Text(
                      'Hold the microphone, or type a message',
                      style: TextStyle(color: _muted, fontSize: 12),
                    ),
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
            child: Column(
              children: [
                _buildTypedComposer(),
                const SizedBox(height: 12),
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _isRecording ? 'Listening…' : 'Hold to speak',
                            style: const TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          Text(
                            _isProcessing
                                ? 'HomeMind is thinking'
                                : 'Release when you are finished',
                            style: const TextStyle(color: _muted, fontSize: 10),
                          ),
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
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildContextSelector() {
    const icons = [
      Icons.mic_none,
      Icons.monitor_outlined,
      Icons.camera_alt_outlined,
    ];
    return Container(
      padding: const EdgeInsets.all(5),
      decoration: BoxDecoration(
        color: _ink,
        borderRadius: BorderRadius.circular(15),
      ),
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
                  _conversationThinking = index == 1;
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
                    Icon(
                      icons[index],
                      size: 18,
                      color: selected ? _mint : _muted,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _contextOptions[index].toUpperCase(),
                      style: TextStyle(
                        color: selected ? Colors.white : _muted,
                        fontSize: 9,
                        fontWeight: FontWeight.w700,
                        letterSpacing: .6,
                      ),
                    ),
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
          const Text(
            'PERCEPTION MODE',
            style: TextStyle(
              color: _muted,
              fontSize: 10,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.4,
            ),
          ),
          const SizedBox(height: 10),
          _buildContextSelector(),
          const SizedBox(height: 16),
          _buildCaptureSourcesPanel(),
          const SizedBox(height: 12),
          _buildCapturePanel(),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _buildToggleSwitch(
                'Conversation',
                _isTalking,
                (v) => setState(() => _isTalking = v),
              ),
              _buildToggleSwitch('Live', _isLive, (v) {
                if (v && _currentContext == 'talker') {
                  _showSnack('Choose Screen or Camera before enabling Live');
                  return;
                }
                setState(() => _isLive = v);
              }),
              _buildToggleSwitch(
                'Memory',
                _useMemory,
                (v) => setState(() => _useMemory = v),
              ),
              _buildToggleSwitch(
                'Thinking',
                _conversationThinking,
                (v) => setState(() => _conversationThinking = v),
              ),
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

  Widget _buildQuickAccessPanel() {
    Widget action({
      required IconData icon,
      required String title,
      required String subtitle,
      required Color color,
      required VoidCallback onTap,
    }) {
      return Material(
        color: _panelRaised.withValues(alpha: .72),
        borderRadius: BorderRadius.circular(14),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(14),
          child: Padding(
            padding: const EdgeInsets.all(13),
            child: Row(
              children: [
                Container(
                  width: 38,
                  height: 38,
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: .13),
                    borderRadius: BorderRadius.circular(11),
                  ),
                  child: Icon(icon, color: color, size: 19),
                ),
                const SizedBox(width: 11),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        subtitle,
                        style: const TextStyle(color: _muted, fontSize: 9.5),
                      ),
                    ],
                  ),
                ),
                const Icon(Icons.chevron_right, color: _muted, size: 18),
              ],
            ),
          ),
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _panel,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _line),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'QUICK ACCESS',
            style: TextStyle(
              color: _muted,
              fontSize: 9,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.3,
            ),
          ),
          const SizedBox(height: 10),
          action(
            icon: Icons.auto_awesome,
            title: 'Grounded assistant',
            subtitle: 'Ask with citations from memory',
            color: _mint,
            onTap: () => _openWorkspace(1, AssistantScreen(apiBase: _apiBase)),
          ),
          const SizedBox(height: 7),
          action(
            icon: Icons.forum_outlined,
            title: 'Rooms',
            subtitle: 'Continue a project or topic',
            color: _violet,
            onTap: () => _openWorkspace(2, RoomsListScreen(apiBase: _apiBase)),
          ),
          const SizedBox(height: 7),
          action(
            icon: Icons.manage_search,
            title: 'Explore memory',
            subtitle: 'Search timeline and entities',
            color: const Color(0xFF62B5FF),
            onTap:
                () =>
                    _openWorkspace(3, MemoryTimelineScreen(apiBase: _apiBase)),
          ),
          const SizedBox(height: 7),
          action(
            icon: _reflecting ? Icons.hourglass_top : Icons.psychology_outlined,
            title: 'Reflect now',
            subtitle:
                _reflecting
                    ? 'Reading the last ${_reflectFrames}s…'
                    : '${_reflectShortcut?.label ?? 'Shortcut disabled'} • '
                        'last ${_reflectFrames}s of frames',
            color: const Color(0xFFFFC857),
            onTap: _reflecting ? () {} : _reflectOnScreen,
          ),
          const SizedBox(height: 7),
          action(
            icon: Icons.add_to_photos_outlined,
            title: 'Reflect from source…',
            subtitle:
                '${_sourceReflectShortcut?.label ?? 'Shortcut disabled'} • '
                'screen, mobile or camera',
            color: const Color(0xFF62B5FF),
            onTap: _reflecting ? () {} : _showReflectionSourcePicker,
          ),
          const SizedBox(height: 7),
          action(
            icon: Icons.auto_awesome_outlined,
            title: 'Guided reflection…',
            subtitle: '9 reading, code and guidance prompts',
            color: _mint,
            onTap: _reflecting ? () {} : _showPromptActionPicker,
          ),
        ],
      ),
    );
  }

  Widget _buildMobileControls() {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
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
                child: _buildToggleSwitch(
                  'Talk',
                  _isTalking,
                  (v) => setState(() => _isTalking = v),
                ),
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
                child: _buildToggleSwitch(
                  'Memory',
                  _useMemory,
                  (v) => setState(() => _useMemory = v),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          _buildToggleSwitch(
            'Thinking',
            _conversationThinking,
            (v) => setState(() => _conversationThinking = v),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: FilledButton.tonalIcon(
                  onPressed: _reflecting ? null : _reflectOnScreen,
                  icon: Icon(
                    _reflecting
                        ? Icons.hourglass_top
                        : Icons.psychology_outlined,
                    size: 17,
                  ),
                  label: Text(_reflecting ? 'Working…' : 'Reflect'),
                ),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _reflecting ? null : _showReflectionSourcePicker,
                  icon: const Icon(Icons.add_to_photos_outlined, size: 17),
                  label: const Text('Source'),
                ),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: OutlinedButton.icon(
                  key: const Key('mobile-guided-reflection-button'),
                  onPressed: _reflecting ? null : _showPromptActionPicker,
                  icon: const Icon(Icons.auto_awesome_outlined, size: 17),
                  label: const Text('Prompts'),
                ),
              ),
            ],
          ),
          ListTile(
            dense: true,
            contentPadding: const EdgeInsets.fromLTRB(4, 4, 4, 0),
            onTap: _showCaptureSheet,
            leading: const Icon(
              Icons.center_focus_strong,
              color: _muted,
              size: 19,
            ),
            title: const Text(
              'Capture & privacy',
              style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
            ),
            subtitle: const Text(
              'Camera, screen and stored memory',
              style: TextStyle(color: _muted, fontSize: 10),
            ),
            trailing: const Icon(
              Icons.chevron_right_rounded,
              color: _muted,
              size: 20,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final Size s = MediaQuery.of(context).size;
    final shortcutBindings = <ShortcutActivator, VoidCallback>{};
    final reflectShortcut = _reflectShortcut;
    if (reflectShortcut != null) {
      shortcutBindings[reflectShortcut.activator] =
          () => _reflectOnScreen(thinking: _thinkingForShortcut('reflect_now'));
    }
    final sourceReflectShortcut = _sourceReflectShortcut;
    if (sourceReflectShortcut != null) {
      shortcutBindings[sourceReflectShortcut.activator] =
          () => _showReflectionSourcePicker(
            thinking: _thinkingForShortcut('reflect_from_source'),
          );
    }
    final clipboardAnswerShortcut = _clipboardAnswerShortcut;
    if (clipboardAnswerShortcut != null) {
      shortcutBindings[clipboardAnswerShortcut.activator] = _answerClipboard;
    }
    for (final preset in reflectionPromptPresets) {
      final shortcut = _promptShortcuts[preset.id];
      if (shortcut != null) {
        shortcutBindings[shortcut.activator] = () => _runPromptPreset(preset);
      }
    }
    // The configured combination works anywhere in the focused app, including
    // inside a text field: the key event bubbles up to this binding.
    return CallbackShortcuts(
      bindings: shortcutBindings,
      child: Focus(autofocus: true, child: _buildShell(s)),
    );
  }

  Widget _buildShell(Size s) {
    return Scaffold(
      key: _scaffoldKey,
      drawer: SizedBox(
        width: 286,
        child: Drawer(
          backgroundColor: const Color(0xFF0B111E),
          shape: const RoundedRectangleBorder(),
          child: _buildWorkspaceSidebar(drawer: true),
        ),
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          final desktop = constraints.maxWidth >= 960;
          final home = Container(
            decoration: const BoxDecoration(
              gradient: RadialGradient(
                center: Alignment(-.8, -1),
                radius: 1.25,
                colors: [Color(0xFF132235), _ink],
              ),
            ),
            child: SafeArea(
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 1380),
                  child: Padding(
                    padding: EdgeInsets.fromLTRB(
                      desktop ? 22 : 14,
                      14,
                      desktop ? 22 : 14,
                      14,
                    ),
                    child:
                        desktop
                            ? Column(
                              children: [
                                _buildHeader(s),
                                const SizedBox(height: 14),
                                Align(
                                  alignment: Alignment.centerLeft,
                                  child: _buildBackendIndicators(),
                                ),
                                const SizedBox(height: 14),
                                Expanded(
                                  child: Row(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.stretch,
                                    children: [
                                      Expanded(
                                        flex: 5,
                                        child: _buildConversationCard(),
                                      ),
                                      const SizedBox(width: 14),
                                      SizedBox(
                                        width: 348,
                                        child: SingleChildScrollView(
                                          child: Column(
                                            children: [
                                              _buildQuickAccessPanel(),
                                              const SizedBox(height: 12),
                                              _buildControlPanel(),
                                            ],
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            )
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
              ),
            ),
          );
          if (!desktop) return home;
          final Widget workspace;
          switch (_workspaceIndex) {
            case 1:
              workspace = AssistantScreen(apiBase: _apiBase);
              break;
            case 2:
              workspace = RoomsListScreen(apiBase: _apiBase);
              break;
            case 3:
              workspace = MemoryTimelineScreen(apiBase: _apiBase);
              break;
            case 4:
              workspace = NotificationsScreen(
                apiBase: _apiBase,
                onUnreadChanged: (count) {
                  if (mounted) {
                    setState(() => _unreadNotifications = count);
                  }
                },
              );
              break;
            case 5:
              workspace = _settingsScreen();
              break;
            default:
              workspace = home;
          }
          return Row(
            children: [
              _buildWorkspaceSidebar(),
              const VerticalDivider(width: 1, thickness: 1, color: _line),
              Expanded(child: workspace),
            ],
          );
        },
      ),
    );
  }

  // Helper method to reduce code duplication for switches
  Widget _buildToggleSwitch(
    String title,
    bool value,
    ValueChanged<bool> onChanged,
  ) {
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
  final String apiBase;

  const MessageBubble({
    Key? key,
    required this.message,
    required this.audioPlayer,
    this.apiBase = '',
  }) : super(key: key);

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
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Flexible(
                  child: Text(
                    message.text,
                    style: const TextStyle(
                      fontSize: 14,
                      color: Color(0xFFE9EEF7),
                      height: 1.45,
                    ),
                  ),
                ),
                if (!isUserMessage &&
                    message.fullAudio != null &&
                    message.fullAudio!.isNotEmpty)
                  _buildReplayButton(),
              ],
            ),
            if (message.clipId != null && apiBase.isNotEmpty)
              _buildClipButton(context, accent),
          ],
        ),
      ),
    );
  }

  /// Opens the footage behind an unprompted remark, and the box for asking
  /// further questions about it.
  Widget _buildClipButton(BuildContext context, Color accent) {
    return TextButton.icon(
      style: TextButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
        minimumSize: Size.zero,
        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
      ),
      onPressed:
          () => showClipSheet(
            context,
            apiBase: apiBase,
            clipId: message.clipId!,
            caption: message.text,
            coversSeconds: message.clipCoversSeconds,
            playsSeconds: message.clipPlaysSeconds,
          ),
      icon: Icon(Icons.play_circle_outline, size: 15, color: accent),
      label: Text(
        'Watch what I saw',
        style: TextStyle(
          fontSize: 11,
          color: accent,
          fontWeight: FontWeight.w700,
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

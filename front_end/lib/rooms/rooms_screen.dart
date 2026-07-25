import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../voice/dictation_controller.dart';

const _ink = Color(0xFF070B14);
const _panel = Color(0xFF111827);
const _panelRaised = Color(0xFF182235);
const _line = Color(0xFF263246);
const _mint = Color(0xFF6EE7D8);
const _violet = Color(0xFF8B7CF6);
const _muted = Color(0xFF91A0B8);

Color _roomColor(dynamic value, Color fallback) {
  final raw = value?.toString().replaceFirst('#', '');
  final parsed = raw == null ? null : int.tryParse(raw, radix: 16);
  return parsed == null ? fallback : Color(0xFF000000 | parsed);
}

IconData _kindIcon(String kind, String name, [String? configured]) {
  switch (configured) {
    case 'work':
      return Icons.work_outline;
    case 'school':
      return Icons.school_outlined;
    case 'fitness':
      return Icons.fitness_center;
    case 'home':
      return Icons.home_outlined;
    case 'book':
      return Icons.menu_book;
    case 'code':
      return Icons.code;
    case 'videocam':
      return Icons.videocam;
    case 'desktop_windows':
      return Icons.desktop_windows;
  }
  switch (kind) {
    case 'daily':
      return Icons.calendar_today;
    case 'camera':
      return Icons.videocam;
    case 'screen':
      return Icons.desktop_windows;
    case 'project':
      return Icons.folder_special;
    case 'topic':
      return Icons.forum;
    default: // activity
      final n = name.toLowerCase();
      if (n.contains('cod')) return Icons.code;
      if (n.contains('read')) return Icons.menu_book;
      if (n.contains('watch')) return Icons.play_circle_outline;
      if (n.contains('brows')) return Icons.public;
      if (n.contains('commun')) return Icons.chat_bubble_outline;
      if (n.contains('term')) return Icons.terminal;
      return Icons.desktop_windows;
  }
}

String _hm(dynamic ts) {
  if (ts is! num) return '';
  final dt = DateTime.fromMillisecondsSinceEpoch((ts * 1000).round());
  return '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
}

/// ---- Rooms list (channels) ----------------------------------------------
class RoomsListScreen extends StatefulWidget {
  final String apiBase;
  const RoomsListScreen({super.key, required this.apiBase});

  @override
  State<RoomsListScreen> createState() => _RoomsListScreenState();
}

class _RoomsListScreenState extends State<RoomsListScreen> {
  bool _loading = false;
  bool _showArchived = false;
  String? _error;
  List<dynamic> _rooms = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final resp = await http
          .get(Uri.parse(
              '${widget.apiBase}/rooms?include_archived=$_showArchived'))
          .timeout(const Duration(seconds: 15));
      if (resp.statusCode == 200) {
        final data = json.decode(resp.body) as Map<String, dynamic>;
        setState(() => _rooms = (data['rooms'] as List?) ?? []);
      } else {
        setState(() => _error = 'HTTP ${resp.statusCode}');
      }
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _open(Map<String, dynamic> room) {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => RoomScreen(
        apiBase: widget.apiBase,
        roomId: room['room_id'].toString(),
        roomName: (room['name'] ?? room['room_id']).toString(),
        kind: (room['kind'] ?? 'activity').toString(),
      ),
    )).then((_) => _load());
  }

  List<String> _csv(String value) => value
      .split(',')
      .map((v) => v.trim())
      .where((v) => v.isNotEmpty)
      .toList();

  Future<void> _editRoom([Map<String, dynamic>? summary]) async {
    Map<String, dynamic>? room = summary;
    if (summary != null) {
      try {
        final resp = await http.get(
            Uri.parse('${widget.apiBase}/rooms/${summary['room_id']}'));
        if (resp.statusCode == 200) {
          room = (json.decode(resp.body) as Map<String, dynamic>)['room']
              as Map<String, dynamic>;
        }
      } catch (_) {}
    }
    if (!mounted) return;
    final matcher = (room?['matcher'] as Map<String, dynamic>?) ?? {};
    final name = TextEditingController(text: room?['name']?.toString() ?? '');
    final description =
        TextEditingController(text: room?['description']?.toString() ?? '');
    final instructions =
        TextEditingController(text: room?['instructions']?.toString() ?? '');
    final color =
        TextEditingController(text: room?['color']?.toString() ?? '#8B7CF6');
    final activities = TextEditingController(
        text: ((matcher['activity_types'] as List?) ?? []).join(', '));
    final apps =
        TextEditingController(text: ((matcher['apps'] as List?) ?? []).join(', '));
    final keywords = TextEditingController(
        text: ((matcher['title_keywords'] as List?) ?? []).join(', '));
    final projects = TextEditingController(
        text: ((matcher['project_ids'] as List?) ?? []).join(', '));
    final entities = TextEditingController(
        text: ((matcher['entity_types'] as List?) ?? []).join(', '));
    final isNew = room == null;
    final result = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: _panel,
        title: Text(isNew ? 'Create room' : 'Edit room',
            style: const TextStyle(color: Colors.white)),
        content: SizedBox(
          width: 480,
          child: SingleChildScrollView(
            child: Column(
              children: [
                _dialogField(name, 'Name'),
                _dialogField(description, 'Description', lines: 2),
                _dialogField(instructions, 'Assistant instructions', lines: 3),
                _dialogField(color, 'Color (hex)'),
                const Padding(
                  padding: EdgeInsets.only(top: 10, bottom: 6),
                  child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text('Automatic matching (comma-separated)',
                        style: TextStyle(color: _mint, fontWeight: FontWeight.w700)),
                  ),
                ),
                _dialogField(activities, 'Activities'),
                _dialogField(apps, 'Applications'),
                _dialogField(keywords, 'Title / summary keywords'),
                _dialogField(projects, 'Project IDs'),
                _dialogField(entities, 'Entity types'),
              ],
            ),
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Save')),
        ],
      ),
    );
    if (result != true || name.text.trim().isEmpty) return;
    final payload = {
      'name': name.text.trim(),
      'description': description.text.trim(),
      'instructions': instructions.text.trim(),
      'color': color.text.trim(),
      'matcher': {
        'activity_types': _csv(activities.text),
        'apps': _csv(apps.text),
        'title_keywords': _csv(keywords.text),
        'project_ids': _csv(projects.text),
        'entity_types': _csv(entities.text),
      },
    };
    try {
      final uri = isNew
          ? Uri.parse('${widget.apiBase}/rooms')
          : Uri.parse('${widget.apiBase}/rooms/${room!['room_id']}');
      final resp = isNew
          ? await http.post(uri,
              headers: {'Content-Type': 'application/json'},
              body: json.encode(payload))
          : await http.patch(uri,
              headers: {'Content-Type': 'application/json'},
              body: json.encode(payload));
      if (resp.statusCode == 200 || resp.statusCode == 201) {
        await _load();
      } else {
        _snack('Could not save room: ${_errorText(resp)}');
      }
    } catch (e) {
      _snack('Could not save room: $e');
    }
  }

  Widget _dialogField(TextEditingController controller, String label,
      {int lines = 1}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 9),
      child: TextField(
        controller: controller,
        minLines: lines,
        maxLines: lines,
        style: const TextStyle(color: Colors.white),
        decoration: InputDecoration(labelText: label),
      ),
    );
  }

  String _errorText(http.Response resp) {
    try {
      return (json.decode(resp.body) as Map<String, dynamic>)['error']
              ?.toString() ??
          'HTTP ${resp.statusCode}';
    } catch (_) {
      return 'HTTP ${resp.statusCode}';
    }
  }

  void _snack(String text) {
    if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(text)));
  }

  Future<void> _roomAction(String action, Map<String, dynamic> room) async {
    final id = room['room_id'].toString();
    try {
      if (action == 'edit') return _editRoom(room);
      if (action == 'archive') {
        await http.patch(Uri.parse('${widget.apiBase}/rooms/$id'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'archived': !(room['archived'] == true)}));
      } else if (action == 'pin') {
        await http.patch(Uri.parse('${widget.apiBase}/rooms/$id'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'pinned': !(room['pinned'] == true)}));
      } else if (action == 'reroute') {
        final resp =
            await http.post(Uri.parse('${widget.apiBase}/rooms/$id/reroute'));
        if (resp.statusCode != 200) _snack(_errorText(resp));
      } else if (action == 'delete') {
        final confirmed = await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            title: Text('Delete ${room['name']}?'),
            content: const Text(
                'Its notes and room chat will be deleted. Captured events remain available.'),
            actions: [
              TextButton(
                  onPressed: () => Navigator.pop(context, false),
                  child: const Text('Cancel')),
              FilledButton(
                  onPressed: () => Navigator.pop(context, true),
                  child: const Text('Delete')),
            ],
          ),
        );
        if (confirmed != true) return;
        final resp =
            await http.delete(Uri.parse('${widget.apiBase}/rooms/$id'));
        if (resp.statusCode != 200) _snack(_errorText(resp));
      }
      await _load();
    } catch (e) {
      _snack('Room action failed: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    // Daily first, then by event count (backend already sorts by count).
    final rooms = [..._rooms]
      ..sort((a, b) {
        if (a['kind'] == 'daily') return -1;
        if (b['kind'] == 'daily') return 1;
        if (a['pinned'] == true && b['pinned'] != true) return -1;
        if (b['pinned'] == true && a['pinned'] != true) return 1;
        final position = ((a['position'] ?? 0) as num)
            .compareTo((b['position'] ?? 0) as num);
        if (position != 0) return position;
        return ((b['events'] ?? 0) as num).compareTo((a['events'] ?? 0) as num);
      });
    return Scaffold(
      backgroundColor: _ink,
      appBar: AppBar(
        backgroundColor: _panel,
        title: const Text('Rooms'),
        actions: [
          IconButton(
            tooltip: _showArchived ? 'Hide archived' : 'Show archived',
            icon: Icon(_showArchived ? Icons.inventory_2 : Icons.inventory_2_outlined,
                color: _muted),
            onPressed: () {
              setState(() => _showArchived = !_showArchived);
              _load();
            },
          ),
          IconButton(
            icon: const Icon(Icons.refresh, color: _mint),
            onPressed: _loading ? null : _load,
          ),
        ],
      ),
      body: Column(
        children: [
          if (_loading) const LinearProgressIndicator(minHeight: 2, color: _mint),
          Expanded(
            child: _error != null
                ? Center(
                    child: Text('Could not load rooms.\n$_error',
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: _muted)))
                : ListView.builder(
                    padding: const EdgeInsets.all(12),
                    itemCount: rooms.length,
                    itemBuilder: (context, i) => _roomTile(rooms[i] as Map<String, dynamic>),
                  ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: _mint,
        foregroundColor: _ink,
        onPressed: () => _editRoom(),
        icon: const Icon(Icons.add),
        label: const Text('New room'),
      ),
    );
  }

  Widget _roomTile(Map<String, dynamic> room) {
    final name = (room['name'] ?? room['room_id']).toString();
    final kind = (room['kind'] ?? 'activity').toString();
    final events = (room['events'] ?? 0);
    final accent = _roomColor(room['color'], kind == 'daily' ? _mint : _violet);
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: _panelRaised.withOpacity(.55),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: kind == 'daily' ? _mint.withOpacity(.4) : _line),
      ),
      child: ListTile(
        leading:
            Icon(_kindIcon(kind, name, room['icon']?.toString()), color: accent),
        title: Text(name,
            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700)),
        subtitle: Text('$kind · $events events',
            style: const TextStyle(color: _muted, fontSize: 12)),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (room['pinned'] == true)
              const Icon(Icons.push_pin, size: 16, color: _mint),
            PopupMenuButton<String>(
              icon: const Icon(Icons.more_vert, color: _muted),
              onSelected: (value) => _roomAction(value, room),
              itemBuilder: (_) => [
                const PopupMenuItem(value: 'edit', child: Text('Edit')),
                PopupMenuItem(
                    value: 'pin',
                    child: Text(room['pinned'] == true ? 'Unpin' : 'Pin')),
                if (kind != 'daily')
                  PopupMenuItem(
                      value: 'archive',
                      child: Text(room['archived'] == true ? 'Restore' : 'Archive')),
                const PopupMenuItem(
                    value: 'reroute', child: Text('Re-route events')),
                if (kind != 'daily')
                  const PopupMenuItem(value: 'delete', child: Text('Delete')),
              ],
            ),
          ],
        ),
        onTap: () => _open(room),
      ),
    );
  }
}

/// ---- Room screen (feed + compose) ---------------------------------------
class RoomScreen extends StatefulWidget {
  final String apiBase;
  final String roomId;
  final String roomName;
  final String kind;
  const RoomScreen({
    super.key,
    required this.apiBase,
    required this.roomId,
    required this.roomName,
    required this.kind,
  });

  @override
  State<RoomScreen> createState() => _RoomScreenState();
}

class _RoomScreenState extends State<RoomScreen> {
  final TextEditingController _input = TextEditingController();
  final TextEditingController _search = TextEditingController();
  final ScrollController _scroll = ScrollController();
  // Speaking into the composer: the transcript lands in _input so it can be
  // corrected, and filed as either a note or a question like anything typed.
  final DictationController _dictation = DictationController();
  bool _listening = false;
  bool _transcribing = false;
  bool _loading = false;
  bool _sending = false;
  String? _error;
  String _mode = 'note'; // 'note' | 'chat'
  String? _date;
  String _eventView = 'useful'; // useful | all | high | low | flagged
  final Set<String> _kinds = {'event', 'note', 'message'};
  List<dynamic> _feed = []; // chronological (oldest -> newest)

  bool get _isDaily => widget.kind == 'daily';
  bool get _isCameraRoom => widget.kind == 'camera';

  /// The app or camera an event came from, as it should read on the bubble.
  ///
  /// Legacy camera events stored their id ('camera:192-168-1-4') as the
  /// application; newer ones store the camera's name. Strip the prefix so both
  /// read the same, and drop the '.exe' noise from Windows process names.
  String _sourceTag(Map<String, dynamic> event) {
    var value = (event['application'] ?? '').toString().trim();
    if (value.isEmpty) return '';
    if (value.startsWith('camera:')) value = value.substring(7);
    if (value.toLowerCase().endsWith('.exe')) {
      value = value.substring(0, value.length - 4);
    }
    return value;
  }

  List<Map<String, dynamic>> get _visibleFeed {
    return _feed.cast<Map<String, dynamic>>().where((item) {
      if ((item['kind'] ?? 'event') != 'event') {
        return _eventView == 'useful' || _eventView == 'all';
      }
      final priority = (item['priority'] ?? 'normal').toString();
      final flagged = item['flagged'] == true;
      switch (_eventView) {
        case 'useful':
          return !flagged && priority != 'low';
        case 'high':
          return !flagged && priority == 'high';
        case 'low':
          return !flagged && priority == 'low';
        case 'flagged':
          return flagged;
        default:
          return true;
      }
    }).toList();
  }

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _input.dispose();
    _search.dispose();
    _scroll.dispose();
    _dictation.dispose();
    super.dispose();
  }

  Future<void> _toggleDictation() async {
    if (_transcribing) return;
    if (!_listening) {
      if (!await _dictation.start()) {
        _snack('Microphone unavailable — check permissions');
        return;
      }
      if (mounted) setState(() => _listening = true);
      return;
    }

    setState(() {
      _listening = false;
      _transcribing = true;
    });
    try {
      final text = await _dictation.stopAndTranscribe(widget.apiBase);
      if (text.isEmpty) {
        _snack('Nothing was heard — try again');
        return;
      }
      // Append rather than replace: dictation can add to a half-typed thought.
      final existing = _input.text.trimRight();
      _input.text = existing.isEmpty ? text : '$existing $text';
      _input.selection =
          TextSelection.collapsed(offset: _input.text.length);
    } catch (e) {
      _snack('Could not transcribe: $e');
    } finally {
      if (mounted) setState(() => _transcribing = false);
    }
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final params = <String, String>{
        'limit': '300',
        if (_date != null) 'date': _date!,
        if (_search.text.trim().isNotEmpty) 'q': _search.text.trim(),
        if (_kinds.length < 3) 'kinds': _kinds.join(','),
      };
      final uri = Uri.parse('${widget.apiBase}/rooms/${widget.roomId}/feed')
          .replace(queryParameters: params);
      final resp = await http
          .get(uri)
          .timeout(const Duration(seconds: 15));
      if (resp.statusCode == 200) {
        final data = json.decode(resp.body) as Map<String, dynamic>;
        final feed = (data['feed'] as List?) ?? [];
        setState(() => _feed = feed.reversed.toList()); // newest at bottom
        _jumpToBottom();
      } else {
        setState(() => _error = 'HTTP ${resp.statusCode}');
      }
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _jumpToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.jumpTo(_scroll.position.maxScrollExtent);
      }
    });
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    if (text.isEmpty || _sending) return;
    final isChat = _mode == 'chat';
    final now = DateTime.now().millisecondsSinceEpoch / 1000.0;
    setState(() {
      _sending = true;
      // Optimistically echo the outgoing item so the feed isn't blank while
      // we wait for the round-trip (chat replies can take up to 60s).
      _feed.add({
        'kind': isChat ? 'message' : 'note',
        'role': 'user',
        'text': text,
        'ts': now,
        '_pending': true,
      });
      // For chat, also show a "thinking" placeholder immediately — the model
      // is a single-sequence vLLM shared with screen capture, so the reply can
      // take a while to come back. This gives instant acknowledgement.
      if (isChat) {
        _feed.add({
          'kind': 'message',
          'role': 'assistant',
          'text': '',
          'ts': now + 0.001,
          '_pending': true,
          '_thinking': true,
        });
      }
    });
    _input.clear();
    _jumpToBottom();
    try {
      final path = isChat ? 'chat' : 'note';
      final key = isChat ? 'message' : 'text';
      final resp = await http
          .post(
            Uri.parse('${widget.apiBase}/rooms/${widget.roomId}/$path'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({key: text}),
          )
          .timeout(const Duration(seconds: 60));
      if (resp.statusCode == 200) {
        await _load(); // reconciles the optimistic item with the server truth
      } else {
        _dropPending();
        _input.text = text; // restore so the user doesn't lose their input
        _snack('Failed: HTTP ${resp.statusCode}');
      }
    } catch (e) {
      _dropPending();
      _input.text = text;
      _snack('Error: $e');
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  void _dropPending() {
    if (mounted) {
      setState(() => _feed.removeWhere((it) => it is Map && it['_pending'] == true));
    }
  }

  Future<void> _generateReport() async {
    setState(() => _sending = true);
    try {
      final resp = await http
          .post(Uri.parse('${widget.apiBase}/rooms/daily/report'))
          .timeout(const Duration(seconds: 90));
      if (resp.statusCode == 200) {
        final data = json.decode(resp.body) as Map<String, dynamic>;
        if (data['posted'] == true) {
          await _load();
        } else {
          _snack('No activity logged today yet — nothing to report.');
        }
      } else {
        _snack('Report failed: HTTP ${resp.statusCode}');
      }
    } catch (e) {
      _snack('Error: $e');
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  void _snack(String m) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(m)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final visibleFeed = _visibleFeed;
    return Scaffold(
      backgroundColor: _ink,
      appBar: AppBar(
        backgroundColor: _panel,
        title: Text(widget.roomName, overflow: TextOverflow.ellipsis),
        actions: [
          if (_isDaily)
            IconButton(
              tooltip: 'Generate Coach report',
              icon: const Icon(Icons.insights, color: _mint),
              onPressed: _sending ? null : _generateReport,
            ),
          IconButton(
            icon: const Icon(Icons.refresh, color: _mint),
            onPressed: _loading ? null : _load,
          ),
        ],
      ),
      body: Column(
        children: [
          if (_loading || _sending)
            const LinearProgressIndicator(minHeight: 2, color: _mint),
          _feedFilters(),
          if (_kinds.contains('event')) _activityOverview(),
          Expanded(
            child: _error != null
                ? Center(
                    child: Text('Could not load feed.\n$_error',
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: _muted)))
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.all(12),
                    itemCount: visibleFeed.length,
                    itemBuilder: (context, i) => _feedItem(visibleFeed[i]),
                  ),
          ),
          _composer(),
        ],
      ),
    );
  }

  Widget _feedFilters() {
    return Container(
      color: _panel,
      padding: const EdgeInsets.fromLTRB(10, 6, 10, 8),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _search,
                  style: const TextStyle(color: Colors.white, fontSize: 13),
                  textInputAction: TextInputAction.search,
                  onSubmitted: (_) => _load(),
                  decoration: InputDecoration(
                    isDense: true,
                    prefixIcon: const Icon(Icons.search, size: 18),
                    hintText: 'Search this room',
                    suffixIcon: _search.text.isEmpty
                        ? null
                        : IconButton(
                            icon: const Icon(Icons.clear, size: 17),
                            onPressed: () {
                              _search.clear();
                              _load();
                            }),
                  ),
                ),
              ),
              const SizedBox(width: 7),
              IconButton(
                tooltip: 'Filter by date',
                icon: Icon(_date == null ? Icons.event : Icons.event_available,
                    color: _date == null ? _muted : _mint),
                onPressed: _pickDate,
              ),
            ],
          ),
          const SizedBox(height: 6),
          Row(
            children: [
              _filterChip('Activity', 'event'),
              const SizedBox(width: 6),
              _filterChip('Notes', 'note'),
              const SizedBox(width: 6),
              _filterChip('Chat', 'message'),
              if (_date != null) ...[
                const Spacer(),
                Text(_date!, style: const TextStyle(color: _mint, fontSize: 12)),
                IconButton(
                    visualDensity: VisualDensity.compact,
                    icon: const Icon(Icons.close, size: 16, color: _muted),
                    onPressed: () {
                      setState(() => _date = null);
                      _load();
                    }),
              ],
            ],
          ),
          if (_kinds.contains('event')) ...[
            const SizedBox(height: 6),
            SizedBox(
              height: 34,
              child: ListView(
                scrollDirection: Axis.horizontal,
                children: [
                  _eventViewChip('Useful', 'useful', Icons.auto_awesome),
                  const SizedBox(width: 6),
                  _eventViewChip('All', 'all', Icons.view_list_outlined),
                  const SizedBox(width: 6),
                  _eventViewChip('Important', 'high', Icons.star_outline),
                  const SizedBox(width: 6),
                  _eventViewChip('Low priority', 'low', Icons.low_priority),
                  const SizedBox(width: 6),
                  _eventViewChip('Flagged', 'flagged', Icons.flag_outlined),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  int _eventCountFor(String view) {
    return _feed.where((raw) {
      final item = raw as Map<String, dynamic>;
      if ((item['kind'] ?? 'event') != 'event') return false;
      final priority = (item['priority'] ?? 'normal').toString();
      final flagged = item['flagged'] == true;
      if (view == 'useful') return !flagged && priority != 'low';
      if (view == 'high') return !flagged && priority == 'high';
      if (view == 'low') return !flagged && priority == 'low';
      if (view == 'flagged') return flagged;
      return true;
    }).length;
  }

  Widget _eventViewChip(String label, String value, IconData icon) {
    final selected = _eventView == value;
    return ChoiceChip(
      selected: selected,
      showCheckmark: false,
      avatar: Icon(icon, size: 14, color: selected ? _ink : _muted),
      label: Text('$label ${_eventCountFor(value)}'),
      labelStyle:
          TextStyle(color: selected ? _ink : _muted, fontSize: 11),
      selectedColor: _mint,
      backgroundColor: _panelRaised,
      side: const BorderSide(color: _line),
      onSelected: (_) => setState(() => _eventView = value),
    );
  }

  Widget _activityOverview() {
    final events = _visibleFeed
        .where((item) => (item['kind'] ?? 'event') == 'event')
        .toList();
    final seconds = events.fold<double>(0, (sum, item) {
      final start = item['ts'];
      final end = item['span_end'];
      return sum +
          (start is num && end is num
              ? (end.toDouble() - start.toDouble())
                  .clamp(0, 86400)
                  .toDouble()
              : 0);
    });
    // Which apps/cameras this room's day is made of — same labels as the tags.
    final applications = events
        .map(_sourceTag)
        .where((name) => name.isNotEmpty)
        .toSet()
        .take(3)
        .join(', ');
    final low = _eventCountFor('low');
    final flagged = _eventCountFor('flagged');

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(14, 10, 14, 10),
      decoration: const BoxDecoration(
        color: _panelRaised,
        border: Border(bottom: BorderSide(color: _line)),
      ),
      child: Row(
        children: [
          const Icon(Icons.summarize_outlined, size: 18, color: _mint),
          const SizedBox(width: 9),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${events.length} events • ${_duration(seconds)}'
                  '${applications.isEmpty ? '' : ' • $applications'}',
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12.5,
                      fontWeight: FontWeight.w600),
                ),
                if (low > 0 || flagged > 0)
                  Text(
                    '${low > 0 ? '$low low priority' : ''}'
                    '${low > 0 && flagged > 0 ? ' • ' : ''}'
                    '${flagged > 0 ? '$flagged flagged for review' : ''}',
                    style: const TextStyle(color: _muted, fontSize: 11),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _filterChip(String label, String kind) {
    return FilterChip(
      label: Text(label),
      selected: _kinds.contains(kind),
      onSelected: (selected) {
        setState(() {
          if (selected) {
            _kinds.add(kind);
          } else if (_kinds.length > 1) {
            _kinds.remove(kind);
          }
        });
        _load();
      },
      labelStyle: TextStyle(
          color: _kinds.contains(kind) ? _ink : _muted, fontSize: 11),
      selectedColor: _mint,
      backgroundColor: _panelRaised,
      side: const BorderSide(color: _line),
      showCheckmark: false,
    );
  }

  Future<void> _pickDate() async {
    final selected = await showDatePicker(
      context: context,
      initialDate: _date == null ? DateTime.now() : DateTime.parse(_date!),
      firstDate: DateTime(2020),
      lastDate: DateTime.now().add(const Duration(days: 1)),
    );
    if (selected == null) return;
    setState(() => _date =
        '${selected.year.toString().padLeft(4, '0')}-${selected.month.toString().padLeft(2, '0')}-${selected.day.toString().padLeft(2, '0')}');
    _load();
  }

  Widget _feedItem(Map<String, dynamic> it) {
    final kind = (it['kind'] ?? 'event').toString();
    final text = (it['text'] ?? '').toString().trim();
    if (kind == 'note') return _noteCard(it, text);
    if (kind == 'message') {
      final role = (it['role'] ?? 'assistant').toString();
      if (it['_thinking'] == true) return const _ThinkingBubble();
      if (role == 'coach') return _coachCard(text);
      return _bubble(text, role == 'user');
    }
    return _eventCard(it, text);
  }

  // Kept temporarily for compatibility while older layouts are phased out.
  // ignore: unused_element
  Widget _eventRow(Map<String, dynamic> it, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
            margin: const EdgeInsets.only(top: 1),
            decoration: BoxDecoration(
              color: _panel,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: _line),
            ),
            child: Text(_hm(it['ts']),
                style: const TextStyle(
                    color: _mint, fontSize: 11, fontWeight: FontWeight.w600)),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(text.isEmpty ? '(no summary)' : text,
                style: const TextStyle(color: Colors.white70, fontSize: 13, height: 1.3)),
          ),
          if (it['event_id'] != null)
            PopupMenuButton<String>(
              padding: EdgeInsets.zero,
              iconSize: 18,
              icon: const Icon(Icons.more_horiz, color: _muted),
              onSelected: (value) =>
                  _assignEvent(it['event_id'].toString(), value),
              itemBuilder: (_) => const [
                PopupMenuItem(value: 'primary', child: Text('Move to room…')),
                PopupMenuItem(value: 'secondary', child: Text('Add to room…')),
                PopupMenuItem(value: 'remove', child: Text('Remove from this room')),
              ],
            ),
        ],
      ),
    );
  }

  Widget _eventCard(Map<String, dynamic> it, String text) {
    final priority = (it['priority'] ?? 'normal').toString();
    final flagged = it['flagged'] == true;
    final priorityColor = priority == 'high'
        ? const Color(0xFFFFC857)
        : priority == 'low'
            ? _muted
            : _mint;
    // Screen and Cameras each hold every source of their kind, so the app or
    // camera an event came from is a tag on the bubble: opera, pycharm64 in
    // Screen; ipc-a22e-g in Cameras.
    final source = _sourceTag(it);
    final activity =
        (it['activity_type'] ?? '').toString().replaceAll('_', ' ');
    final start = it['ts'];
    final end = it['span_end'];
    final seconds = start is num && end is num
        ? (end.toDouble() - start.toDouble()).clamp(0, 86400).toDouble()
        : 0.0;
    final time = end is num ? '${_hm(start)}–${_hm(end)}' : _hm(start);

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.fromLTRB(12, 10, 6, 11),
      decoration: BoxDecoration(
        color: flagged ? const Color(0xFF2A1B27) : _panel,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
            color: flagged
                ? const Color(0xFFFF7A9B).withOpacity(.55)
                : _line),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                flagged
                    ? Icons.flag
                    : priority == 'high'
                        ? Icons.star
                        : priority == 'low'
                            ? Icons.low_priority
                            : Icons.bolt,
                size: 16,
                color: flagged ? const Color(0xFFFF7A9B) : priorityColor,
              ),
              const SizedBox(width: 7),
              Expanded(
                child: Text(
                  activity.isEmpty ? 'Activity' : activity,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.w700),
                ),
              ),
              Text(time, style: const TextStyle(color: _muted, fontSize: 11)),
              if (it['event_id'] != null)
                PopupMenuButton<String>(
                  padding: EdgeInsets.zero,
                  iconSize: 18,
                  icon: const Icon(Icons.more_horiz, color: _muted),
                  onSelected: (value) => _handleEventAction(it, value),
                  itemBuilder: (_) => [
                    const PopupMenuItem(
                        value: 'priority_high',
                        child: Text('Mark important')),
                    const PopupMenuItem(
                        value: 'priority_normal',
                        child: Text('Normal priority')),
                    const PopupMenuItem(
                        value: 'priority_low',
                        child: Text('Low priority')),
                    PopupMenuItem(
                        value: flagged ? 'unflag' : 'flag',
                        child: Text(flagged
                            ? 'Return from review'
                            : 'Flag for review')),
                    const PopupMenuDivider(),
                    const PopupMenuItem(
                        value: 'primary', child: Text('Move to room…')),
                    const PopupMenuItem(
                        value: 'secondary', child: Text('Add to room…')),
                    const PopupMenuItem(
                        value: 'remove',
                        child: Text('Remove from this room')),
                  ],
                ),
            ],
          ),
          const SizedBox(height: 7),
          Text(text.isEmpty ? 'No useful description was captured.' : text,
              style: TextStyle(
                  color: priority == 'low' ? _muted : Colors.white70,
                  fontSize: 13,
                  height: 1.38)),
          const SizedBox(height: 8),
          Wrap(
            spacing: 6,
            runSpacing: 5,
            children: [
              if (source.isNotEmpty)
                _eventTag(
                    source,
                    _isCameraRoom ? Icons.videocam : Icons.desktop_windows,
                    _isCameraRoom ? const Color(0xFFF59E0B) : _violet),
              _eventTag(_duration(seconds), Icons.schedule, _muted),
              _eventTag(
                  priority == 'high'
                      ? 'Important'
                      : priority == 'low'
                          ? 'Low priority'
                          : 'Normal',
                  priority == 'high'
                      ? Icons.star_outline
                      : priority == 'low'
                          ? Icons.low_priority
                          : Icons.bolt_outlined,
                  priorityColor),
              if (it['priority_source'] == 'automatic')
                _eventTag('AI ranked', Icons.auto_awesome, _violet),
              if (flagged)
                _eventTag('Review later', Icons.flag_outlined,
                    const Color(0xFFFF7A9B)),
            ],
          ),
          if (flagged &&
              (it['flag_reason'] ?? '').toString().trim().isNotEmpty) ...[
            const SizedBox(height: 7),
            Text('Reason: ${it['flag_reason']}',
                style: const TextStyle(
                    color: Color(0xFFFFA0B8), fontSize: 11.5)),
          ],
        ],
      ),
    );
  }

  Widget _eventTag(String label, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
      decoration: BoxDecoration(
        color: color.withOpacity(.10),
        borderRadius: BorderRadius.circular(7),
        border: Border.all(color: color.withOpacity(.28)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: color),
          const SizedBox(width: 4),
          Text(label, style: TextStyle(color: color, fontSize: 10.5)),
        ],
      ),
    );
  }

  String _duration(double seconds) {
    if (seconds < 60) return '${seconds.round()} sec';
    final minutes = (seconds / 60).round();
    if (minutes < 60) return '$minutes min';
    final hours = minutes ~/ 60;
    final remainder = minutes % 60;
    return remainder == 0 ? '$hours hr' : '$hours hr $remainder min';
  }

  Future<void> _handleEventAction(
      Map<String, dynamic> item, String action) async {
    final eventId = item['event_id']?.toString();
    if (eventId == null) return;
    if (action.startsWith('priority_')) {
      await _triageEvent(
          eventId, priority: action.replaceFirst('priority_', ''));
    } else if (action == 'flag') {
      await _flagEvent(item);
    } else if (action == 'unflag') {
      await _triageEvent(eventId, flagged: false, flagReason: '');
    } else {
      await _assignEvent(eventId, action);
    }
  }

  Future<void> _flagEvent(Map<String, dynamic> item) async {
    final eventId = item['event_id']?.toString();
    if (eventId == null) return;
    final reason = TextEditingController();
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Flag event for review'),
        content: TextField(
          controller: reason,
          autofocus: true,
          minLines: 2,
          maxLines: 4,
          decoration: const InputDecoration(
              hintText: 'Optional: why should this be reviewed?'),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Flag')),
        ],
      ),
    );
    if (confirm == true) {
      await _triageEvent(
          eventId, flagged: true, flagReason: reason.text.trim());
    }
    reason.dispose();
  }

  Future<void> _triageEvent(String eventId,
      {String? priority, bool? flagged, String? flagReason}) async {
    try {
      final body = <String, dynamic>{
        if (priority != null) 'priority': priority,
        if (flagged != null) 'flagged': flagged,
        if (flagReason != null) 'flag_reason': flagReason,
      };
      final response = await http
          .patch(
            Uri.parse('${widget.apiBase}/memory/events/$eventId'),
            headers: {'Content-Type': 'application/json'},
            body: json.encode(body),
          )
          .timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        await _load();
      } else {
        _snack('Could not update event: HTTP ${response.statusCode}');
      }
    } catch (e) {
      _snack('Could not update event: $e');
    }
  }

  Widget _noteCard(Map<String, dynamic> it, String text) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: _violet.withOpacity(.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _violet.withOpacity(.4)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.push_pin, size: 15, color: _violet),
          const SizedBox(width: 8),
          Expanded(
            child: Text(text,
                style: const TextStyle(color: Colors.white, fontSize: 13, height: 1.3)),
          ),
          PopupMenuButton<String>(
            padding: EdgeInsets.zero,
            iconSize: 18,
            icon: const Icon(Icons.more_vert, color: _muted),
            onSelected: (value) =>
                value == 'edit' ? _editNote(it, text) : _deleteNote(it),
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'edit', child: Text('Edit note')),
              PopupMenuItem(value: 'delete', child: Text('Delete note')),
            ],
          ),
        ],
      ),
    );
  }

  Future<void> _editNote(Map<String, dynamic> item, String current) async {
    final id = item['note_id']?.toString();
    if (id == null) return;
    final controller = TextEditingController(text: current);
    final save = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Edit note'),
        content: TextField(controller: controller, minLines: 3, maxLines: 8),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Save')),
        ],
      ),
    );
    if (save != true || controller.text.trim().isEmpty) return;
    final resp = await http.patch(
      Uri.parse('${widget.apiBase}/rooms/${widget.roomId}/notes/$id'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'text': controller.text.trim()}),
    );
    if (resp.statusCode == 200) {
      _load();
    } else {
      _snack('Could not update note: HTTP ${resp.statusCode}');
    }
  }

  Future<void> _deleteNote(Map<String, dynamic> item) async {
    final id = item['note_id']?.toString();
    if (id == null) return;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete note?'),
        content: const Text('This cannot be undone.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Delete')),
        ],
      ),
    );
    if (confirmed != true) return;
    final resp = await http.delete(
        Uri.parse('${widget.apiBase}/rooms/${widget.roomId}/notes/$id'));
    if (resp.statusCode == 200) {
      _load();
    } else {
      _snack('Could not delete note: HTTP ${resp.statusCode}');
    }
  }

  Future<void> _assignEvent(String eventId, String mode) async {
    if (mode == 'remove') {
      if (_isDaily) {
        _snack('Daily membership cannot be removed');
        return;
      }
      final resp = await http.delete(Uri.parse(
          '${widget.apiBase}/events/$eventId/rooms/${widget.roomId}'));
      if (resp.statusCode == 200) {
        _load();
      } else {
        _snack('Could not remove event: HTTP ${resp.statusCode}');
      }
      return;
    }
    try {
      final resp = await http.get(Uri.parse('${widget.apiBase}/rooms'));
      if (resp.statusCode != 200) {
        _snack('Could not load rooms');
        return;
      }
      final rooms =
          ((json.decode(resp.body) as Map<String, dynamic>)['rooms'] as List)
              .cast<Map<String, dynamic>>()
              .where((r) => r['kind'] != 'daily')
              .toList();
      if (!mounted) return;
      final selected = await showDialog<String>(
        context: context,
        builder: (context) => SimpleDialog(
          title: Text(mode == 'primary' ? 'Move to room' : 'Add to room'),
          children: rooms
              .map((room) => SimpleDialogOption(
                    onPressed: () =>
                        Navigator.pop(context, room['room_id'].toString()),
                    child: Text(room['name'].toString()),
                  ))
              .toList(),
        ),
      );
      if (selected == null) return;
      final update = await http.put(
        Uri.parse('${widget.apiBase}/events/$eventId/room'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'room_id': selected, 'mode': mode}),
      );
      if (update.statusCode == 200) {
        _load();
      } else {
        _snack('Could not assign event: HTTP ${update.statusCode}');
      }
    } catch (e) {
      _snack('Could not assign event: $e');
    }
  }

  Widget _bubble(String text, bool isUser) {
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 5),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * .75),
        decoration: BoxDecoration(
          color: isUser ? _mint.withOpacity(.16) : _panelRaised,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: isUser ? _mint.withOpacity(.4) : _line),
        ),
        child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 13, height: 1.3)),
      ),
    );
  }

  Widget _coachCard(String text) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _panel,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _mint.withOpacity(.5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(children: [
            Icon(Icons.insights, size: 16, color: _mint),
            SizedBox(width: 6),
            Text('Coach', style: TextStyle(color: _mint, fontWeight: FontWeight.w700)),
          ]),
          const SizedBox(height: 8),
          Text(text,
              style: const TextStyle(
                  color: Colors.white70, fontSize: 12.5, height: 1.4, fontFamily: 'monospace')),
        ],
      ),
    );
  }

  Widget _composer() {
    return Container(
      padding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
      decoration: const BoxDecoration(
        color: _panel,
        border: Border(top: BorderSide(color: _line)),
      ),
      child: Column(
        children: [
          Row(
            children: [
              _modeChip('Note', 'note', Icons.push_pin),
              const SizedBox(width: 8),
              _modeChip('Ask agent', 'chat', Icons.chat_bubble_outline),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _input,
                  minLines: 1,
                  maxLines: 4,
                  style: const TextStyle(color: Colors.white),
                  decoration: InputDecoration(
                    hintText: _listening
                        ? 'Listening… tap the mic to stop'
                        : _transcribing
                            ? 'Transcribing…'
                            : _mode == 'note'
                                ? 'Write or speak a thought…'
                                : 'Ask the agent about this room…',
                    hintStyle: const TextStyle(color: _muted),
                    isDense: true,
                  ),
                ),
              ),
              const SizedBox(width: 4),
              IconButton(
                tooltip: _listening ? 'Stop and transcribe' : 'Dictate',
                onPressed: _transcribing ? null : _toggleDictation,
                icon: _transcribing
                    ? const SizedBox(
                        width: 18, height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2, color: _mint))
                    : Icon(_listening ? Icons.stop_circle : Icons.mic_none,
                        color: _listening ? const Color(0xFFFF607C) : _muted),
              ),
              const SizedBox(width: 4),
              IconButton(
                style: IconButton.styleFrom(backgroundColor: _mint),
                icon: _sending
                    ? const SizedBox(
                        width: 18, height: 18,
                        child: CircularProgressIndicator(strokeWidth: 2, color: _ink))
                    : const Icon(Icons.send, color: _ink),
                onPressed: (_sending || _listening) ? null : _send,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _modeChip(String label, String value, IconData icon) {
    final selected = _mode == value;
    return ChoiceChip(
      selected: selected,
      showCheckmark: false,
      avatar: Icon(icon, size: 15, color: selected ? _ink : _muted),
      label: Text(label),
      labelStyle: TextStyle(color: selected ? _ink : _muted, fontSize: 12),
      selectedColor: _mint,
      backgroundColor: _panelRaised,
      side: const BorderSide(color: _line),
      onSelected: (_) => setState(() => _mode = value),
    );
  }
}

/// A left-aligned assistant bubble with animated dots, shown while the reply
/// is in flight (the model can take a while — single-sequence vLLM).
class _ThinkingBubble extends StatefulWidget {
  const _ThinkingBubble();

  @override
  State<_ThinkingBubble> createState() => _ThinkingBubbleState();
}

class _ThinkingBubbleState extends State<_ThinkingBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1200))
        ..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 5),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: _panelRaised,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: _line),
        ),
        child: AnimatedBuilder(
          animation: _c,
          builder: (context, _) {
            return Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (i) {
                final t = ((_c.value + i / 3) % 1.0);
                final opacity = 0.3 + 0.7 * (1 - (2 * t - 1).abs());
                return Padding(
                  padding: EdgeInsets.only(right: i < 2 ? 5 : 0),
                  child: Opacity(
                    opacity: opacity,
                    child: Container(
                      width: 7,
                      height: 7,
                      decoration: const BoxDecoration(
                        color: _mint,
                        shape: BoxShape.circle,
                      ),
                    ),
                  ),
                );
              }),
            );
          },
        ),
      ),
    );
  }
}

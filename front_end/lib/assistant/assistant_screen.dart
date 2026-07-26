import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../memory/timeline_screen.dart';

const _ink = Color(0xFF070B14);
const _panel = Color(0xFF111827);
const _panelRaised = Color(0xFF182235);
const _line = Color(0xFF263246);
const _mint = Color(0xFF6EE7D8);
const _violet = Color(0xFF8B7CF6);
const _muted = Color(0xFF91A0B8);

String _iso(DateTime value) =>
    '${value.year.toString().padLeft(4, '0')}-'
    '${value.month.toString().padLeft(2, '0')}-'
    '${value.day.toString().padLeft(2, '0')}';

class AssistantScreen extends StatefulWidget {
  final String apiBase;
  const AssistantScreen({super.key, required this.apiBase});

  @override
  State<AssistantScreen> createState() => _AssistantScreenState();
}

class _AssistantScreenState extends State<AssistantScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs;
  List<dynamic> _conversations = [];
  List<dynamic> _rooms = [];
  Map<String, dynamic>? _conversation;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 3, vsync: this);
    _loadRooms();
    _loadConversations();
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  void _snack(String text) {
    if (mounted) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text(text)));
    }
  }

  Future<void> _loadRooms() async {
    try {
      final resp = await http.get(Uri.parse('${widget.apiBase}/rooms'));
      if (resp.statusCode == 200 && mounted) {
        final data = json.decode(resp.body) as Map<String, dynamic>;
        setState(() => _rooms = (data['rooms'] as List?) ?? []);
      }
    } catch (_) {}
  }

  Future<void> _loadConversations({String? selectId}) async {
    setState(() => _loading = true);
    try {
      final resp = await http
          .get(Uri.parse('${widget.apiBase}/assistant/conversations'));
      if (resp.statusCode != 200) throw Exception('HTTP ${resp.statusCode}');
      final data = json.decode(resp.body) as Map<String, dynamic>;
      final items = (data['conversations'] as List?) ?? [];
      setState(() => _conversations = items);
      final id = selectId ??
          _conversation?['conversation_id']?.toString() ??
          (items.isEmpty ? null : items.first['conversation_id'].toString());
      if (id != null) await _loadConversation(id);
    } catch (e) {
      _snack('Could not load conversations: $e');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _loadConversation(String id) async {
    final resp = await http
        .get(Uri.parse('${widget.apiBase}/assistant/conversations/$id'));
    if (resp.statusCode != 200) return;
    final data = json.decode(resp.body) as Map<String, dynamic>;
    if (mounted) {
      setState(() =>
          _conversation = data['conversation'] as Map<String, dynamic>);
    }
  }

  Future<void> _newConversation() async {
    String scope = 'all';
    String? roomId;
    final title = TextEditingController(text: 'New conversation');
    final save = await showDialog<bool>(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('New grounded conversation'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                  controller: title,
                  decoration: const InputDecoration(labelText: 'Title')),
              const SizedBox(height: 10),
              DropdownButtonFormField<String>(
                initialValue: scope,
                decoration: const InputDecoration(labelText: 'Memory scope'),
                items: const [
                  DropdownMenuItem(
                      value: 'all', child: Text('All memory')),
                  DropdownMenuItem(
                      value: 'today', child: Text('Today only')),
                  DropdownMenuItem(
                      value: 'room', child: Text('One room')),
                ],
                onChanged: (value) =>
                    setDialogState(() => scope = value ?? 'all'),
              ),
              if (scope == 'room') ...[
                const SizedBox(height: 10),
                DropdownButtonFormField<String>(
                  initialValue: roomId,
                  decoration: const InputDecoration(labelText: 'Room'),
                  items: _rooms
                      .map((room) => DropdownMenuItem<String>(
                            value: room['room_id'].toString(),
                            child: Text(room['name'].toString()),
                          ))
                      .toList(),
                  onChanged: (value) =>
                      setDialogState(() => roomId = value),
                ),
              ],
            ],
          ),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: const Text('Cancel')),
            FilledButton(
                onPressed: () => Navigator.pop(
                    context, scope != 'room' || roomId != null),
                child: const Text('Create')),
          ],
        ),
      ),
    );
    if (save != true) return;
    final resp = await http.post(
      Uri.parse('${widget.apiBase}/assistant/conversations'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'title': title.text.trim().isEmpty
            ? 'New conversation'
            : title.text.trim(),
        'scope': scope,
        'room_id': roomId,
      }),
    );
    if (resp.statusCode == 201) {
      final data = json.decode(resp.body) as Map<String, dynamic>;
      final id = data['conversation']['conversation_id'].toString();
      await _loadConversations(selectId: id);
    } else {
      _snack('Could not create conversation');
    }
  }

  Future<void> _deleteConversation() async {
    final id = _conversation?['conversation_id']?.toString();
    if (id == null) return;
    final resp = await http
        .delete(Uri.parse('${widget.apiBase}/assistant/conversations/$id'));
    if (resp.statusCode == 200) {
      setState(() => _conversation = null);
      _loadConversations();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _ink,
      appBar: AppBar(
        backgroundColor: _panel,
        title: const Text('Assistant'),
        bottom: TabBar(
          controller: _tabs,
          indicatorColor: _mint,
          labelColor: _mint,
          unselectedLabelColor: _muted,
          tabs: const [
            Tab(icon: Icon(Icons.chat_bubble_outline), text: 'Conversations'),
            Tab(icon: Icon(Icons.insights), text: 'Reviews'),
            Tab(icon: Icon(Icons.center_focus_strong), text: 'Focus'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabs,
        children: [
          _conversationTab(),
          ReviewPanel(apiBase: widget.apiBase),
          FocusPanel(apiBase: widget.apiBase, rooms: _rooms),
        ],
      ),
    );
  }

  Widget _conversationTab() {
    return Column(
      children: [
        Container(
          color: _panel,
          padding: const EdgeInsets.all(10),
          child: Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<String>(
                  initialValue: _conversation?['conversation_id']?.toString(),
                  isExpanded: true,
                  decoration: const InputDecoration(
                      isDense: true, labelText: 'Conversation'),
                  dropdownColor: _panelRaised,
                  items: _conversations
                      .map((item) => DropdownMenuItem<String>(
                            value: item['conversation_id'].toString(),
                            child: Text(item['title'].toString(),
                                overflow: TextOverflow.ellipsis),
                          ))
                      .toList(),
                  onChanged: (id) {
                    if (id != null) _loadConversation(id);
                  },
                ),
              ),
              IconButton(
                tooltip: 'New conversation',
                icon: const Icon(Icons.add, color: _mint),
                onPressed: _newConversation,
              ),
              IconButton(
                tooltip: 'Delete conversation',
                icon: const Icon(Icons.delete_outline, color: _muted),
                onPressed:
                    _conversation == null ? null : _deleteConversation,
              ),
            ],
          ),
        ),
        if (_loading)
          const LinearProgressIndicator(minHeight: 2, color: _mint),
        Expanded(
          child: _conversation == null
              ? Center(
                  child: FilledButton.icon(
                    onPressed: _newConversation,
                    icon: const Icon(Icons.add),
                    label: const Text('Start a grounded conversation'),
                  ),
                )
              : ConversationPanel(
                  apiBase: widget.apiBase,
                  conversation: _conversation!,
                  onChanged: () => _loadConversation(
                      _conversation!['conversation_id'].toString()),
                ),
        ),
      ],
    );
  }
}

class ConversationPanel extends StatefulWidget {
  final String apiBase;
  final Map<String, dynamic> conversation;
  final VoidCallback onChanged;
  const ConversationPanel(
      {super.key,
      required this.apiBase,
      required this.conversation,
      required this.onChanged});

  @override
  State<ConversationPanel> createState() => _ConversationPanelState();
}

class _ConversationPanelState extends State<ConversationPanel> {
  final TextEditingController _input = TextEditingController();
  final ScrollController _scroll = ScrollController();
  bool _sending = false;

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    if (text.isEmpty || _sending) return;
    setState(() => _sending = true);
    try {
      final id = widget.conversation['conversation_id'];
      final resp = await http.post(
        Uri.parse('${widget.apiBase}/assistant/conversations/$id/messages'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'message': text}),
      ).timeout(const Duration(seconds: 90));
      if (resp.statusCode == 200) {
        _input.clear();
        widget.onChanged();
      } else if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Assistant failed: HTTP ${resp.statusCode}')));
      }
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final messages = (widget.conversation['messages'] as List?) ?? [];
    final scope = widget.conversation['scope']?.toString() ?? 'all';
    return Column(
      children: [
        Container(
          width: double.infinity,
          color: _panel.withValues(alpha: .55),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
          child: Text(
            'Grounding: $scope'
            '${widget.conversation['room_id'] != null ? ' · ${widget.conversation['room_id']}' : ''}',
            style: const TextStyle(color: _muted, fontSize: 11),
          ),
        ),
        Expanded(
          child: messages.isEmpty
              ? const Center(
                  child: Text(
                    'Ask about your work, activities, notes, or remembered context.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: _muted),
                  ),
                )
              : ListView.builder(
                  controller: _scroll,
                  padding: const EdgeInsets.all(12),
                  itemCount: messages.length,
                  itemBuilder: (_, index) => _message(
                      messages[index] as Map<String, dynamic>),
                ),
        ),
        if (_sending)
          const LinearProgressIndicator(minHeight: 2, color: _mint),
        Container(
          color: _panel,
          padding: const EdgeInsets.all(10),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _input,
                  minLines: 1,
                  maxLines: 5,
                  textInputAction: TextInputAction.newline,
                  style: const TextStyle(color: Colors.white),
                  decoration: const InputDecoration(
                      hintText: 'Ask with memory citations…'),
                ),
              ),
              const SizedBox(width: 8),
              IconButton.filled(
                onPressed: _sending ? null : _send,
                style: IconButton.styleFrom(backgroundColor: _mint),
                icon: const Icon(Icons.send, color: _ink),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _message(Map<String, dynamic> message) {
    final user = message['role'] == 'user';
    final citations = (message['citations'] as List?) ?? [];
    return Align(
      alignment: user ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints:
            BoxConstraints(maxWidth: MediaQuery.of(context).size.width * .82),
        margin: const EdgeInsets.symmetric(vertical: 5),
        padding: const EdgeInsets.all(11),
        decoration: BoxDecoration(
          color: user ? _mint.withValues(alpha: .15) : _panelRaised,
          borderRadius: BorderRadius.circular(13),
          border: Border.all(
              color: user ? _mint.withValues(alpha: .4) : _line),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SelectableText(message['text'].toString(),
                style: const TextStyle(color: Colors.white, height: 1.35)),
            if (citations.isNotEmpty) ...[
              const SizedBox(height: 9),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: citations
                    .map((citation) => ActionChip(
                          avatar: Text('[${citation['number']}]',
                              style: const TextStyle(fontSize: 10)),
                          label: Text(
                              (citation['title'] ?? citation['kind']).toString(),
                              overflow: TextOverflow.ellipsis),
                          onPressed: () => _openCitation(
                              citation as Map<String, dynamic>),
                        ))
                    .toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }

  void _openCitation(Map<String, dynamic> citation) {
    if (citation['kind'] == 'event') {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => MemoryEventScreen(
              apiBase: widget.apiBase, eventId: citation['id'].toString()),
        ),
      );
    } else {
      showModalBottomSheet(
        context: context,
        backgroundColor: _panel,
        showDragHandle: true,
        builder: (_) => Padding(
          padding: const EdgeInsets.all(20),
          child: Text(citation['text'].toString(),
              style: const TextStyle(color: Colors.white70, height: 1.4)),
        ),
      );
    }
  }
}

class ReviewPanel extends StatefulWidget {
  final String apiBase;
  const ReviewPanel({super.key, required this.apiBase});

  @override
  State<ReviewPanel> createState() => _ReviewPanelState();
}

class _ReviewPanelState extends State<ReviewPanel> {
  DateTime _date = DateTime.now();
  bool _weekly = false;
  bool _loading = false;
  Map<String, dynamic>? _data;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final path = _weekly
          ? '/reviews/weekly?end_date=${_iso(_date)}'
          : '/reviews/daily?date=${_iso(_date)}';
      final resp = await http.get(Uri.parse('${widget.apiBase}$path'));
      if (resp.statusCode != 200) throw Exception('HTTP ${resp.statusCode}');
      setState(() =>
          _data = json.decode(resp.body) as Map<String, dynamic>);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          color: _panel,
          padding: const EdgeInsets.all(10),
          child: Row(
            children: [
              SegmentedButton<bool>(
                segments: const [
                  ButtonSegment(value: false, label: Text('Daily')),
                  ButtonSegment(value: true, label: Text('Weekly')),
                ],
                selected: {_weekly},
                onSelectionChanged: (value) {
                  setState(() => _weekly = value.first);
                  _load();
                },
              ),
              const Spacer(),
              TextButton.icon(
                onPressed: _pick,
                icon: const Icon(Icons.event),
                label: Text(_iso(_date)),
              ),
            ],
          ),
        ),
        if (_loading)
          const LinearProgressIndicator(minHeight: 2, color: _mint),
        Expanded(
          child: _data == null
              ? const Center(
                  child: Text('No review available.',
                      style: TextStyle(color: _muted)))
              : _weekly
                  ? _weeklyView()
                  : _dailyView(),
        ),
      ],
    );
  }

  Future<void> _pick() async {
    final value = await showDatePicker(
      context: context,
      initialDate: _date,
      firstDate: DateTime(2020),
      lastDate: DateTime.now().add(const Duration(days: 1)),
    );
    if (value == null) return;
    setState(() => _date = value);
    _load();
  }

  Widget _dailyView() {
    final metrics =
        (_data!['metrics'] as Map<String, dynamic>?) ?? {};
    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        _metricGrid(metrics),
        const SizedBox(height: 16),
        SelectableText((_data!['report'] ?? '').toString(),
            style:
                const TextStyle(color: Colors.white70, height: 1.45)),
        const SizedBox(height: 16),
        FilledButton.icon(
          onPressed: _generateCoach,
          icon: const Icon(Icons.auto_awesome),
          label: const Text('Generate Coach feedback in Daily room'),
        ),
      ],
    );
  }

  Widget _weeklyView() {
    final summary =
        (_data!['summary'] as Map<String, dynamic>?) ?? {};
    final days = (_data!['days'] as List?) ?? [];
    return ListView(
      padding: const EdgeInsets.all(14),
      children: [
        _metricGrid({
          'active_minutes': summary['active_minutes'],
          'events': summary['events'],
          'focus_score': summary['average_focus_score'],
          'switches': summary['switches'],
        }),
        const SizedBox(height: 18),
        const Text('Daily trend',
            style: TextStyle(
                color: _mint, fontWeight: FontWeight.w700)),
        ...days.map((day) => ListTile(
              title: Text(day['date'].toString(),
                  style: const TextStyle(color: Colors.white)),
              subtitle: Text(
                  '${day['active_minutes']} min · ${day['events']} events',
                  style: const TextStyle(color: _muted)),
              trailing: Text('${day['focus_score']}/100',
                  style: const TextStyle(color: _mint)),
            )),
      ],
    );
  }

  Widget _metricGrid(Map<String, dynamic> metrics) {
    final items = [
      ('Active', '${metrics['active_minutes'] ?? 0} min'),
      ('Events', '${metrics['events'] ?? 0}'),
      ('Focus', '${metrics['focus_score'] ?? 0}/100'),
      ('Switches', '${metrics['switches'] ?? 0}'),
    ];
    return Wrap(
      spacing: 9,
      runSpacing: 9,
      children: items
          .map((item) => Container(
                width: 145,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                    color: _panelRaised,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: _line)),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(item.$1,
                        style:
                            const TextStyle(color: _muted, fontSize: 11)),
                    Text(item.$2,
                        style: const TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.w700)),
                  ],
                ),
              ))
          .toList(),
    );
  }

  Future<void> _generateCoach() async {
    final resp = await http.post(Uri.parse(
        '${widget.apiBase}/rooms/daily/report?date=${_iso(_date)}&post=true'));
    if (resp.statusCode == 200 && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Coach report posted to Daily')));
    }
  }
}

class FocusPanel extends StatefulWidget {
  final String apiBase;
  final List<dynamic> rooms;
  const FocusPanel({super.key, required this.apiBase, required this.rooms});

  @override
  State<FocusPanel> createState() => _FocusPanelState();
}

class _FocusPanelState extends State<FocusPanel> {
  Map<String, dynamic>? _active;
  List<dynamic> _history = [];
  bool _loading = false;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _load();
    _timer = Timer.periodic(const Duration(seconds: 30), (_) {
      if (mounted && _active != null) setState(() {});
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final resp =
          await http.get(Uri.parse('${widget.apiBase}/focus/sessions'));
      if (resp.statusCode != 200) return;
      final data = json.decode(resp.body) as Map<String, dynamic>;
      setState(() {
        _active = data['active'] as Map<String, dynamic>?;
        _history = (data['sessions'] as List?) ?? [];
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        if (_loading)
          const LinearProgressIndicator(minHeight: 2, color: _mint),
        if (_active != null) _activeCard() else _startCard(),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(12),
            itemCount: _history.where((item) => item['state'] != 'active').length,
            itemBuilder: (_, index) {
              final completed = _history
                  .where((item) => item['state'] != 'active')
                  .toList();
              final item = completed[index] as Map<String, dynamic>;
              return Card(
                color: _panelRaised,
                child: ListTile(
                  leading:
                      const Icon(Icons.task_alt, color: _mint),
                  title: Text(item['goal'].toString(),
                      style: const TextStyle(color: Colors.white)),
                  subtitle: Text(
                      '${((((item['active_seconds'] as num?) ?? 0) / 60).round())} active min · ${item['events'] ?? 0} events',
                      style: const TextStyle(color: _muted)),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _startCard() {
    return Container(
      margin: const EdgeInsets.all(14),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
          color: _panelRaised,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: _line)),
      child: Column(
        children: [
          const Icon(Icons.center_focus_strong,
              color: _violet, size: 42),
          const SizedBox(height: 10),
          const Text('Start an intentional focus session',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 17,
                  fontWeight: FontWeight.w700)),
          const SizedBox(height: 12),
          FilledButton.icon(
              onPressed: _start,
              icon: const Icon(Icons.play_arrow),
              label: const Text('Start focus')),
        ],
      ),
    );
  }

  Widget _activeCard() {
    final started = DateTime.fromMillisecondsSinceEpoch(
        (((_active!['started_at'] as num?) ?? 0) * 1000).round());
    final elapsed = DateTime.now().difference(started).inMinutes;
    return Container(
      margin: const EdgeInsets.all(14),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
          color: _mint.withValues(alpha: .1),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: _mint)),
      child: Column(
        children: [
          Text(_active!['goal'].toString(),
              textAlign: TextAlign.center,
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 19,
                  fontWeight: FontWeight.w700)),
          const SizedBox(height: 8),
          Text('$elapsed / ${_active!['planned_minutes']} minutes',
              style: const TextStyle(color: _mint)),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: _stop,
            icon: const Icon(Icons.stop),
            label: const Text('Finish and summarize'),
          ),
        ],
      ),
    );
  }

  Future<void> _start() async {
    final goal = TextEditingController();
    final minutes = TextEditingController(text: '25');
    String? roomId;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('Start focus'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                  controller: goal,
                  decoration: const InputDecoration(labelText: 'Goal')),
              const SizedBox(height: 8),
              TextField(
                  controller: minutes,
                  keyboardType: TextInputType.number,
                  decoration:
                      const InputDecoration(labelText: 'Planned minutes')),
              const SizedBox(height: 8),
              DropdownButtonFormField<String?>(
                initialValue: roomId,
                decoration:
                    const InputDecoration(labelText: 'Room (optional)'),
                items: [
                  const DropdownMenuItem<String?>(
                      value: null, child: Text('Any activity')),
                  ...widget.rooms
                      .where((room) => room['kind'] != 'daily')
                      .map((room) => DropdownMenuItem<String?>(
                            value: room['room_id'].toString(),
                            child: Text(room['name'].toString()),
                          )),
                ],
                onChanged: (value) =>
                    setDialogState(() => roomId = value),
              ),
            ],
          ),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: const Text('Cancel')),
            FilledButton(
                onPressed: () => Navigator.pop(
                    context, goal.text.trim().isNotEmpty),
                child: const Text('Start')),
          ],
        ),
      ),
    );
    if (confirmed != true) return;
    final resp = await http.post(
      Uri.parse('${widget.apiBase}/focus/sessions'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'goal': goal.text.trim(),
        'planned_minutes': int.tryParse(minutes.text) ?? 25,
        'room_id': roomId,
      }),
    );
    if (resp.statusCode == 201) _load();
  }

  Future<void> _stop() async {
    final id = _active!['focus_id'];
    final resp = await http
        .post(Uri.parse('${widget.apiBase}/focus/sessions/$id/stop'));
    if (resp.statusCode == 200) {
      final data = json.decode(resp.body) as Map<String, dynamic>;
      final metrics = data['focus']['metrics'] as Map<String, dynamic>;
      await _load();
      if (!mounted) return;
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Focus session complete'),
          content: Text(
              '${metrics['events'] ?? 0} events\n'
              '${(((metrics['active_seconds'] as num?) ?? 0) / 60).round()} active minutes\n'
              '${((metrics['applications'] as List?) ?? []).join(', ')}'),
          actions: [
            FilledButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Done')),
          ],
        ),
      );
    }
  }
}

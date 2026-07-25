import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

const _ink = Color(0xFF070B14);
const _panel = Color(0xFF111827);
const _panelRaised = Color(0xFF182235);
const _line = Color(0xFF263246);
const _mint = Color(0xFF6EE7D8);
const _violet = Color(0xFF8B7CF6);
const _muted = Color(0xFF91A0B8);

String _dateIso(DateTime value) =>
    '${value.year.toString().padLeft(4, '0')}-'
    '${value.month.toString().padLeft(2, '0')}-'
    '${value.day.toString().padLeft(2, '0')}';

String _hm(dynamic ts) {
  if (ts is! num) return '--:--';
  final dt = DateTime.fromMillisecondsSinceEpoch((ts * 1000).round());
  return '${dt.hour.toString().padLeft(2, '0')}:'
      '${dt.minute.toString().padLeft(2, '0')}';
}

String _dayTime(dynamic ts) {
  if (ts is! num) return '';
  final dt = DateTime.fromMillisecondsSinceEpoch((ts * 1000).round());
  return '${_dateIso(dt)} ${_hm(ts)}';
}

IconData _kindIcon(String kind) {
  switch (kind) {
    case 'event':
      return Icons.history;
    case 'entity':
      return Icons.hub_outlined;
    case 'claim':
      return Icons.fact_check_outlined;
    case 'note':
      return Icons.push_pin_outlined;
    case 'message':
      return Icons.chat_bubble_outline;
    case 'room':
      return Icons.dashboard_customize_outlined;
    default:
      return Icons.memory;
  }
}

class MemoryTimelineScreen extends StatefulWidget {
  final String apiBase;
  const MemoryTimelineScreen({super.key, required this.apiBase});

  @override
  State<MemoryTimelineScreen> createState() => _MemoryTimelineScreenState();
}

class _MemoryTimelineScreenState extends State<MemoryTimelineScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs;
  final TextEditingController _query = TextEditingController();
  final TextEditingController _timelineFilter = TextEditingController();
  DateTime _date = DateTime.now();
  bool _timelineLoading = false;
  bool _searching = false;
  bool _entitiesLoading = false;
  String? _timelineError;
  String? _searchError;
  List<dynamic> _sessions = [];
  List<dynamic> _results = [];
  List<dynamic> _entities = [];
  List<dynamic> _rooms = [];
  String _memoryDomain = 'personal';
  DateTimeRange? _searchRange;
  String? _searchRoomId;
  final Set<String> _searchKinds = {
    'event',
    'note',
    'message',
    'entity',
    'claim',
    'room',
  };

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 3, vsync: this);
    _loadTimeline();
    _loadEntities();
    _loadRooms();
  }

  @override
  void dispose() {
    _tabs.dispose();
    _query.dispose();
    _timelineFilter.dispose();
    super.dispose();
  }

  void _snack(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text(message)));
    }
  }

  Future<void> _loadTimeline() async {
    setState(() {
      _timelineLoading = true;
      _timelineError = null;
    });
    try {
      final uri = Uri.parse('${widget.apiBase}/memory/timeline').replace(
        queryParameters: {
          'date': _dateIso(_date),
          'domain': _memoryDomain,
        },
      );
      final resp = await http.get(uri).timeout(const Duration(seconds: 20));
      if (resp.statusCode != 200) throw Exception('HTTP ${resp.statusCode}');
      final data = json.decode(resp.body) as Map<String, dynamic>;
      setState(() => _sessions = (data['sessions'] as List?) ?? []);
    } catch (e) {
      setState(() => _timelineError = e.toString());
    } finally {
      if (mounted) setState(() => _timelineLoading = false);
    }
  }

  Future<void> _loadEntities() async {
    setState(() => _entitiesLoading = true);
    try {
      final uri = Uri.parse('${widget.apiBase}/memory/entities').replace(
        queryParameters: {
          'date': _dateIso(_date),
          'limit': '100',
          'domain': _memoryDomain,
        },
      );
      final resp = await http.get(uri).timeout(const Duration(seconds: 20));
      if (resp.statusCode != 200) throw Exception('HTTP ${resp.statusCode}');
      final data = json.decode(resp.body) as Map<String, dynamic>;
      setState(() => _entities = (data['entities'] as List?) ?? []);
    } catch (e) {
      _snack('Could not load entities: $e');
    } finally {
      if (mounted) setState(() => _entitiesLoading = false);
    }
  }

  Future<void> _loadRooms() async {
    try {
      final resp = await http.get(Uri.parse('${widget.apiBase}/rooms'));
      if (resp.statusCode != 200) return;
      final data = json.decode(resp.body) as Map<String, dynamic>;
      if (mounted) setState(() => _rooms = (data['rooms'] as List?) ?? []);
    } catch (_) {}
  }

  Future<void> _search() async {
    final query = _query.text.trim();
    if (query.isEmpty) return;
    setState(() {
      _searching = true;
      _searchError = null;
    });
    try {
      final uri = Uri.parse('${widget.apiBase}/memory/search').replace(
        queryParameters: {
          'q': query,
          'limit': '100',
          'domain': _memoryDomain,
          if (_searchKinds.length < 6) 'kinds': _searchKinds.join(','),
          if (_searchRange != null) 'from_date': _dateIso(_searchRange!.start),
          if (_searchRange != null) 'to_date': _dateIso(_searchRange!.end),
          if (_searchRoomId != null) 'room_id': _searchRoomId!,
        },
      );
      final resp = await http.get(uri).timeout(const Duration(seconds: 30));
      if (resp.statusCode != 200) throw Exception('HTTP ${resp.statusCode}');
      final data = json.decode(resp.body) as Map<String, dynamic>;
      setState(() => _results = (data['results'] as List?) ?? []);
    } catch (e) {
      setState(() => _searchError = e.toString());
    } finally {
      if (mounted) setState(() => _searching = false);
    }
  }

  void _changeDay(int delta) {
    setState(() => _date = _date.add(Duration(days: delta)));
    _loadTimeline();
    _loadEntities();
  }

  void _changeMemoryDomain(String domain) {
    if (domain == _memoryDomain) return;
    setState(() {
      _memoryDomain = domain;
      _searchRoomId = null;
      _results = [];
    });
    _loadTimeline();
    _loadEntities();
    if (_query.text.trim().isNotEmpty) _search();
  }

  Future<void> _pickDay() async {
    final value = await showDatePicker(
      context: context,
      initialDate: _date,
      firstDate: DateTime(2020),
      lastDate: DateTime.now().add(const Duration(days: 1)),
    );
    if (value == null) return;
    setState(() => _date = value);
    _loadTimeline();
    _loadEntities();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _ink,
      appBar: AppBar(
        backgroundColor: _panel,
        title: Text(
          _memoryDomain == 'personal' ? 'Personal memory' : 'Home memory',
        ),
        actions: [
          IconButton(
            tooltip: 'Refresh',
            icon: const Icon(Icons.refresh, color: _mint),
            onPressed: () {
              _loadTimeline();
              _loadEntities();
              if (_query.text.trim().isNotEmpty) _search();
            },
          ),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(102),
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 4, 12, 6),
                child: SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(
                      value: 'personal',
                      icon: Icon(Icons.person_outline),
                      label: Text('Personal'),
                    ),
                    ButtonSegment(
                      value: 'home',
                      icon: Icon(Icons.home_outlined),
                      label: Text('Home'),
                    ),
                  ],
                  selected: {_memoryDomain},
                  onSelectionChanged: (value) =>
                      _changeMemoryDomain(value.first),
                  showSelectedIcon: false,
                ),
              ),
              TabBar(
                controller: _tabs,
                indicatorColor: _mint,
                labelColor: _mint,
                unselectedLabelColor: _muted,
                tabs: const [
                  Tab(icon: Icon(Icons.search), text: 'Search'),
                  Tab(icon: Icon(Icons.timeline), text: 'Timeline'),
                  Tab(icon: Icon(Icons.hub_outlined), text: 'Entities'),
                ],
              ),
            ],
          ),
        ),
      ),
      body: TabBarView(
        controller: _tabs,
        children: [_searchTab(), _timelineTab(), _entitiesTab()],
      ),
    );
  }

  Widget _searchTab() {
    return Column(
      children: [
        Container(
          color: _panel,
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
          child: Column(
            children: [
              TextField(
                controller: _query,
                autofocus: false,
                textInputAction: TextInputAction.search,
                onSubmitted: (_) => _search(),
                style: const TextStyle(color: Colors.white),
                decoration: InputDecoration(
                  hintText: 'Search activity, notes, people, claims, rooms…',
                  prefixIcon: const Icon(Icons.search),
                  suffixIcon: IconButton(
                    icon: const Icon(Icons.arrow_forward, color: _mint),
                    onPressed: _search,
                  ),
                ),
              ),
              const SizedBox(height: 8),
              SizedBox(
                height: 34,
                child: ListView(
                  scrollDirection: Axis.horizontal,
                  children: [
                    for (final kind
                        in ['event', 'note', 'message', 'entity', 'claim', 'room'])
                      Padding(
                        padding: const EdgeInsets.only(right: 6),
                        child: FilterChip(
                          label: Text(kind),
                          selected: _searchKinds.contains(kind),
                          onSelected: (selected) {
                            setState(() {
                              if (selected) {
                                _searchKinds.add(kind);
                              } else if (_searchKinds.length > 1) {
                                _searchKinds.remove(kind);
                              }
                            });
                            if (_query.text.trim().isNotEmpty) _search();
                          },
                          selectedColor: _mint,
                          backgroundColor: _panelRaised,
                          labelStyle: TextStyle(
                              color: _searchKinds.contains(kind) ? _ink : _muted,
                              fontSize: 11),
                          showCheckmark: false,
                          side: const BorderSide(color: _line),
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 7),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _pickSearchRange,
                      icon: const Icon(Icons.date_range, size: 16),
                      label: Text(
                        _searchRange == null
                            ? 'Any date'
                            : '${_dateIso(_searchRange!.start)} → ${_dateIso(_searchRange!.end)}',
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: DropdownButtonFormField<String?>(
                      initialValue: _searchRoomId,
                      isExpanded: true,
                      decoration: const InputDecoration(
                          isDense: true, labelText: 'Room'),
                      dropdownColor: _panelRaised,
                      items: [
                        const DropdownMenuItem<String?>(
                            value: null, child: Text('All rooms')),
                        ..._rooms
                            .where((room) => _memoryDomain == 'home'
                                ? room['kind'] == 'camera'
                                : room['kind'] != 'camera')
                            .map((room) => DropdownMenuItem<String?>(
                                  value: room['room_id'].toString(),
                                  child: Text(room['name'].toString(),
                                      overflow: TextOverflow.ellipsis),
                                )),
                      ],
                      onChanged: (value) {
                        setState(() => _searchRoomId = value);
                        if (_query.text.trim().isNotEmpty) _search();
                      },
                    ),
                  ),
                  if (_searchRange != null)
                    IconButton(
                      tooltip: 'Clear date range',
                      icon: const Icon(Icons.close, color: _muted),
                      onPressed: () {
                        setState(() => _searchRange = null);
                        if (_query.text.trim().isNotEmpty) _search();
                      },
                    ),
                ],
              ),
            ],
          ),
        ),
        if (_searching)
          const LinearProgressIndicator(minHeight: 2, color: _mint),
        Expanded(
          child: _searchError != null
              ? _empty('Search failed.\n$_searchError')
              : _results.isEmpty
                  ? _empty(_query.text.trim().isEmpty
                      ? (_memoryDomain == 'personal'
                          ? 'Search PC activity, work, and personal assistant context.'
                          : 'Search cameras, rooms, and household observations.')
                      : 'No matching memories.')
                  : ListView.builder(
                      padding: const EdgeInsets.all(12),
                      itemCount: _results.length,
                      itemBuilder: (_, index) => _searchResult(
                          _results[index] as Map<String, dynamic>),
                    ),
        ),
      ],
    );
  }

  Future<void> _pickSearchRange() async {
    final value = await showDateRangePicker(
      context: context,
      firstDate: DateTime(2020),
      lastDate: DateTime.now().add(const Duration(days: 1)),
      initialDateRange: _searchRange,
    );
    if (value == null) return;
    setState(() => _searchRange = value);
    if (_query.text.trim().isNotEmpty) _search();
  }

  Widget _searchResult(Map<String, dynamic> item) {
    final kind = (item['kind'] ?? 'memory').toString();
    final rooms = (item['rooms'] as List?) ?? [];
    return Card(
      color: _panelRaised,
      margin: const EdgeInsets.only(bottom: 9),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () => _openResult(item),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(_kindIcon(kind), color: kind == 'entity' ? _violet : _mint),
              const SizedBox(width: 11),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text((item['title'] ?? kind).toString(),
                              style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w700)),
                        ),
                        Text(kind,
                            style:
                                const TextStyle(color: _muted, fontSize: 10)),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text((item['text'] ?? '').toString(),
                        maxLines: 4,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                            color: Colors.white70, fontSize: 13, height: 1.3)),
                    if (item['ts'] != null || rooms.isNotEmpty) ...[
                      const SizedBox(height: 7),
                      Text(
                        [
                          if (item['ts'] != null) _dayTime(item['ts']),
                          if (rooms.isNotEmpty)
                            rooms
                                .map((room) => room['name'])
                                .where((name) => name != null)
                                .join(', '),
                        ].join(' · '),
                        style: const TextStyle(color: _muted, fontSize: 11),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _openResult(Map<String, dynamic> item) {
    final kind = item['kind']?.toString();
    final id = item['id']?.toString();
    if (id == null) return;
    if (kind == 'event') {
      _openEvent(id);
    } else if (kind == 'entity') {
      _openEntity(id);
    } else {
      showModalBottomSheet(
        context: context,
        backgroundColor: _panel,
        showDragHandle: true,
        builder: (_) => Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text((item['title'] ?? kind).toString(),
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w700)),
              const SizedBox(height: 12),
              Text((item['text'] ?? '').toString(),
                  style: const TextStyle(color: Colors.white70, height: 1.4)),
            ],
          ),
        ),
      );
    }
  }

  Widget _timelineTab() {
    final needle = _timelineFilter.text.trim().toLowerCase();
    final sessions = _sessions.where((raw) {
      if (needle.isEmpty) return true;
      final session = raw as Map<String, dynamic>;
      return [
        session['application'],
        session['activity_type'],
        session['project_id'],
        ...((session['events'] as List?) ?? [])
            .map((event) => event['summary']),
      ].join(' ').toLowerCase().contains(needle);
    }).toList();
    return Column(
      children: [
        _dateBar(),
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 8, 12, 4),
          child: TextField(
            controller: _timelineFilter,
            onChanged: (_) => setState(() {}),
            style: const TextStyle(color: Colors.white, fontSize: 13),
            decoration: const InputDecoration(
                isDense: true,
                prefixIcon: Icon(Icons.filter_list),
                hintText: 'Filter this day'),
          ),
        ),
        if (_timelineLoading)
          const LinearProgressIndicator(minHeight: 2, color: _mint),
        Expanded(
          child: _timelineError != null
              ? _empty('Could not load timeline.\n$_timelineError')
              : sessions.isEmpty
                  ? _empty(_memoryDomain == 'personal'
                      ? 'No personal activity recorded for this day.'
                      : 'No home observations recorded for this day.')
                  : ListView.builder(
                      padding: const EdgeInsets.fromLTRB(12, 8, 12, 24),
                      itemCount: sessions.length,
                      itemBuilder: (_, index) => _sessionCard(
                          sessions[index] as Map<String, dynamic>),
                    ),
        ),
      ],
    );
  }

  Widget _dateBar() {
    return Container(
      color: _panel,
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: Row(
        children: [
          IconButton(
              icon: const Icon(Icons.chevron_left, color: _muted),
              onPressed: () => _changeDay(-1)),
          Expanded(
            child: InkWell(
              onTap: _pickDay,
              child: Text(_dateIso(_date),
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                      color: Colors.white, fontWeight: FontWeight.w700)),
            ),
          ),
          IconButton(
              icon: const Icon(Icons.chevron_right, color: _muted),
              onPressed: () => _changeDay(1)),
          PopupMenuButton<String>(
            icon: const Icon(Icons.more_vert, color: _muted),
            onSelected: (value) {
              if (value == 'forget') _forgetDay();
            },
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'forget', child: Text('Forget this day')),
            ],
          ),
        ],
      ),
    );
  }

  Widget _sessionCard(Map<String, dynamic> session) {
    final activity = (session['activity_type'] ?? '?').toString();
    final app = (session['application'] ?? '?').toString();
    final events = (session['events'] as List?) ?? [];
    final mins = (((session['active_seconds'] as num?) ?? 0) / 60).round();
    return Card(
      color: _panelRaised,
      margin: const EdgeInsets.only(bottom: 10),
      child: ExpansionTile(
        initiallyExpanded: true,
        iconColor: _mint,
        collapsedIconColor: _muted,
        leading: const Icon(Icons.desktop_windows, color: _mint),
        title: Text(app,
            style: const TextStyle(
                color: Colors.white, fontWeight: FontWeight.w700)),
        subtitle: Text('$activity · $mins min',
            style: const TextStyle(color: _muted, fontSize: 12)),
        trailing: PopupMenuButton<String>(
          icon: const Icon(Icons.more_vert, color: _muted),
          onSelected: (_) =>
              _forgetSession(session['session_id']?.toString() ?? ''),
          itemBuilder: (_) => const [
            PopupMenuItem(value: 'forget', child: Text('Forget session')),
          ],
        ),
        children: [
          for (final event in events)
            ListTile(
              dense: true,
              title: Text((event['summary'] ?? '(no summary)').toString(),
                  style: const TextStyle(color: Colors.white70, fontSize: 13)),
              subtitle: Text(
                  '${_hm(event['span_start'])}–${_hm(event['span_end'])}',
                  style: const TextStyle(color: _mint, fontSize: 11)),
              onTap: () => _openEvent(event['event_id'].toString()),
            ),
        ],
      ),
    );
  }

  Widget _entitiesTab() {
    return Column(
      children: [
        _dateBar(),
        if (_entitiesLoading)
          const LinearProgressIndicator(minHeight: 2, color: _mint),
        Expanded(
          child: _entities.isEmpty && !_entitiesLoading
              ? _empty(_memoryDomain == 'personal'
                  ? 'No personal-memory entities for this day.'
                  : 'No home-memory entities for this day.')
              : ListView.separated(
                  padding: const EdgeInsets.all(12),
                  itemCount: _entities.length,
                  separatorBuilder: (_, __) => const Divider(color: _line),
                  itemBuilder: (_, index) {
                    final entity =
                        _entities[index] as Map<String, dynamic>;
                    return ListTile(
                      leading: const CircleAvatar(
                          backgroundColor: _panelRaised,
                          child: Icon(Icons.hub_outlined, color: _violet)),
                      title: Text(entity['name'].toString(),
                          style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w700)),
                      subtitle: Text(
                          '${entity['type'] ?? 'entity'} · ${entity['mentions'] ?? 0} mentions',
                          style: const TextStyle(color: _muted)),
                      trailing:
                          const Icon(Icons.chevron_right, color: _muted),
                      onTap: () => _openEntity(
                          (entity['entity_id'] ?? entity['name'])
                              .toString()
                              .toLowerCase()),
                    );
                  },
                ),
        ),
      ],
    );
  }

  Widget _empty(String text) => Center(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Text(text,
              textAlign: TextAlign.center,
              style: const TextStyle(color: _muted, height: 1.4)),
        ),
      );

  Future<void> _openEvent(String eventId) async {
    final changed = await Navigator.of(context).push<bool>(MaterialPageRoute(
      builder: (_) => MemoryEventScreen(
        apiBase: widget.apiBase,
        eventId: eventId,
        memoryDomain: _memoryDomain,
      ),
    ));
    if (changed == true) {
      _loadTimeline();
      if (_query.text.trim().isNotEmpty) _search();
    }
  }

  Future<void> _openEntity(String entityId) async {
    final changed = await Navigator.of(context).push<bool>(MaterialPageRoute(
      builder: (_) => MemoryEntityScreen(
        apiBase: widget.apiBase,
        entityId: entityId,
        memoryDomain: _memoryDomain,
      ),
    ));
    if (changed == true) {
      _loadEntities();
      if (_query.text.trim().isNotEmpty) _search();
    }
  }

  Future<bool> _confirm(String title, String body) async {
    return await showDialog<bool>(
          context: context,
          builder: (_) => AlertDialog(
            title: Text(title),
            content: Text(body),
            actions: [
              TextButton(
                  onPressed: () => Navigator.pop(context, false),
                  child: const Text('Cancel')),
              FilledButton(
                  onPressed: () => Navigator.pop(context, true),
                  child: const Text('Forget')),
            ],
          ),
        ) ??
        false;
  }

  Future<void> _forgetSession(String sessionId) async {
    if (sessionId.isEmpty ||
        !await _confirm('Forget this session?',
            'All of its events and vector memories will be permanently removed.')) {
      return;
    }
    final resp = await http
        .delete(Uri.parse('${widget.apiBase}/memory/sessions/$sessionId'));
    if (resp.statusCode == 200) {
      _loadTimeline();
    } else {
      _snack('Could not forget session: HTTP ${resp.statusCode}');
    }
  }

  Future<void> _forgetDay() async {
    final day = _dateIso(_date);
    final label = _memoryDomain == 'personal' ? 'personal' : 'home';
    if (!await _confirm('Forget $label memory for $day?',
        'Only $label events and their vector memories for this day will be permanently removed.')) {
      return;
    }
    final uri = Uri.parse('${widget.apiBase}/memory/days/$day')
        .replace(queryParameters: {'domain': _memoryDomain});
    final resp = await http.delete(uri);
    if (resp.statusCode == 200) {
      _loadTimeline();
      _loadEntities();
    } else {
      _snack('Could not forget day: HTTP ${resp.statusCode}');
    }
  }
}

class MemoryEventScreen extends StatefulWidget {
  final String apiBase;
  final String eventId;
  final String? memoryDomain;
  const MemoryEventScreen({
    super.key,
    required this.apiBase,
    required this.eventId,
    this.memoryDomain,
  });

  @override
  State<MemoryEventScreen> createState() => _MemoryEventScreenState();
}

class _MemoryEventScreenState extends State<MemoryEventScreen> {
  Map<String, dynamic>? _event;
  bool _loading = true;
  String? _error;
  bool _changed = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final resp = await http.get(
          Uri.parse('${widget.apiBase}/memory/events/${widget.eventId}'));
      if (resp.statusCode != 200) throw Exception('HTTP ${resp.statusCode}');
      final data = json.decode(resp.body) as Map<String, dynamic>;
      setState(() {
        _event = data['event'] as Map<String, dynamic>;
        _error = null;
      });
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) {
        if (!didPop) Navigator.pop(context, _changed);
      },
      child: Scaffold(
        backgroundColor: _ink,
        appBar: AppBar(
          backgroundColor: _panel,
          title: const Text('Memory event'),
          actions: [
            IconButton(
                tooltip: 'Edit summary',
                icon: const Icon(Icons.edit, color: _mint),
                onPressed: _event == null ? null : _edit),
            IconButton(
                tooltip: 'Forget event',
                icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                onPressed: _event == null ? null : _forget),
          ],
        ),
        body: _loading
            ? const Center(child: CircularProgressIndicator(color: _mint))
            : _error != null
                ? Center(
                    child: Text(_error!,
                        style: const TextStyle(color: _muted)))
                : _body(),
      ),
    );
  }

  Widget _body() {
    final event = _event!;
    final entities = (event['entities'] as List?) ?? [];
    final claims = (event['claims'] as List?) ?? [];
    final rooms = (event['rooms'] as List?) ?? [];
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text((event['summary'] ?? '(no summary)').toString(),
            style: const TextStyle(
                color: Colors.white, fontSize: 18, height: 1.4)),
        const SizedBox(height: 10),
        Text(
            '${_dayTime(event['span_start'])}–${_hm(event['span_end'])} · '
            '${event['application'] ?? event['activity_type'] ?? ''}',
            style: const TextStyle(color: _muted)),
        if (event['original_summary'] != null) ...[
          const SizedBox(height: 12),
          _section('Original extraction'),
          Text(event['original_summary'].toString(),
              style: const TextStyle(color: _muted, fontStyle: FontStyle.italic)),
        ],
        _section('Rooms'),
        Wrap(
          spacing: 7,
          runSpacing: 7,
          children: rooms
              .map((room) => Chip(
                    label: Text(room['name'].toString()),
                    avatar: Icon(
                        room['manual'] == true ? Icons.lock : Icons.auto_awesome,
                        size: 15),
                  ))
              .toList(),
        ),
        _section('Entities'),
        if (entities.isEmpty)
          const Text('None', style: TextStyle(color: _muted))
        else
          ...entities.map((entity) => ListTile(
                contentPadding: EdgeInsets.zero,
                leading: const Icon(Icons.hub_outlined, color: _violet),
                title: Text(entity['name'].toString(),
                    style: const TextStyle(color: Colors.white)),
                subtitle: Text(entity['type'].toString(),
                    style: const TextStyle(color: _muted)),
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => MemoryEntityScreen(
                      apiBase: widget.apiBase,
                      entityId: entity['entity_id'].toString(),
                      memoryDomain: widget.memoryDomain,
                    ),
                  ),
                ),
              )),
        _section('Claims'),
        if (claims.isEmpty)
          const Text('None', style: TextStyle(color: _muted))
        else
          ...claims.map((claim) => ListTile(
                contentPadding: EdgeInsets.zero,
                leading:
                    const Icon(Icons.fact_check_outlined, color: _mint),
                title: Text(claim['text'].toString(),
                    style: const TextStyle(color: Colors.white70)),
                trailing: PopupMenuButton<String>(
                  onSelected: (value) => value == 'edit'
                      ? _editClaim(claim as Map<String, dynamic>)
                      : _deleteClaim(claim as Map<String, dynamic>),
                  itemBuilder: (_) => const [
                    PopupMenuItem(value: 'edit', child: Text('Correct')),
                    PopupMenuItem(value: 'delete', child: Text('Forget')),
                  ],
                ),
              )),
      ],
    );
  }

  Widget _section(String title) => Padding(
        padding: const EdgeInsets.only(top: 24, bottom: 8),
        child: Text(title,
            style: const TextStyle(
                color: _mint, fontWeight: FontWeight.w700, fontSize: 13)),
      );

  Future<void> _edit() async {
    final controller =
        TextEditingController(text: (_event!['summary'] ?? '').toString());
    if (!await _textDialog('Correct event summary', controller)) return;
    final resp = await http.patch(
      Uri.parse('${widget.apiBase}/memory/events/${widget.eventId}'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'summary': controller.text.trim()}),
    );
    if (resp.statusCode == 200) {
      _changed = true;
      _load();
    }
  }

  Future<void> _editClaim(Map<String, dynamic> claim) async {
    final controller = TextEditingController(text: claim['text'].toString());
    if (!await _textDialog('Correct claim', controller)) return;
    final resp = await http.patch(
      Uri.parse('${widget.apiBase}/memory/claims/${claim['claim_id']}'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'text': controller.text.trim()}),
    );
    if (resp.statusCode == 200) {
      _changed = true;
      _load();
    }
  }

  Future<void> _deleteClaim(Map<String, dynamic> claim) async {
    final resp = await http.delete(
        Uri.parse('${widget.apiBase}/memory/claims/${claim['claim_id']}'));
    if (resp.statusCode == 200) {
      _changed = true;
      _load();
    }
  }

  Future<bool> _textDialog(
      String title, TextEditingController controller) async {
    return await showDialog<bool>(
          context: context,
          builder: (_) => AlertDialog(
            title: Text(title),
            content:
                TextField(controller: controller, minLines: 3, maxLines: 8),
            actions: [
              TextButton(
                  onPressed: () => Navigator.pop(context, false),
                  child: const Text('Cancel')),
              FilledButton(
                  onPressed: () =>
                      Navigator.pop(context, controller.text.trim().isNotEmpty),
                  child: const Text('Save correction')),
            ],
          ),
        ) ??
        false;
  }

  Future<void> _forget() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Forget this event?'),
        content: const Text(
            'The graph event, its private claims and vector embedding will be removed.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Forget')),
        ],
      ),
    );
    if (confirmed != true) return;
    final resp = await http.delete(
        Uri.parse('${widget.apiBase}/memory/events/${widget.eventId}'));
    if (resp.statusCode == 200 && mounted) {
      Navigator.pop(context, true);
    }
  }
}

class MemoryEntityScreen extends StatefulWidget {
  final String apiBase;
  final String entityId;
  final String? memoryDomain;
  const MemoryEntityScreen({
    super.key,
    required this.apiBase,
    required this.entityId,
    this.memoryDomain,
  });

  @override
  State<MemoryEntityScreen> createState() => _MemoryEntityScreenState();
}

class _MemoryEntityScreenState extends State<MemoryEntityScreen> {
  Map<String, dynamic>? _entity;
  bool _loading = true;
  String? _error;
  bool _changed = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final uri =
          Uri.parse('${widget.apiBase}/memory/entities/${widget.entityId}')
              .replace(queryParameters: {
        if (widget.memoryDomain != null) 'domain': widget.memoryDomain!,
      });
      final resp = await http.get(uri);
      if (resp.statusCode != 200) throw Exception('HTTP ${resp.statusCode}');
      final data = json.decode(resp.body) as Map<String, dynamic>;
      setState(() {
        _entity = data['entity'] as Map<String, dynamic>;
        _error = null;
      });
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) {
        if (!didPop) Navigator.pop(context, _changed);
      },
      child: Scaffold(
        backgroundColor: _ink,
        appBar: AppBar(
          backgroundColor: _panel,
          title: Text(_entity?['name']?.toString() ?? 'Entity'),
          actions: [
            IconButton(
                icon: const Icon(Icons.edit, color: _mint),
                onPressed: _entity == null ? null : _edit),
            PopupMenuButton<String>(
              enabled: _entity != null,
              icon: const Icon(Icons.call_split, color: _violet),
              onSelected: (value) =>
                  value == 'merge' ? _merge() : _split(),
              itemBuilder: (_) => const [
                PopupMenuItem(
                    value: 'merge', child: Text('Merge into another entity')),
                PopupMenuItem(
                    value: 'split', child: Text('Split selected events')),
              ],
            ),
            IconButton(
                icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                onPressed: _entity == null ? null : _forget),
          ],
        ),
        body: _loading
            ? const Center(child: CircularProgressIndicator(color: _mint))
            : _error != null
                ? Center(
                    child: Text(_error!,
                        style: const TextStyle(color: _muted)))
                : _body(),
      ),
    );
  }

  Widget _body() {
    final entity = _entity!;
    final events = (entity['events'] as List?) ?? [];
    final rooms = (entity['rooms'] as List?) ?? [];
    final related = (entity['co_occurring'] as List?) ?? [];
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Row(
          children: [
            const CircleAvatar(
                radius: 28,
                backgroundColor: _panelRaised,
                child: Icon(Icons.hub_outlined, color: _violet, size: 28)),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(entity['name'].toString(),
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.w700)),
                  Text(
                      '${entity['type'] ?? 'entity'} · ${entity['mentions'] ?? 0} events · ${entity['memory_status'] ?? 'unknown'}',
                      style: const TextStyle(color: _muted)),
                ],
              ),
            ),
          ],
        ),
        _heading('Rooms'),
        Wrap(
          spacing: 7,
          children: rooms
              .map((room) => Chip(
                  label: Text('${room['name']} (${room['events']})')))
              .toList(),
        ),
        _heading('Related entities'),
        Wrap(
          spacing: 7,
          runSpacing: 7,
          children: related
              .map((item) => ActionChip(
                    label: Text(item['name'].toString()),
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => MemoryEntityScreen(
                          apiBase: widget.apiBase,
                          entityId: item['name'].toString().toLowerCase(),
                          memoryDomain: widget.memoryDomain,
                        ),
                      ),
                    ),
                  ))
              .toList(),
        ),
        _heading('Event history'),
        ...events.map((event) => Card(
              color: _panelRaised,
              child: ListTile(
                title: Text((event['summary'] ?? '(no summary)').toString(),
                    style: const TextStyle(color: Colors.white70)),
                subtitle: Text(_dayTime(event['span_start']),
                    style: const TextStyle(color: _mint, fontSize: 11)),
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => MemoryEventScreen(
                        apiBase: widget.apiBase,
                        eventId: event['event_id'].toString()),
                  ),
                ),
              ),
            )),
      ],
    );
  }

  Widget _heading(String text) => Padding(
        padding: const EdgeInsets.only(top: 24, bottom: 8),
        child: Text(text,
            style: const TextStyle(
                color: _mint, fontWeight: FontWeight.w700)),
      );

  Future<void> _edit() async {
    final name = TextEditingController(text: _entity!['name'].toString());
    final type = TextEditingController(text: _entity!['type'].toString());
    final save = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Correct entity'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(controller: name, decoration: const InputDecoration(labelText: 'Name')),
            const SizedBox(height: 10),
            TextField(controller: type, decoration: const InputDecoration(labelText: 'Type')),
          ],
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Save correction')),
        ],
      ),
    );
    if (save != true || name.text.trim().isEmpty || type.text.trim().isEmpty) {
      return;
    }
    final resp = await http.patch(
      Uri.parse('${widget.apiBase}/memory/entities/${widget.entityId}'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'name': name.text.trim(), 'type': type.text.trim()}),
    );
    if (resp.statusCode == 200) {
      _changed = true;
      _load();
    }
  }

  Future<void> _forget() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('Forget ${_entity!['name']}?'),
        content: const Text(
            'The entity and its links will be removed. Source events remain.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Forget')),
        ],
      ),
    );
    if (confirmed != true) return;
    final resp = await http.delete(
        Uri.parse('${widget.apiBase}/memory/entities/${widget.entityId}'));
    if (resp.statusCode == 200 && mounted) Navigator.pop(context, true);
  }

  Future<void> _merge() async {
    final target = TextEditingController();
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Merge into entity'),
        content: TextField(
          controller: target,
          decoration: const InputDecoration(
              labelText: 'Target entity ID',
              helperText: 'Shown in search results or entity URLs'),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () =>
                  Navigator.pop(context, target.text.trim().isNotEmpty),
              child: const Text('Merge')),
        ],
      ),
    );
    if (confirmed != true) return;
    final resp = await http.post(
      Uri.parse(
          '${widget.apiBase}/memory/entities/${widget.entityId}/merge'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'target_id': target.text.trim()}),
    );
    if (resp.statusCode == 200 && mounted) {
      Navigator.pop(context, true);
    }
  }

  Future<void> _split() async {
    final events = ((_entity!['events'] as List?) ?? [])
        .cast<Map<String, dynamic>>();
    if (events.isEmpty) return;
    final selected = <String>{};
    final name = TextEditingController();
    final type = TextEditingController(text: _entity!['type'].toString());
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('Split entity'),
          content: SizedBox(
            width: 520,
            child: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  TextField(
                      controller: name,
                      decoration:
                          const InputDecoration(labelText: 'New entity name')),
                  const SizedBox(height: 8),
                  TextField(
                      controller: type,
                      decoration:
                          const InputDecoration(labelText: 'Entity type')),
                  const SizedBox(height: 12),
                  const Align(
                    alignment: Alignment.centerLeft,
                    child: Text('Move these event mentions:',
                        style: TextStyle(fontWeight: FontWeight.w700)),
                  ),
                  for (final event in events)
                    CheckboxListTile(
                      dense: true,
                      value: selected.contains(event['event_id'].toString()),
                      title: Text(
                        (event['summary'] ?? '(no summary)').toString(),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      onChanged: (checked) => setDialogState(() {
                        final id = event['event_id'].toString();
                        checked == true
                            ? selected.add(id)
                            : selected.remove(id);
                      }),
                    ),
                ],
              ),
            ),
          ),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: const Text('Cancel')),
            FilledButton(
                onPressed: () => Navigator.pop(
                    context,
                    name.text.trim().isNotEmpty &&
                        type.text.trim().isNotEmpty &&
                        selected.isNotEmpty),
                child: const Text('Split')),
          ],
        ),
      ),
    );
    if (confirmed != true) return;
    final resp = await http.post(
      Uri.parse(
          '${widget.apiBase}/memory/entities/${widget.entityId}/split'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'name': name.text.trim(),
        'type': type.text.trim(),
        'event_ids': selected.toList(),
      }),
    );
    if (resp.statusCode == 200) {
      _changed = true;
      _load();
    }
  }
}

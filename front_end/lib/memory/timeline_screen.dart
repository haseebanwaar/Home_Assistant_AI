import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// Memory Timeline — shows a day's sessions (per app/project) and the events
/// within them, from the backend memory graph (GET /memory/timeline).
class MemoryTimelineScreen extends StatefulWidget {
  final String apiBase;
  const MemoryTimelineScreen({super.key, required this.apiBase});

  @override
  State<MemoryTimelineScreen> createState() => _MemoryTimelineScreenState();
}

class _MemoryTimelineScreenState extends State<MemoryTimelineScreen> {
  static const _ink = Color(0xFF070B14);
  static const _panel = Color(0xFF111827);
  static const _panelRaised = Color(0xFF182235);
  static const _line = Color(0xFF263246);
  static const _mint = Color(0xFF6EE7D8);
  static const _muted = Color(0xFF91A0B8);

  DateTime _date = DateTime.now();
  bool _loading = false;
  String? _error;
  List<dynamic> _sessions = [];

  String get _dateIso =>
      '${_date.year.toString().padLeft(4, '0')}-'
      '${_date.month.toString().padLeft(2, '0')}-'
      '${_date.day.toString().padLeft(2, '0')}';

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
          .get(Uri.parse('${widget.apiBase}/memory/timeline?date=$_dateIso'))
          .timeout(const Duration(seconds: 15));
      if (resp.statusCode == 200) {
        final data = json.decode(resp.body) as Map<String, dynamic>;
        setState(() => _sessions = (data['sessions'] as List?) ?? []);
      } else {
        setState(() => _error = 'HTTP ${resp.statusCode}');
      }
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _shiftDay(int delta) {
    setState(() => _date = _date.add(Duration(days: delta)));
    _load();
  }

  String _hm(dynamic ts) {
    if (ts is! num) return '--:--';
    final dt = DateTime.fromMillisecondsSinceEpoch((ts * 1000).round());
    return '${dt.hour.toString().padLeft(2, '0')}:'
        '${dt.minute.toString().padLeft(2, '0')}';
  }

  String _mins(dynamic sec) {
    final m = ((sec is num) ? sec : 0) / 60;
    return '${m.toStringAsFixed(0)} min';
  }

  IconData _iconFor(String activity) {
    switch (activity) {
      case 'coding':
        return Icons.code;
      case 'browsing':
        return Icons.public;
      case 'reading':
        return Icons.menu_book;
      case 'watching':
        return Icons.play_circle_outline;
      case 'writing':
        return Icons.edit_note;
      case 'communication':
        return Icons.chat_bubble_outline;
      case 'terminal':
        return Icons.terminal;
      default:
        return Icons.desktop_windows;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _ink,
      appBar: AppBar(
        backgroundColor: _panel,
        title: const Text('Memory Timeline'),
        actions: [
          IconButton(
            tooltip: 'Refresh',
            icon: const Icon(Icons.refresh, color: _mint),
            onPressed: _loading ? null : _load,
          ),
        ],
      ),
      body: Column(
        children: [
          _dateBar(),
          if (_loading) const LinearProgressIndicator(minHeight: 2, color: _mint),
          Expanded(child: _content()),
        ],
      ),
    );
  }

  Widget _dateBar() {
    return Container(
      color: _panel.withOpacity(.5),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          IconButton(
            icon: const Icon(Icons.chevron_left, color: _muted),
            onPressed: () => _shiftDay(-1),
          ),
          Text(_dateIso,
              style: const TextStyle(
                  color: Colors.white, fontWeight: FontWeight.w700, fontSize: 15)),
          IconButton(
            icon: const Icon(Icons.chevron_right, color: _muted),
            onPressed: () => _shiftDay(1),
          ),
        ],
      ),
    );
  }

  Widget _content() {
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text('Could not load timeline.\n$_error',
              textAlign: TextAlign.center, style: const TextStyle(color: _muted)),
        ),
      );
    }
    if (_sessions.isEmpty && !_loading) {
      return const Center(
        child: Text('No recorded activity for this day.',
            style: TextStyle(color: _muted)),
      );
    }
    return ListView.builder(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 24),
      itemCount: _sessions.length,
      itemBuilder: (context, i) => _sessionCard(_sessions[i] as Map<String, dynamic>),
    );
  }

  Widget _sessionCard(Map<String, dynamic> s) {
    final activity = (s['activity_type'] ?? '?').toString();
    final app = (s['application'] ?? '?').toString();
    final project = s['project_id'];
    final resumes = (s['resume_count'] is num) ? (s['resume_count'] as num).toInt() : 0;
    final events = (s['events'] as List?) ?? [];

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: _panelRaised.withOpacity(.55),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _line),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          initiallyExpanded: true,
          tilePadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 2),
          childrenPadding: const EdgeInsets.only(left: 14, right: 14, bottom: 10),
          leading: Icon(_iconFor(activity), color: _mint),
          title: Text(app,
              style: const TextStyle(
                  color: Colors.white, fontWeight: FontWeight.w700)),
          subtitle: Text(
            '$activity · ${_mins(s['active_seconds'])}'
            '${resumes > 0 ? ' · $resumes resume(s)' : ''}'
            '${project != null ? ' · $project' : ''}',
            style: const TextStyle(color: _muted, fontSize: 12),
          ),
          children: [
            for (final e in events) _eventRow(e as Map<String, dynamic>),
          ],
        ),
      ),
    );
  }

  Widget _eventRow(Map<String, dynamic> e) {
    final summary = (e['summary'] ?? '').toString().trim();
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
            child: Text(
              '${_hm(e['span_start'])}–${_hm(e['span_end'])}',
              style: const TextStyle(
                  color: _mint, fontSize: 11, fontWeight: FontWeight.w600),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              summary.isEmpty ? '(no summary)' : summary,
              style: const TextStyle(color: Colors.white70, fontSize: 13, height: 1.3),
            ),
          ),
        ],
      ),
    );
  }
}

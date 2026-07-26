import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

const _ink = Color(0xFF070B14);
const _panel = Color(0xFF111827);
const _panelRaised = Color(0xFF182235);
const _line = Color(0xFF263246);
const _mint = Color(0xFF6EE7D8);
const _muted = Color(0xFF91A0B8);
const _critical = Color(0xFFFF607C);
const _important = Color(0xFFFFC857);

class NotificationsScreen extends StatefulWidget {
  final String apiBase;
  final ValueChanged<int>? onUnreadChanged;

  const NotificationsScreen({
    super.key,
    required this.apiBase,
    this.onUnreadChanged,
  });

  @override
  State<NotificationsScreen> createState() => _NotificationsScreenState();
}

class _NotificationsScreenState extends State<NotificationsScreen> {
  bool _loading = false;
  String _filter = 'all';
  String? _error;
  List<Map<String, dynamic>> _items = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  List<Map<String, dynamic>> get _visible => _items.where((item) {
        return _filter == 'all' || item['severity'] == _filter;
      }).toList();

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final response = await http
          .get(Uri.parse('${widget.apiBase}/notifications?limit=200'))
          .timeout(const Duration(seconds: 10));
      if (response.statusCode != 200) {
        throw Exception('HTTP ${response.statusCode}');
      }
      final data = json.decode(response.body) as Map<String, dynamic>;
      final items = ((data['notifications'] as List?) ?? const [])
          .cast<Map<String, dynamic>>();
      if (mounted) {
        setState(() => _items = items);
        widget.onUnreadChanged
            ?.call((data['unread_count'] as num?)?.toInt() ?? 0);
      }
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _markRead(Map<String, dynamic> item) async {
    if (item['read'] == true) return;
    final id = item['id']?.toString();
    if (id == null) return;
    final response = await http
        .post(Uri.parse('${widget.apiBase}/notifications/$id/read'))
        .timeout(const Duration(seconds: 8));
    if (response.statusCode == 200) await _load();
  }

  Future<void> _markAllRead() async {
    final response = await http
        .post(Uri.parse('${widget.apiBase}/notifications/actions/read-all'))
        .timeout(const Duration(seconds: 8));
    if (response.statusCode == 200) await _load();
  }

  @override
  Widget build(BuildContext context) {
    final visible = _visible;
    return Scaffold(
      backgroundColor: _ink,
      appBar: AppBar(
        backgroundColor: _panel,
        title: const Text('Notifications'),
        actions: [
          TextButton(
              onPressed: _items.any((item) => item['read'] != true)
                  ? _markAllRead
                  : null,
              child: const Text('Mark all read')),
          IconButton(
              tooltip: 'Refresh',
              onPressed: _loading ? null : _load,
              icon: const Icon(Icons.refresh, color: _mint)),
        ],
      ),
      body: Column(
        children: [
          if (_loading)
            const LinearProgressIndicator(minHeight: 2, color: _mint),
          _summary(),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 4),
            child: Row(
              children: [
                _filterChip('All', 'all', _mint),
                const SizedBox(width: 7),
                _filterChip('Critical', 'critical', _critical),
                const SizedBox(width: 7),
                _filterChip('Important', 'important', _important),
              ],
            ),
          ),
          Expanded(
            child: _error != null
                ? Center(
                    child: Text('Could not load notifications.\n$_error',
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: _muted)))
                : visible.isEmpty
                    ? const Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.notifications_none,
                                size: 44, color: _muted),
                            SizedBox(height: 10),
                            Text('Nothing needs your attention',
                                style: TextStyle(color: Colors.white70)),
                            SizedBox(height: 4),
                            Text(
                                'Critical and high-value events will appear here.',
                                style:
                                    TextStyle(color: _muted, fontSize: 11)),
                          ],
                        ),
                      )
                    : ListView.builder(
                        padding: const EdgeInsets.all(12),
                        itemCount: visible.length,
                        itemBuilder: (_, index) =>
                            _notificationCard(visible[index]),
                      ),
          ),
        ],
      ),
    );
  }

  Widget _summary() {
    final critical =
        _items.where((item) => item['severity'] == 'critical').length;
    final important =
        _items.where((item) => item['severity'] == 'important').length;
    final unread = _items.where((item) => item['read'] != true).length;
    return Container(
      width: double.infinity,
      color: _panelRaised,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
      child: Text(
        '$unread unread • $critical critical • $important important',
        style: const TextStyle(
            color: Colors.white70, fontSize: 12, fontWeight: FontWeight.w600),
      ),
    );
  }

  Widget _filterChip(String label, String value, Color color) {
    final selected = _filter == value;
    return ChoiceChip(
      selected: selected,
      showCheckmark: false,
      label: Text(label),
      selectedColor: color,
      backgroundColor: _panelRaised,
      side: const BorderSide(color: _line),
      labelStyle:
          TextStyle(color: selected ? _ink : _muted, fontSize: 11),
      onSelected: (_) => setState(() => _filter = value),
    );
  }

  Widget _notificationCard(Map<String, dynamic> item) {
    final isCritical = item['severity'] == 'critical';
    final color = isCritical ? _critical : _important;
    final read = item['read'] == true;
    final timestamp = item['timestamp'];
    final date = timestamp is num
        ? DateTime.fromMillisecondsSinceEpoch((timestamp * 1000).round())
        : null;
    final when = date == null
        ? ''
        : '${date.month}/${date.day} '
            '${date.hour.toString().padLeft(2, '0')}:'
            '${date.minute.toString().padLeft(2, '0')}';
    return InkWell(
      onTap: () => _markRead(item),
      borderRadius: BorderRadius.circular(14),
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.all(13),
        decoration: BoxDecoration(
          color: read ? _panel : color.withValues(alpha: .09),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
              color: read ? _line : color.withValues(alpha: .48)),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                  color: color.withValues(alpha: .14),
                  borderRadius: BorderRadius.circular(10)),
              child: Icon(
                  isCritical
                      ? Icons.warning_amber_rounded
                      : Icons.notifications_active_outlined,
                  color: color,
                  size: 20),
            ),
            const SizedBox(width: 11),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text((item['title'] ?? '').toString(),
                            style: TextStyle(
                                color: Colors.white,
                                fontSize: 12.5,
                                fontWeight:
                                    read ? FontWeight.w600 : FontWeight.w800)),
                      ),
                      Text(when,
                          style:
                              const TextStyle(color: _muted, fontSize: 10)),
                    ],
                  ),
                  const SizedBox(height: 5),
                  Text((item['body'] ?? '').toString(),
                      style: const TextStyle(
                          color: Colors.white70, fontSize: 12, height: 1.35)),
                  const SizedBox(height: 7),
                  Text(
                    '${(item['category'] ?? 'activity').toString()}'
                    ' • ${(item['source'] ?? 'screen').toString()}',
                    style: TextStyle(color: color, fontSize: 10.5),
                  ),
                ],
              ),
            ),
            if (!read)
              Container(
                width: 7,
                height: 7,
                margin: const EdgeInsets.only(left: 7, top: 3),
                decoration:
                    BoxDecoration(color: color, shape: BoxShape.circle),
              ),
          ],
        ),
      ),
    );
  }
}

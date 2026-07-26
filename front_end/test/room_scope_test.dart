import 'package:flutter_test/flutter_test.dart';
import 'package:untitled/rooms/rooms_screen.dart';

void main() {
  // A Thursday afternoon, so weekday arithmetic is actually exercised.
  final now = DateTime(2026, 7, 23, 14, 30);

  double? secondsOf(DateTime value) => value.millisecondsSinceEpoch / 1000.0;

  group('timeWindowBounds', () {
    test('no window selected means all of time', () {
      expect(timeWindowBounds(null, now: now), (null, null));
      expect(timeWindowBounds('nonsense', now: now), (null, null));
    });

    test('last hour starts exactly an hour back', () {
      final (start, end) = timeWindowBounds('hour', now: now);
      expect(start, secondsOf(DateTime(2026, 7, 23, 13, 30)));
      expect(end, isNull, reason: 'the window must stay open at the top');
    });

    test('today starts at local midnight', () {
      final (start, _) = timeWindowBounds('today', now: now);
      expect(start, secondsOf(DateTime(2026, 7, 23)));
    });

    test('this week starts on Monday, not seven days back', () {
      final (start, _) = timeWindowBounds('week', now: now);
      expect(start, secondsOf(DateTime(2026, 7, 20)));
    });

    test('a Monday is its own start of week', () {
      final monday = DateTime(2026, 7, 20, 9, 5);
      final (start, _) = timeWindowBounds('week', now: monday);
      expect(start, secondsOf(DateTime(2026, 7, 20)));
    });

    test('a Sunday still belongs to the week that began on Monday', () {
      final sunday = DateTime(2026, 7, 26, 23, 59);
      final (start, _) = timeWindowBounds('week', now: sunday);
      expect(start, secondsOf(DateTime(2026, 7, 20)));
    });

    test('this month starts on the first', () {
      final (start, _) = timeWindowBounds('month', now: now);
      expect(start, secondsOf(DateTime(2026, 7, 1)));
    });

    test('windows widen in order: hour < today < week < month', () {
      final starts = ['hour', 'today', 'week', 'month']
          .map((w) => timeWindowBounds(w, now: now).$1!)
          .toList();
      final sorted = [...starts]..sort((a, b) => b.compareTo(a));
      expect(starts, sorted, reason: 'each window should reach further back');
    });
  });

  group('dayBounds', () {
    test('covers exactly one day', () {
      final (start, end) = dayBounds('2026-07-25');
      expect(start, secondsOf(DateTime(2026, 7, 25)));
      expect(end, secondsOf(DateTime(2026, 7, 26)));
      expect(end! - start!, 86400);
    });
  });
}

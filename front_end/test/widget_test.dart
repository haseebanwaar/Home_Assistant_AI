import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:untitled/main.dart';

void main() {
  Future<void> pumpAt(WidgetTester tester, Size size) async {
    tester.view.physicalSize = size;
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const HomeMindApp());
    await tester.pump();
  }

  testWidgets('desktop workspace shows persistent navigation', (
    WidgetTester tester,
  ) async {
    await pumpAt(tester, const Size(1440, 900));

    expect(find.text('Personal workspace'), findsOneWidget);
    expect(find.text('Home'), findsWidgets);
    expect(find.text('Assistant'), findsOneWidget);
    expect(find.text('Rooms'), findsWidgets);
    expect(find.text('Memory'), findsWidgets);
    expect(find.text('Initiative & alerts'), findsOneWidget);
    expect(find.text('Settings'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('mobile home exposes workspace menu without overflow', (
    WidgetTester tester,
  ) async {
    await pumpAt(tester, const Size(390, 844));

    expect(find.byIcon(Icons.menu_rounded), findsOneWidget);
    expect(find.text('Your home, in sync'), findsOneWidget);
    final promptsButton = find.byKey(
      const Key('mobile-guided-reflection-button'),
    );
    expect(promptsButton, findsOneWidget);
    await tester.tap(promptsButton);
    await tester.pumpAndSettle();
    expect(find.text('Guided reflection'), findsOneWidget);
    expect(find.text('Do you agree?'), findsOneWidget);
    final challengeAction = find.byKey(
      const ValueKey('prompt-action-challenge_assumptions'),
    );
    await tester.drag(find.text('Do you agree?'), const Offset(0, -600));
    await tester.pump(const Duration(milliseconds: 300));
    expect(challengeAction, findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  // The phone home screen used to fit everything on screen by squeezing it,
  // which clipped content outright once the screen was short — a landscape
  // phone, or a portrait one with the keyboard up. It scrolls now, so no size
  // may produce an overflow.
  const crampedSizes = <String, Size>{
    'small phone': Size(320, 568),
    'common phone': Size(360, 640),
    'landscape phone': Size(844, 390),
    'keyboard open': Size(390, 420),
  };
  crampedSizes.forEach((label, size) {
    testWidgets('mobile home and drawer fit on a $label', (
      WidgetTester tester,
    ) async {
      final errors = <Object>[];
      final previous = FlutterError.onError;
      FlutterError.onError = (details) => errors.add(details.exception);
      addTearDown(() => FlutterError.onError = previous);

      await pumpAt(tester, size);
      expect(errors, isEmpty, reason: '$label home');

      await tester.tap(find.byIcon(Icons.menu_rounded));
      await tester.pumpAndSettle();
      expect(errors, isEmpty, reason: '$label drawer');
    });
  });
}

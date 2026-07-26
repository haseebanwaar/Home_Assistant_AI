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

  testWidgets('desktop workspace shows persistent navigation',
      (WidgetTester tester) async {
    await pumpAt(tester, const Size(1440, 900));

    expect(find.text('Personal workspace'), findsOneWidget);
    expect(find.text('Home'), findsWidgets);
    expect(find.text('Assistant'), findsOneWidget);
    expect(find.text('Rooms'), findsWidgets);
    expect(find.text('Memory'), findsWidgets);
    expect(find.text('Initiative & alerts'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('mobile home exposes workspace menu without overflow',
      (WidgetTester tester) async {
    await pumpAt(tester, const Size(390, 844));

    expect(find.byIcon(Icons.menu_rounded), findsOneWidget);
    expect(find.text('Your home, in sync'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}

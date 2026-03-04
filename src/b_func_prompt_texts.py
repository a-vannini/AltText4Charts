import pandas as pd
import os

#----------------------------------prompt text for alt_text generation---------------------------------

def alt_prompt_text_1(chart_info, csv_path, png_path):
    
    Event_length = 0
    LD_linie_simple = 50
    LD_linie_complex = 90
    LD_bar_simple = 50
    LD_bar_complex = 90
    LD_stacked_bar_simple = 50
    LD_stacked_bar_complex = 90

    if chart_info.get("events", "") != "":
        Event_length = 20

    title = chart_info.get("title", "").strip()
    subtitle = (chart_info.get("subtitle") or "").strip()
    chart_type = chart_info.get('chart_type').lower().strip()
    raw_is_complex = int(chart_info.get("complex"))
    is_complex = bool(raw_is_complex)

    print(f"chart type: {chart_type}")
    print(f"complexity: {is_complex}")
    
    try:
        df = pd.read_csv(csv_path)
        df_valid = not df.empty
    except Exception as e:
        raise ValueError(f"Fehler beim Einlesen der CSV-Datei: {e}")
    
    # NA's mit "-" ersetzen
    df.fillna("-", inplace=True)
    
    print(f"length data: {len(df)}")

    prompt = "Erstelle einen alternativen Beschreibungstext für ein Diagramm. \n\n"

    # === PROMPT-FORMULIERUNG + BEISPIELE ===
    if chart_type in ["line", "area"]:
        # Line, simple
        if not is_complex:
            print("Line, simple")
            prompt += (f"""
    **Kurzbeschreibung**:
    Diagrammtyp: Liniendiagramm. Titel: {title}. (Untertitel(falls vorhanden): {subtitle}). Waagerechte Achse: Zeitraum [von] bis [bis] [Einheit]. Senkrechte Achse: (Ausschnitt) [Min] bis [Max] [Einheit]. Daten: [täglich/wöchentlich/monatlich/jährlich] Werte.
    (Markierungen: falls vorhanden).
    **Überblick**:
    Trend ohne exakte Zahlen, z. B. steigt/sinkt/schwankt; erwähne Plateaus/Spitzen.
    **Lange Beschreibung**:
    Verlauf vom Anfangs- zum Endwert in Zeitabschnitten; nenne ungefähre Zeitpunkte von Hoch-/Tiefpunkten, Phasen (Anstieg/Rückgang/Plateau) und ggf. Markierungen.
    Lange Beschreibung maximal {LD_linie_simple + Event_length} Wörter.                
            """)

            prompt += ("""
    Beispiel 1:
    **Kurzbeschreibung**: Liniendiagramm. Titel: Abkehr von Treasuries. Untertitel: Rendite 10-jährige US-Staatsanleihen, in %. Waagrechte Achse: Zeitraum 2. Januar bis 4. Dezember 2017. Senkrechte Achse: Ausschnitt von 2 bis 3 Prozent. Daten: Täglich Werte.
    **Überblick**: Die Rendite sinkt bis im Herbst 2017 mit Schwankungen und steigt bis Ende Jahr auf das Anfangsniveau.
    **Lange Beschreibung**: Anfangs Januar 2017 liegt die Rendite bei rund 2.4 Prozent und steigt bis Mitte März auf den Jahreshöchststand von gut 2.6 Prozent. Anschliessend sinkt sie mit mehreren Zwischenanstiegen bis Anfang September auf den Tiefpunkt von gut 2 Prozent. Danach steigt sie bis Jahresende auf rund 2.4 Prozent.
                       
    Beispiel 2:
    **Kurzbeschreibung**: Liniendiagramm. Titel: Die illegalen Grenzübertritte haben unter Biden stark zugenommen. Untertitel: Zahl der monatlichen Aufgriffe an der Südgrenze der USA. Waagrechte Achse: Zeitraum von 2013 bis 2023. Senkrechte Achse: 0 bis 300'000 Aufgriffe. Daten: Monatliche Werte. Vier senkrechte Linien als Markierungen: 20.1.2017: Donald Trump wird Präsident. 20.3.2020: Die Pandemieregelung «Title 42» tritt in Kraft. 20.1.2021: Joe Biden übernimmt die Präsidentschaft. 11.5.2023: «Title 42» läuft aus, stattdessen beginnt Bidens Grenzregime.
    **Überblick**: Die monatlichen Aufgriffe bleiben mehrheitlich niedrig, bevor sie 2020 sprunghaft ansteigen. Dieses höhere Niveau hält bis zum Ende des Zeitraums 2023 an.
    **Lange Beschreibung**: 2013 bis 2018: Aufgriffe relativ niedrig, meist zwischen 25'000 und 60'000. Tiefstwert von gut 11'000 im April 2017 kurz nach Trumps Amtsantritt.
    2019 bis 2020: Anstieg auf 130'000 im Mai 2019, gefolgt von einem Rückgang auf rund 30'000 anfangs 2020. Nach Inkrafttreten von „Title 42“ im März 2020 steiget die Anzahl Aufgriffe bis Ende 2020 auf rund 74'000.
    2021 bis 2023: Nach Bidens Amtsübernahme im Januar 2021 steigen die Aufgriffe weiter auf über 200'000. Bis zum Ende (2023) bewegen sie sich überwiegend zwischen 150'000 und 270'000. Im Mai 2023 endet „Title 42“, und Bidens neues Grenzregime tritt in Kraft. Daraufhin steigen die Aufgriffe auf ihren Höchststand von fast 270'000, bevor sie bis Ende 2023 leicht sinken auf gut 240'000.
            """)
                            

        elif is_complex:
            # Line, complex
            print("Line, complex")
            prompt += (f"""
    **Kurzbeschreibung**:
    Diagrammtyp: Liniendiagramm. Titel: {title}. (Untertitel(falls vorhanden): {subtitle}). Waagerechte Achse: Zeitraum [von] bis [bis] [Einheit]. Senkrechte Achse: (Ausschnitt) [Min] bis [Max] [Einheit]. Daten: [täglich/wöchentlich/monatlich/jährlich] Werte. (Markierungen: falls vorhanden). [Linienzahl] Linien: [Linie A (Farbe)], [Linie B (Farbe)], …. (Farbe nur falls Linienzahl kleiner als 5 ist).
    **Überblick**:
    Beschreibe grob den Vergleich der Linien (Leittrend, auffällige Abweichungen, Kreuzungen, Rangfolgen oder Parallelverläufe).
    Verwende keine absoluten Zahlen; beschreibe relative Entwicklungen (z. B. „steigt stärker“, „bleibt konstant“, „fällt ab“).
    **Lange Beschreibung**:
    - Wenn alle Linien ähnlich verlaufen: Beschreibe Anfangs- und Endwerte jeder Linie, nenne Unterschiede.
    - Wenn Linien sich nur phasenweise ähneln (z. B. zu Beginn oder am Ende): Gliedere in Zeitabschnitte („von [Zeit] bis [Zeit]“) und beschreibe die Entwicklungen in diesen Phasen.
    - Wenn Linien sich stark unterscheiden: Beschreibe jede Linie einzeln; bei vielen Linien (>5) fasse ähnliche Trends zu Gruppen zusammen (z. B. „hohes Niveau“, „mittlere Gruppe“, „niedrig“).
    Lange Beschreibung maximal {LD_linie_complex + Event_length} Wörter.  
            """)

            prompt += ("""
    **Beispiel 1:**
    **Kurzbeschreibung**: Liniendiagramm. Titel: Die Inflation in der Schweiz stammt vor allem von den Importgütern. Untertitel: Landesindex der Konsumentenpreise (Dezember 2020 = 100). Waagrechte Achse: Zeitraum März 2020 bis März 2023. Senkrechte Achse: Ausschnitt Index von 98 bis 112. Daten: Monatliche Werte. Drei Linien: Total (blau), Importgüter (türkis), Inlandgüter (orange).
    **Überblick**: 2020 bleiben die Konsumentenpreise weitgehend stabil. Ab 2021 steigen die Indizes in unterschiedlichem Ausmass: Importgüter haben den grössten Anstieg, Inlandgüter den geringsten und der totale Konsumentenpreis liegt dazwischen.
    **Lange Beschreibung**: März 2020 bis März 2021: Konsumentenpreise aller drei Kategorien liegen zwischen 99 und 102 Punkten. Von März 2021 bis März 2023 entwicklen sich die Indizes wie folgt: Total: steigt von rund 101 auf rund 106. Importgüter: Anstieg von etwa 101 auf etwa 111 im Juni 2022 und bleiben dort bis am Schluss. Inlandgüter: Steigen von rund 100 auf gut 104.
                       
    **Beispiel 2:**
    **Kurzbeschreibung**: Liniendiagramm. Titel: Chinas Abhängigkeit von Ausfuhren schrumpft. Untertitel: Aussenhandelsintensität, in % des BIP im Vergleich. Waagrechte Achse: Zeitraum 1981 bis 2017. Senkrechte Achse: Ausschnitt von 20 bis 140 Prozent. Daten: Jährliche Werte. Acht Linien: China, Deutschland, USA, UK, JPN, CH, EU und Welt.
    **Überblick**: Die Aussenhandelsintensität steigt in allen Ländern sowie in der EU und weltweit. Die Schweiz bewegt sich zwischen 20 und 40 Prozent über den anderen Regionen. USA und Japan bewegen sich zuunterst. Die anderen Regionen liegen dazwischen.
    **Lange Beschreibung**:
    China: Beginnt 1980 bei rund 12 Prozent, steigt ab den 1990er-Jahren stark an, erreicht 2006 mit rund 64 Prozent den Höchstwert und sinkt danach bis 2017 auf etwa 40 Prozent.
    Deutschland: 1980 bei rund 40 Prozent, bleibt bis Anfang der 1990er-Jahre relativ stabil und steigt danach kontinuierlich auf rund 85 Prozent.
    USA: 1980 bei rund 20 Prozent und steigt moderat auf etwa 27 Prozent (tiefster Endwert).
    UK: 1980 bei knapp 50 Prozent und steigt leicht auf rund 60 Prozent.
    Japan: 1980 bei rund 27 Prozent und steigt auf rund 34 Prozent.
    CH: 1980 bei rund 96Prozent leichter Rückgang bis 1994 (knapp 77Prozent) und steigt auf knapp 120Prozent (höchster Endwert).
    EU: 1980 bei etwa 50 Prozent und steigt auf rund 85 Prozent.
    Welt: 1980 bei etwa 40 Prozent und steigt auf knapp 60 Prozent im 2017.
            """)
            
        else:
            raise ValueError("Unbekannte Datenstruktur – Verarbeitung nicht möglich.")

    elif chart_type in ["bar"]: 
        # Bar, simple
        if not is_complex:
            print("Bar, simple")
            prompt += (f"""
    **Kurzbeschreibung**: Diagrammtyp: (Senkrecht / Waagerecht) Balkendiagramm. Titel: {title}. Untertitel: {subtitle}. Waagrechte Achse:
    – Falls Zeitreihe: Zeitraum [von …] bis […], ggf. in [x-Jahres-]Abständen
    – Bei bis zu 6 Kategorien: [Liste der Kategorien]
    – Bei mehr als 10 Kategorien: [Anzahl] [Überbegriff, z. B. Länder / Regionen]
    Senkrechte Achse: (Ausschnitt) [Min] bis [Max] [Einheit]. (Markierungen: falls vorhanden)
    **Überblick**:
    - Bei Zeitreihen: Beschreibe den Verlauf der Balken (steigend / sinkend / schwankend) und markante Jahre oder Spitzen.
    - Bei Kategorien: Gib den Gesamteindruck wieder (Rangfolgen, Cluster oder auffällige Gruppen).
    Falls der Titel eine Aussage enthält, beziehe dich darauf. 
    **Lange Beschreibung**:
    - Zeitreihe / sortierte Kategorien (z. B. Alter): Nenne Anfangswert, Entwicklung (steigt / schwankt / sinkt; Spannweite) und Endwert.
    - Zeitreihe mit unterschiedlich großen / kleinen Werten: Beschreibe den Durchschnitt; nenne große (mit Jahr) und kleine (mit Jahr) Werte; erwähne ggf. hervorgehobene Balken.
    - Zeitreihe mit wenigen Balken und Markierungen: Nenne das Ereignis und den zugehörigen Wert.
    - Kategorien (nicht sortiert, ≤ 5): Formuliere „[Kategorie]: [Wert]“ für jede Kategorie.
    - Kategorien (sortiert, viele): Beschreibe von größter zu kleinster; fasse ähnliche Werte zu Gruppen („hoch“, „mittel“, „niedrig“) zusammen und präzisiere nur wesentliche Unterschiede.
    Lange Beschreibung maximal {LD_bar_simple + Event_length} Wörter.
            """)

            prompt += ("""
    **Beispiel 1:**
    **Kurzbeschreibung**: Senkrechtes Balkendiagramm. Titel: Besucher am Züri-Fäscht. Untertitel: (in Millionen). Waagrechte Achse: 1998 bis 2019 in Dreijahresabständen. Senkrechte Achse: 0 bis 2.5 Millionen Besucher.
    **Überblick**: Die Anzahl Besucher am Züri-Fäscht steigt.
    **Lange Beschreibung**: 1998 sind es 1.5 Millionen Besucher am Züri-Fäscht. In den folgenden Jahren steigt die Zahl stetig an und überschreitet 2007 die 2-Millionen-Marke. Danach schwanken die Werte zwischen 2 und 2.3 Millionen, bis 2019 die Besucherzahl auf gut 2.5 Millionen steigt, den Höchststand.
                       
    **Beispiel 2:**
    **Kurzbeschreibung**: Waagrechtes Balkendiagramm. Titel: Graubünden wurde von Postauto am stärksten geschröpft. Untertitel: Umgebuchter Gewinn in der Sparte "Regionaler Personenverkehr", nach Region, 2007 bis 2015, in Mio. Fr. "Umgebuchter Gewinn" meint vereinfacht gesagt den Betrag, den die Regionen über die eigentlichen Kosten der Firma Postauto hinaus zu viel an Subventionen bezahlt haben. Waagrechte Achse: Minus 5 bis 35 Millionen Franken. Senkrechte Achse: 16 Regionen.
    **Überblick**: Graubünden hat den höchsten umgebuchten Gewinn, vor Tessin und Aarau. Die meisten anderen Regionen zahlen deutlich weniger.
    **Lange Beschreibung**: Graubünden zahlt gut 30 Millionen Franken, Tessin 13 Millionen und Aargau etwa 12 Millionen. Bern, St. Gallen und Basel: Zwischen 5 und 8 Millionen. Sion, Zentralschweiz, Interlaken, Brig, Delémont, Yverdon, Balsthal, Frauenfeld und Uznach: Zwischen 1 und 3 Millionen. Der Hauptsitz weist einen umgebuchten Gewinn von minus 0,1 Millionen Franken auf.
            """)

        # Bar, complex
        elif is_complex:
            print("Bar, complex")
            prompt += (f"""
    **Kurzbeschreibung**:
    Diagrammtyp: (Senkrecht / Waagerecht) Balkendiagramm. Titel: {title}. Untertitel: {subtitle}.
    Waagrechte Achse:
    – Zeitreihe [von Jahr X bis Jahr Y (Intervall/Abstand)]
    – Kategorien [bis 10 → Liste] oder [> 10 → Anzahl + Überbegriff, z. B. „Länder“]
    Senkrechte Achse: (Ausschnitt) [Min] bis [Max] [Einheit]. [Anzahl Balken] pro [Jahr/Kategorie]: [Serie A (Farbe)], [Serie B (Farbe)], … (Farbe nur falls Kategorieanzahl kleiner als 5 ist). (Markierungen: falls vorhanden.)
    **Überblick**:
    Fasse die Hauptaussage des Diagramms in 1–2 Sätzen zusammen – etwa Trend, Lücke, Rangfolge, Unterschied oder Korrelation zwischen Serien.                       
    **Lange Beschreibung**: je nach Untertyp:
    A) Kategorie in Zeitreihe:
    – Entwicklung jeder Serie (gleichfarbige Balken über die Zeit): Beschreibe Start → Verlauf → Ende.
    – Vergleich innerhalb eines Jahres (Balken unterschiedlicher Farben in derselben Periode).
    B) Zeitreihe in Kategorie:
    – Optionale Spanne über alle Kategorien. Beispiel: Im 2017 in allen Regionen zwischen 700 und 1'000 kg pro Million Einwohner, im 2015 zwischen 700 und über 2600.
    – Pflicht: Entwicklung je Kategorie beschreiben.
    Beispiel: Bern: 900 auf 2600, Berlin: 600 auf 1400, München: 300 auf 1300.
    C) Unterkategorien in Oberkategorien:
    – Spannweite je Oberkategorie angeben.
    – Hervorgehobene Oberkategorien benennen.
    – Sortierung (sofern nach Unterkategorie geordnet). Beispiel: Sortiert nach Unterkategorie liegt Oberkategorie1 an Stelle 11, Oberkategorie1 an 15, Oberkategorie1 an 30.
    – Vergleiche innerhalb der Unterkategorien pro Oberkategorie.
    Lange Beschreibung maximal {LD_bar_complex + Event_length} Wörter
            """) 

            prompt += ("""
    **Beispiel 1 (A):**
    **Kurzbeschreibung**: Senkrechtes Balkendiagramm. Titel: Auch der Handel mit der EU mit Autoteilen ist aus Sicht der USA defizitär. Untertitel: in Mrd. $. Waagrechte Achse: 2012 bis 2017. Senkrechte Achse: 0 bis 20 Milliarden $. Zwei Balken pro Jahr: US-Exporte (blau), US-Importe (türkis).
    **Überblick**: Jedes Jahr übersteigen die US-Importe die Exporte um etwa 10 bis 14 Milliarden $. Die Exporte steigen leicht, die Importe bleiben auf ähnlichem Niveau.
    **Lange Beschreibung**: Die US-Importe bewegen sich zwischen 17 und 21 Milliarden $. Die US-Exporte steigen von etwa 5 auf knapp 8 Milliarden $. Das Handelsdefizit liegt zwischen 10 und 14 Milliarden $.
                       
    **Beispiel 2 (B):**
    **Kurzbeschreibung**: Waagrechtes Balkendiagramm. Titel: Die Zentraleuropäer testen heute viel mehr. Untertitel: Anzahl Tests pro Millionen Einwohner (Sieben-Tage-Durchschnitt). Waagrechte Achse: 0 bis 2800 Tests. Senkrechte Achse: Wien, Österreich, Tschechien, Ungarn und Slowakei. Zwei Balken pro Region: 14.6. (blau), 14.9. (türkis).
    **Überblick**: Zwischen Juni und September steigen die durchschnittlichen Testzahlen in allen dargestellten Regionen zwischen zweieinhalb und viermal.
    **Lange Beschreibung**: Im Juni in allen Regionen zwischen 200 und 900 Tests pro Million Einwohner. Im September rund 700 bis über 2600 Tests pro Million Einwohner. Entwicklung wie folgt: Wien: 900 auf 2600, Österreich: 600 auf 1400, Tschechien: 300 auf 1300, Ungarn: 400 auf 1000, Slowakei: 200 auf 700.
                       
    **Beispiel 3 (C):**
    **Kurzbeschreibung**: Senkrechtes Balkendiagramm. Titel: Wirtschaftlich gesehen sind die Bauern in Deutschland und der Schweiz eher unwichtig. Untertitel: Anteil des Primärsektors (Landwirtschaft) an der Gesamtwirtschaft in der EU und Efta-Ländern, provisorische Zahlen 2022, in Prozent. Waagrechte Achse: 0 bis 30 Prozent. Senkrechte Achse: 30 Länder sowie den EU-Schnitt. Pro Land zwei Balken: Anteil Beschäftigter (blau), Anteil Bruttowertschöpfung (türkis).
    **Überblick**: Der Anteil Beschäftigter ist in den meisten Ländern höher als der Anteil der Bruttowertschöpfung. Ungarn, Island und Estland sind eine Ausnahme.
    **Lange Beschreibung**: Der Anteil der im Primärsektor Beschäftigten beträgt zwischen knapp 1 Prozent (Luxemburg) bis über 21 Prozent (Rumänien). Die Spannweite der Bruttowertschöpfung liegt zwischen 0,3 Prozent (Luxemburg) und knapp 6 Prozent (Lettland). Im EU-Schnitt beträgt der Anteil der Bruttowertschöpfung etwa 2 Prozent und der Anteil der Beschäftigten rund 4 Prozent. In der Schweiz sind es knapp 1 und rund 2 Prozent, in Deutschland sind beide Werte etwa 1 Prozent. Sortiert nach Bruttowertschöpfung liegt der EU-Schnitt an Stelle 10, die Schweiz an 23. und Deutschland an 29. Stelle.
            """)

        else:
            raise ValueError("Unbekannte Datenstruktur – Verarbeitung nicht möglich.")

    else:

        # Stacked Bar, simple
        if not is_complex:
            print("Bar, simple and 7 or less bars")
            prompt += (f"""
    **Kurzbeschreibung**:
    Diagrammtyp: (Waagrecht / Senkrecht) gestapeltes Balkendiagramm. Titel: {title}. Untertitel: {subtitle}. Waagrechte Achse: [Min] bis [Max] [Einheit]. Senkrechte Achse: [Zeitpunkte oder Kategorien]. Segmente pro Balken: [A (Farbe)], [B (Farbe)], …. (Farbe nur falls Kategorieanzahl kleiner als 5 ist). (Markierungen: falls vorhanden.)
    **Überblick**:
    - Bei 100 Prozent-Diagrammen: Ändern sich die Anteile der Segmente im Zeitverlauf oder zwischen Kategorien? Gibt es dominante Segmente?
    - Bei absoluten Werten: Bleibt die Gesamthöhe der Balken stabil, steigt oder sinkt sie über Zeit oder zwischen Gruppen?     
    **Lange Beschreibung**: je nach Variante:      
    A) Auf 100 Prozent, mehrere Balken (2+ Segmente) 
    – Beschreibe die Entwicklung der Anteile eines Segments über Zeit oder Kategorien.
    – Nenne dominante Segmente, falls erkennbar.
    B) Auf 100 Prozent, ein Balken (Aufschlüsselung in Prozent)
    – Beschreibe die Zusammensetzung: Welche Segmente dominieren? Welche haben geringe Anteile?
    C) Auf absolute Zahl, ein Balken (Aufschlüsselung in absoluten Werten)
    – Beschreibe die Gesamtgröße und die Verteilung der Teilsegmente (z. B. in Millionen, Tonnen, Einwohnern). 
    Lange Beschreibung maximal {LD_stacked_bar_simple + Event_length} Wörter               
            """)

            prompt += ("""
    **Beispiel 1 (A):**
    **Kurzbeschreibung**: Waagrechtes Balkendiagramm. Titel: Die Berufswahl folgt alten Rollenmustern. Untertitel: Frauen- und Männeranteile in verschiedenen Berufen im Jahr 2022, in Prozent. Waagrechte Achse: 0 bis 100 Prozent. Senkrechte Achse: Krankenpflege und Geburtshilfe (Berufliche Grundbildung), Elektrizität und Energie (Berufliche Grundbildung), Soziale Arbeit (Studierende an den Fachhochschulen), Technik und IT (Studierende an den Fachhochschulen), Humanmedizin (Studierende an den universitären Hochschulen), Maschinen- und Elektroingenieurwesen (Studierende an den universitären Hochschulen). Pro Beruf ein Balken, unterteilt in zwei Anteile: Frauen (violett), Männer (grün).
    **Überblick**: In den meisten Berufsbereichen zeigt sich eine geschlechtsspezifische Verteilung.
    **Lange Beschreibung**: Frauen sind in der Krankenpflege und Geburtshilfe (85 Prozent) sowie in der Sozialen Arbeit (72 Prozent) stark vertreten. Männer dagegen sind in den Berufen Elektrizität und Energie (97 Prozent), in Technik und IT (86 Prozent) sowie im Maschinen- und Elektroingenieurwesen (80 Prozent) stärker vertreten. Eine ausgeglichenere Verteilung zeigt sich in der Humanmedizin mit einem Verhältnis von rund 60 zu 40 Prozent Frauen zu Männern.
                       
    **Beispiel 2 (B):**
    **Kurzbeschreibung**: Waagrechtes Balkendiagramm. Titel: Das Vereinigte Königreich ist 2022 bislang der grösste Geldgeber. Untertitel: Aufschlüsselung der bisher zugesagten 101,6 Millionen $ (in %-Anteilen). Waagrechte Achse: 0 bis 100 Prozent. Ein Balken unterteilt in sechs Anteile: Vereinigtes Königreich, EU, Deutschland, Italien, Irland und Schweiz.
    **Überblick**: Das Vereinigte Königreich stellt mit 88 Prozent den grössten Anteil.
    **Lange Beschreibung**: Vereinigtes Königreich: 88 Prozent. Anteile EU und Deutschland je etwa 4 Prozent, Italien, Irland und Schweiz je etwa 1 Prozent
                       
    **Beispiel 3 (C):**
    **Kurzbeschreibung**: Waagrechtes Balkendiagramm. Titel: So viel gibt die Waadt für Prämienverbilligungen aus. Untertitel: Zahlen für 2023, in Millionen Franken, gerundet. Waagrechte Achse: 0 bis 900 Millionen Franken. Ein Balken unterteilt in drei Anteile: «Normale» Prämienverbilligungen (blau), AHV/IV-Ergänzungsleistungen und Sozialhilfe (türkis), 10-Prozent-Deckel (orange).
    **Überblick**: Der Kanton Waadt gibt im Jahr 2023 rund 800 Millionen Franken für Prämienverbilligungen aus.
    **Lange Beschreibung**: Der Anteil von normalen Prämienverbilligungen beträgt 370 Millionen Franken, von AHV/IV-Ergänzungsleistungen und Sozialhilfe 310 Millionen Franken und derjenige vom 10-Prozent-Deckel rund 120 Millionen.
            """)

        # Stacked Bar, complex
        elif is_complex:
            print("Bar, simple, more than 7 bars")
            prompt += (f"""
    **Kurzbeschreibung**:
    Diagrammtyp: (Waagrecht / Senkrecht) gestapeltes Balkendiagramm. Titel: {title}. Untertitel: {subtitle}. Waagrechte Achse:
    – Zeitreihe: [von X bis Y] (Intervall: [jährlich/vierteljährlich/…] falls vorhanden. Sonst Jahre nennen)
    – Kategorien: [Liste bis 10] / [Anzahl + Überbegriff bei >10].
    Senkrechte Achse: (Ausschnitt) [Min] bis [Max] [Einheit]. Segmente pro Balken: [A (Farbe)], [B (Farbe)], [C (Farbe)], … (Farbe nur falls Kategorieanzahl kleiner als 5 ist). (Markierungen: falls vorhanden.)
    **Überblick**:
    - Entwicklung der Gesamthöhe (stabil / steigend / rückläufig; nenne ggf. Spitzenjahre).
    - Segmentdominanz und Verschiebungen zwischen Segmenten (Wechsel der führenden Segmente, deutliche Zunahmen/Abnahmen).
                       
    **Lange Beschreibung**:
    - Gesamtentwicklung: Nenne Anfangs-, Peak-, Tief- und Endniveau; gliedere in Trendabschnitte (z. B. „X–Z Anstieg, Z–Y Rückgang“).
    - Zusammensetzung der Segmente: Identifiziere Leit- und Sekundärsegmente; beschreibe Entwicklung, Anteils­schwankungen und möglichen Rollentausch.
    - Ereignisse/Marken: Ordne die Wirkung kurz ein (Sprung, Trendwechsel, temporär?).
    - Kategorienvergleich (falls keine Zeitreihe): Rangfolge nach Gesamthöhe und Unterschiede im Anteilsmix (ähnliche Gruppen ggf. clustern).  
    Lange Beschreibung maximal {LD_stacked_bar_complex + Event_length} Wörter
            """) 

            prompt += (
                """
    **Beispiel 1:**
    **Kurzbeschreibung**: Senkrechtes Balkendiagramm. Titel: Verbreitung von Kurzarbeit bleibt hartnäckig hoch. Untertitel: Beschäftigte in Kurzarbeit, nach Wirtschaftssektoren. Waagrechte Achse: Monate Januar 2020 bis Februar 2021. Senkrechte Achse: 0 bis 1,4 Millionen Beschäftigte. Pro Jahr ein Balken unterteilt in zwölf Anteile.
    **Überblick**: Zwischen März und Mai 2020 steigt die Zahl der Kurzarbeitenden von rund 5'000 auf bis zu 1,5 Millionen und sinkt ab Juli wieder auf etwa 400'000.
    **Lange Beschreibung**:
    Januar bis Februar 2020: kaum Kurzarbeit (rund 5 000 Personen). März und April 2020: starker Anstieg auf über 1,3 Millionen Beschäftigte. Mai bis Juli 2020: deutlicher Rückgang auf etwa 420'000. August und Oktober 2020: weiterer Rückgang auf rund 250'000. November 2020 bis Februar 2021: erneuter Anstieg auf etwa 400'000 bis 450'000.
    Branchenanteile: Die drei grössten Sektoren ( Restaurants/Hotellerie, Handel/Garagen sowie Industrie/Chemie/Pharma) stellen zusammen meist 50 bis 60 Prozent der Kurzarbeitenden. Restaurants/Hotellerie häufig 15 bis 25 Prozent. Handel/Garagen und Industrie/Chemie/Pharma jeweils 10 bis 20 Prozent. Freiberufliche Dienstleistungen, Transport/Logistik sowie Kunst/Unterhaltung/Erholung tragen je 5 bis 10 Prozent bei.
    Baugewerbe/Bau, Medien/Kommunikation/IT, Gesundheits- und Sozialwesen, Banken/Versicherungen sowie Immobilien liegen jeweils bei 0,5 bis 4 Prozent.

    *Beispiel 2:**
    **Kurzbeschreibung**: Waagrechtes Balkendiagramm. Titel: Die Discounter legen bei Bio zu. Untertitel: Marktanteile der Verkaufskanäle am gesamten Bio-Umsatz. Waagrechte Achse: 0 bis 100 Prozent. Senkrechte Achse: 2018 bis 2022. Pro Jahr ein Balken unterteilt in vier Anteile: Klassischer Detailhandel mit Online (blau), Discounter (türkis), Fachhandel (orange), Marktstand/Bauernhöfe (rot).
    **Überblick**: Von 2018 bis 2022 verschieben sich die Marktanteile der Verkaufskanäle zwischen jeweils 1 bis 3 Prozent. Klassischer Detailhandel macht immer fast 90 Prozent aus. Die drei anderen Verkaufskanäle teilen sich die restlichen Prozente.
    **Lange Beschreibung**: Der Anteil vom klassischen Detailhandel beträgt 2018 fast 90 Prozent und sinkt bis 2022 um knapp ein Prozent. Die Anteile von Fachhandel und Marktstände/Bauernhöhe sind 2.5 und 4.5 Prozent und sinken je knapp ein Prozent. Dafür stiegt der Anteil der Discounter von etwa 3.5 auf 6 Prozent. 
            """)         
            
        else:
            raise ValueError("Unbekannte Datenstruktur – Verarbeitung nicht möglich.")

    # === CSV-Datenanzeige ===
    prompt += "\n\n**CSV-Datenauszug:**\n" + df.to_string(index=False)

    # === Optional: Bildanzeige ===
    if os.path.exists(png_path):
        try:
            img = mpimg.imread(png_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(title)
            plt.show()
        except Exception as e:
            print(f"Fehler beim Anzeigen des Diagrammbilds: {e}")

    # === Stilregeln anhängen ===
    prompt += """
    Stilregeln: Kurz, klar, prägnant, keine Interpretation der Ursachen, neutrale Sprache. Nutze die Verben 'steigen' und 'sinken' für Trendbeschreibungen anstelle von 'erreichen' oder 'einbrechen' um eine neutrale Sprache zu gewährleisten. Schreibe minus aus und nutze nicht das Zeichen dafür. Ziel: Der Text soll Leser das Diagramm ohne visuelle Hilfe verständlich machen. Gib die Texte in folgendem Format zurück:\n\n**Kurzbeschreibung:** [Text]\n\n**Überblick:** [Text]\n\n**Lange Beschreibung:** [Text]
    """

    return prompt

#----------------------------------prompt1 text for evaluation--------------------------------------


def prompt_klarheit(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Positional Bias, Verbosity Bias
    (längere Texte nicht automatisch besser bewerten), Beauty Bias, Emotion Bias, Overconfidence Bias
    und stilistische Vorlieben. Nutze das Diagramm-Bild und die Datentabelle ausschließlich als
    Faktenbasis – ohne zusätzliche Vermutungen oder Interpretationen.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Klarheit**

    Definition:
    Ein klarer Alt-Text ist logisch strukturiert, präzise formuliert und unmittelbar verständlich,
    ohne unnötig komplexe Sprache oder gedankliche Sprünge. Er erlaubt Leser:innen, den dargestellten
    Sachverhalt gedanklich leicht nachzuvollziehen.

    Bewertungsskala (1–5):
    1 = sehr unklar: schwer verständlich, verwirrend, inkonsistente Struktur
    2 = teilweise unklar: mehrere Verständlichkeitsprobleme, unpräzise Formulierungen
    3 = akzeptabel: im Kern verständlich, aber mit deutlichen Schwächen
    4 = klar: gut verständlich und strukturiert, nur minimale Schwächen
    5 = sehr klar: präzise, logisch aufgebaut, hervorragend verständlich

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Prüfe die Struktur: Einleitung → Kernelemente → ggf. Fazit oder Zusammenfassung.
    - Prüfe sprachliche Präzision und Vermeidung unnötig komplexer Sätze.
    - Prüfe mentale Vorstellbarkeit: Lässt sich das Diagramm auf Basis des Textes gedanklich rekonstruieren?
    - Prüfe Kohärenz und Nachvollziehbarkeit, unabhängig von Stil oder Länge (nur Klarheit bewerten).
    - Bewerte NICHT:
      - Vollständigkeit der Inhalte (dafür gibt es das Kriterium „Vollständigkeit“)
      - Textlänge oder Prägnanz (dafür gibt es das Kriterium „Kürze“)

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_vollstaendigkeit(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Positional Bias, Verbosity Bias
    (längere Texte nicht automatisch besser bewerten), Beauty Bias, Emotion Bias, Overconfidence Bias
    und stilistische Vorlieben. Nutze das Diagramm-Bild und die Datentabelle ausschließlich als
    Faktenbasis – ohne zusätzliche Vermutungen oder Interpretationen.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Vollständigkeit**

    Definition:
    Ein vollständig formulierter Alt-Text beschreibt alle wesentlichen Diagrammelemente,
    die für das Verständnis des Inhalts nötig sind. Dazu gehören – sofern im Diagramm vorhanden –
    Achsen und Achsentitel, Einheiten, Kategorien, Legenden/Markierungen, relevante Werte,
    besondere Punkte oder Bereiche, Muster, Trends und zentrale Zusammenhänge.

    Bewertungsskala (1–5):
    1 = stark unvollständig: zentrale Diagrammelemente fehlen; keine sinnvolle Beschreibung möglich
    2 = unvollständig: mehrere wichtige Elemente fehlen oder sind falsch/ungenau erwähnt
    3 = teilweise vollständig: einige wichtige Elemente enthalten, aber erkennbare Lücken
    4 = weitgehend vollständig: fast alle wesentlichen Elemente richtig beschrieben, nur geringe Lücken
    5 = sehr vollständig: alle wesentlichen Elemente klar, korrekt und vollständig beschrieben

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Identifiziere anhand des Diagrammbilds die tatsächlich vorhandenen Kernelemente.
    - Vergleiche diese strukturiert mit dem Alt-Text:
      - Achsen? Achsentitel? Einheiten?
      - Kategorien / Gruppen / Zeiträume?
      - Legende / Datenreihen?
      - Hauptwerte, Trends, Peaks, Ausreißer?
    - Bewerte NICHT:
      - Schönheit der Sprache oder Stil
      - Kürze oder Länge (dafür gibt es „Kürze“)
      - Klarheit oder Struktur (dafür gibt es „Klarheit“)
      - Interpretationen oder Schlussfolgerungen (nur deskriptive Inhalte zählen)
    - Überflüssige Details erhöhen die Bewertung NICHT.

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_kuerze(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Verbosity Bias
    (längere Texte nicht automatisch besser bewerten), Positional Bias, Beauty Bias,
    Emotion Bias, Overconfidence Bias und stilistische Vorlieben. Nutze das Diagramm-Bild
    und die Datentabelle ausschließlich als Faktenbasis – ohne zusätzliche Vermutungen
    oder Interpretationen.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Kürze (Prägnanz bei ausreichender Information)**

    Definition:
    Ein guter Alt-Text zur Kürze enthält die wesentlichen Informationen, ist aber nicht unnötig lang.
    Er ist weder zu knapp (wichtige Inhalte fehlen) noch überladen (viele redundante oder irrelevante
    Details). Bewertet wird das Verhältnis von Informationsgehalt zu Länge – nicht Stil oder Wortwahl.

    Bewertungsskala (1–5):
    1 = stark unausgewogen:
        - viel zu kurz: zentrale Informationen fehlen offensichtlich
        ODER
        - viel zu lang: zahlreiche unnötige/redundante Details
    2 = unausgewogen:
        - eher zu kurz oder eher zu lang; mehrere klare Probleme in der Balance
    3 = akzeptabel:
        - im Großen und Ganzen okay, aber mit erkennbaren Längen- oder Kürzemängeln
    4 = gut ausgewogen:
        - alle wichtigen Inhalte vorhanden, nur geringe Über- oder Unterlänge
    5 = optimal prägnant:
        - wesentliche Inhalte klar abgedeckt, kaum redundante oder irrelevante Passagen

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Bestimme anhand von Diagramm-Bild und Datentabelle mental, welche Inhalte MINDESTENS
      nötig sind, damit das Diagramm verstanden werden kann (Kernaussage + grundlegende Struktur).
    - Prüfe dann:
      - Fehlen offensichtlich notwendige Aspekte? → eher „zu kurz“
      - Enthält der Text viele Nebensätze, Wiederholungen, Ausschmückungen oder irrelevante Details,
        die für das Verständnis des Diagramms nicht nötig sind? → eher „zu lang“
    - Bewerte NICHT:
      - sprachliche Schönheit oder Stil
      - formale Klarheit oder Struktur (dafür gibt es „Klarheit“)
      - absolute Vollständigkeit aller möglichen Details (dafür gibt es „Vollständigkeit“)
    - Fokussiere NUR auf das Verhältnis von Informationsgehalt zu Länge.

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_wahrgenommene_vollstaendigkeit(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Verbosity Bias
    (längere Texte nicht automatisch besser bewerten), Beauty Bias, Positional Bias,
    Emotion Bias, Overconfidence Bias und stilistische Vorlieben. Nutze das Diagramm-Bild
    und die Datentabelle ausschließlich als Faktenbasis – ohne zusätzliche Vermutungen
    oder Interpretationen.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Wahrgenommene Vollständigkeit (Abdeckung der Hauptaussage)**

    Definition:
    Ein Alt-Text ist wahrgenommen vollständig, wenn er genügend Informationen vermittelt,
    damit die zentrale Hauptaussage des Diagramms klar verstanden werden kann – auch dann,
    wenn Neben- oder Detailaspekte weggelassen werden. Entscheidend ist, ob Leser:innen
    den Kerninhalt und die Aussage des Diagramms korrekt erfassen können.

    Bewertungsskala (1–5):
    1 = vermittelt kaum Verständnis:
        - Hauptaussage ist unklar oder falsch; der Zusammenhang des Diagramms bleibt weitgehend unverständlich
    2 = unzureichend für die Hauptaussage:
        - Teile der Aussage erkennbar, aber zentrale Aspekte fehlen oder sind irreführend
    3 = teilweise verständlich:
        - Hauptaussage in Grundzügen erkennbar, aber wichtige Nuancen oder Kontext fehlen
    4 = vermittelt die Hauptaussage gut:
        - Kernbotschaft ist klar und weitgehend korrekt, kleinere Ergänzungen wären möglich
    5 = vermittelt die Hauptaussage vollständig und klar:
        - zentrale Aussage und Kontext des Diagramms sind klar, korrekt und verständlich beschrieben

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Bestimme anhand von Diagramm-Bild und Datentabelle explizit:
      - Was ist die Hauptaussage bzw. Kernbotschaft dieses Diagramms?
        (z. B. stärkster Anstieg, wichtigster Vergleich, auffälligste Entwicklung)
    - Prüfe dann den Alt-Text:
      - Spiegelt er diese Hauptaussage korrekt wider?
      - Sind zentrale Bezüge (z. B. welche Kategorien/Zeitspanne/Vergleiche die Aussage betreffen)
        vorhanden, sodass die Kernaussage ohne Blick auf das Bild nachvollziehbar ist?
    - Details, kleine Ausnahmen oder alle Einzelwerte müssen NICHT enthalten sein.
    - Bewerte NICHT:
      - sprachliche Schönheit oder Stil
      - Vollständigkeit aller Diagrammelemente (dafür gibt es „Vollständigkeit“)
      - Textlänge an sich (dafür gibt es „Kürze“)
    - Fokussiere NUR darauf, ob die Hauptaussage des Diagramms verständlich und korrekt vermittelt wird.

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_neutralität(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Beauty Bias (wohlklingende Texte
    nicht bevorzugen), Emotion Bias, Positional Bias, Overconfidence Bias und stilistische Vorlieben.
    Nutze das Diagramm-Bild und die Datentabelle ausschließlich als groben Kontext – deine Bewertung
    bezieht sich hier **nur auf die Sprachwahl**, nicht auf die inhaltliche Korrektheit der Daten.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Neutralität und Abwesenheit von Wertungen in der SPRACHWAHL**

    Definition:
    Ein neutral formulierter Alt-Text verwendet eine sachliche, beschreibende Sprache.
    Er enthält keine wertenden, emotionalen oder suggerierenden Ausdrücke.
    Es geht hier ausschließlich um die Wortwahl, nicht darum, ob Zahlen, Kategorien
    oder Inhalte faktisch korrekt sind.

    Bewertungsskala (1–5):
    1 = deutlich wertend / emotional:
        - mehrere klar wertende, emotionale oder suggestive Formulierungen
    2 = teilweise wertend:
        - einzelne deutlich wertende Ausdrücke oder emotionale Sprache
    3 = überwiegend neutral:
        - überwiegend sachliche Sprache, aber kleinere wertende oder suggestive Elemente
    4 = neutral:
        - klar sachliche, nüchterne Sprache ohne erkennbare Wertungen
    5 = vollständig neutral:
        - ausschließlich deskriptive, objektive Formulierungen; keinerlei Bewertungen oder
          implizit wertende Sprache

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Prüfe nur die Wortwahl, NICHT die inhaltliche Richtigkeit der Aussagen.
    - Wertende / emotionale Begriffe:
      - z. B. „stark“, „schwach“, „erfreulich“, „alarmierend“, „beeindruckend“, „problematisch“,
        „schlimm“, „gut“, „traurig“
    - Suggestive Worte:
      - z. B. „nur“, „kaum“, „offensichtlich“, „typischerweise“, „ausgerechnet“
    - Implizite Wertungen:
      - z. B. „bereits“, „immer noch“, „endlich“, wenn sie eine Haltung transportieren
    - Bewerte NICHT:
      - ob Werte, Trends oder Kategorien korrekt beschrieben sind
      - Klarheit, Textlänge, Vollständigkeit oder Struktur

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_faktenkorrektheit(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Beauty Bias (wohlklingende Texte
    nicht bevorzugen), Emotion Bias, Positional Bias, Overconfidence Bias und stilistische Vorlieben.
    Nutze das Diagramm-Bild und die Datentabelle ausschließlich als Faktenbasis – ohne zusätzliche
    Vermutungen oder Erklärungsversuche.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Faktenkorrektheit (Faktentreue / Halluzinationsfreiheit / keine Selbstinterpretationen)**

    Definition:
    Ein faktentreuer Alt-Text beschreibt ausschließlich Inhalte, die im Diagramm oder
    in der Datentabelle eindeutig vorhanden und direkt ablesbar sind. Er enthält keine
    erfundenen Werte, Kategorien, Achsentitel, Farben, Trends, Ursachenbehauptungen
    oder darüber hinausgehende Interpretationen des Diagramms.

    Bewertungsskala (1–5):
    1 = stark fehlerhaft:
        - mehrere falsche oder erfundene Inhalte; deutliche Abweichung von Diagramm/Daten;
          starke eigene Interpretation oder Erklärung
    2 = fehlerhaft:
        - einige falsche Elemente oder klare Halluzinationen; einzelne Interpretationen,
          die nicht direkt aus den Daten hervorgehen
    3 = teilweise korrekt:
        - Mehrheit der Inhalte korrekt, aber einzelne Ungenauigkeiten, überzogene
          Interpretationen oder kleinere Halluzinationen
    4 = weitgehend korrekt:
        - überwiegend faktentreu; nur minimale, nicht schwerwiegende Abweichungen,
          kaum bis keine Interpretation über direkt Erkennbares hinaus
    5 = vollständig korrekt:
        - keine Halluzinationen; alle genannten Fakten stimmen mit Bild und Daten überein;
          es werden nur Informationen beschrieben, die eindeutig aus Diagramm oder CSV
          ablesbar sind, ohne eigene Deutung oder Erklärung.

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Prüfe ausschließlich inhaltliche Korrektheit, NICHT die Art der Formulierung oder Tonalität.
    - Prüfe, ob der Alt-Text erfundene:
      - Kategorien, Werte, Einheiten, Zeiträume, Achsentitel, Farben oder Beziehungen nennt.
    - Prüfe, ob Interpretationen oder Erklärungen vorkommen, die über das direkt Sichtbare
      hinausgehen, z. B.:
      - Ursachen („weil“, „aufgrund“, „infolge“)
      - Vermutungen („vermutlich“, „scheint“, „wahrscheinlich“)
      - Bewertungen von Trends („nimmt stark zu“, „bricht ein“), wenn diese nicht klar
        aus den Daten ableitbar sind.
    - Vergleiche alle genannten Fakten mit Bild und CSV:
      - Stimmen Größenordnungen, Richtungen, Kategorien, Zeiträume und Bezeichnungen?
    - Bewerte NICHT:
      - sprachliche Qualität, Neutralität der Wortwahl, Textlänge oder Struktur

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """



#----------------------------------prompt2 text for evaluation--------------------------------------

def prompt_klarheit_reason(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Positional Bias, Verbosity Bias
    (längere Texte nicht automatisch besser bewerten), Beauty Bias, Emotion Bias, Overconfidence Bias
    und stilistische Vorlieben. Nutze das Diagramm-Bild und die Datentabelle ausschließlich als
    Faktenbasis – ohne zusätzliche Vermutungen oder Interpretationen.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Klarheit**

    Definition:
    Ein klarer Alt-Text ist logisch strukturiert, präzise formuliert und unmittelbar verständlich,
    ohne unnötig komplexe Sprache oder gedankliche Sprünge. Er erlaubt Leser:innen, den dargestellten
    Sachverhalt gedanklich leicht nachzuvollziehen.

    Bewertungsskala (1–5):
    1 = sehr unklar: schwer verständlich, verwirrend, inkonsistente Struktur
    2 = teilweise unklar: mehrere Verständlichkeitsprobleme, unpräzise Formulierungen
    3 = akzeptabel: im Kern verständlich, aber mit deutlichen Schwächen
    4 = klar: gut verständlich und strukturiert, nur minimale Schwächen
    5 = sehr klar: präzise, logisch aufgebaut, hervorragend verständlich

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Prüfe die Struktur: Einleitung → Kernelemente → ggf. Fazit oder Zusammenfassung.
    - Prüfe sprachliche Präzision und Vermeidung unnötig komplexer Sätze.
    - Prüfe mentale Vorstellbarkeit: Lässt sich das Diagramm auf Basis des Textes gedanklich rekonstruieren?
    - Prüfe Kohärenz und Nachvollziehbarkeit, unabhängig von Stil oder Länge (nur Klarheit bewerten).
    - Bewerte NICHT:
      - Vollständigkeit der Inhalte (dafür gibt es das Kriterium „Vollständigkeit“)
      - Textlänge oder Prägnanz (dafür gibt es das Kriterium „Kürze“)

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Reason: <1-2 Sätze>
    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_vollstaendigkeit_reason(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Positional Bias, Verbosity Bias
    (längere Texte nicht automatisch besser bewerten), Beauty Bias, Emotion Bias, Overconfidence Bias
    und stilistische Vorlieben. Nutze das Diagramm-Bild und die Datentabelle ausschließlich als
    Faktenbasis – ohne zusätzliche Vermutungen oder Interpretationen.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Vollständigkeit**

    Definition:
    Ein vollständig formulierter Alt-Text beschreibt alle wesentlichen Diagrammelemente,
    die für das Verständnis des Inhalts nötig sind. Dazu gehören – sofern im Diagramm vorhanden –
    Achsen und Achsentitel, Einheiten, Kategorien, Legenden/Markierungen, relevante Werte,
    besondere Punkte oder Bereiche, Muster, Trends und zentrale Zusammenhänge.

    Bewertungsskala (1–5):
    1 = stark unvollständig: zentrale Diagrammelemente fehlen; keine sinnvolle Beschreibung möglich
    2 = unvollständig: mehrere wichtige Elemente fehlen oder sind falsch/ungenau erwähnt
    3 = teilweise vollständig: einige wichtige Elemente enthalten, aber erkennbare Lücken
    4 = weitgehend vollständig: fast alle wesentlichen Elemente richtig beschrieben, nur geringe Lücken
    5 = sehr vollständig: alle wesentlichen Elemente klar, korrekt und vollständig beschrieben

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Identifiziere anhand des Diagrammbilds die tatsächlich vorhandenen Kernelemente.
    - Vergleiche diese strukturiert mit dem Alt-Text:
      - Achsen? Achsentitel? Einheiten?
      - Kategorien / Gruppen / Zeiträume?
      - Legende / Datenreihen?
      - Hauptwerte, Trends, Peaks, Ausreißer?
    - Bewerte NICHT:
      - Schönheit der Sprache oder Stil
      - Kürze oder Länge (dafür gibt es „Kürze“)
      - Klarheit oder Struktur (dafür gibt es „Klarheit“)
      - Interpretationen oder Schlussfolgerungen (nur deskriptive Inhalte zählen)
    - Überflüssige Details erhöhen die Bewertung NICHT.

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Reason: <1-2 Sätze>
    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_kuerze_reason(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Verbosity Bias
    (längere Texte nicht automatisch besser bewerten), Positional Bias, Beauty Bias,
    Emotion Bias, Overconfidence Bias und stilistische Vorlieben. Nutze das Diagramm-Bild
    und die Datentabelle ausschließlich als Faktenbasis – ohne zusätzliche Vermutungen
    oder Interpretationen.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Kürze (Prägnanz bei ausreichender Information)**

    Definition:
    Ein guter Alt-Text zur Kürze enthält die wesentlichen Informationen, ist aber nicht unnötig lang.
    Er ist weder zu knapp (wichtige Inhalte fehlen) noch überladen (viele redundante oder irrelevante
    Details). Bewertet wird das Verhältnis von Informationsgehalt zu Länge – nicht Stil oder Wortwahl.

    Bewertungsskala (1–5):
    1 = stark unausgewogen:
        - viel zu kurz: zentrale Informationen fehlen offensichtlich
        ODER
        - viel zu lang: zahlreiche unnötige/redundante Details
    2 = unausgewogen:
        - eher zu kurz oder eher zu lang; mehrere klare Probleme in der Balance
    3 = akzeptabel:
        - im Großen und Ganzen okay, aber mit erkennbaren Längen- oder Kürzemängeln
    4 = gut ausgewogen:
        - alle wichtigen Inhalte vorhanden, nur geringe Über- oder Unterlänge
    5 = optimal prägnant:
        - wesentliche Inhalte klar abgedeckt, kaum redundante oder irrelevante Passagen

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Bestimme anhand von Diagramm-Bild und Datentabelle mental, welche Inhalte MINDESTENS
      nötig sind, damit das Diagramm verstanden werden kann (Kernaussage + grundlegende Struktur).
    - Prüfe dann:
      - Fehlen offensichtlich notwendige Aspekte? → eher „zu kurz“
      - Enthält der Text viele Nebensätze, Wiederholungen, Ausschmückungen oder irrelevante Details,
        die für das Verständnis des Diagramms nicht nötig sind? → eher „zu lang“
    - Bewerte NICHT:
      - sprachliche Schönheit oder Stil
      - formale Klarheit oder Struktur (dafür gibt es „Klarheit“)
      - absolute Vollständigkeit aller möglichen Details (dafür gibt es „Vollständigkeit“)
    - Fokussiere NUR auf das Verhältnis von Informationsgehalt zu Länge.

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Reason: <1-2 Sätze>
    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_wahrgenommene_vollstaendigkeit_reason(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Verbosity Bias
    (längere Texte nicht automatisch besser bewerten), Beauty Bias, Positional Bias,
    Emotion Bias, Overconfidence Bias und stilistische Vorlieben. Nutze das Diagramm-Bild
    und die Datentabelle ausschließlich als Faktenbasis – ohne zusätzliche Vermutungen
    oder Interpretationen.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Wahrgenommene Vollständigkeit (Abdeckung der Hauptaussage)**

    Definition:
    Ein Alt-Text ist wahrgenommen vollständig, wenn er genügend Informationen vermittelt,
    damit die zentrale Hauptaussage des Diagramms klar verstanden werden kann – auch dann,
    wenn Neben- oder Detailaspekte weggelassen werden. Entscheidend ist, ob Leser:innen
    den Kerninhalt und die Aussage des Diagramms korrekt erfassen können.

    Bewertungsskala (1–5):
    1 = vermittelt kaum Verständnis:
        - Hauptaussage ist unklar oder falsch; der Zusammenhang des Diagramms bleibt weitgehend unverständlich
    2 = unzureichend für die Hauptaussage:
        - Teile der Aussage erkennbar, aber zentrale Aspekte fehlen oder sind irreführend
    3 = teilweise verständlich:
        - Hauptaussage in Grundzügen erkennbar, aber wichtige Nuancen oder Kontext fehlen
    4 = vermittelt die Hauptaussage gut:
        - Kernbotschaft ist klar und weitgehend korrekt, kleinere Ergänzungen wären möglich
    5 = vermittelt die Hauptaussage vollständig und klar:
        - zentrale Aussage und Kontext des Diagramms sind klar, korrekt und verständlich beschrieben

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Bestimme anhand von Diagramm-Bild und Datentabelle explizit:
      - Was ist die Hauptaussage bzw. Kernbotschaft dieses Diagramms?
        (z. B. stärkster Anstieg, wichtigster Vergleich, auffälligste Entwicklung)
    - Prüfe dann den Alt-Text:
      - Spiegelt er diese Hauptaussage korrekt wider?
      - Sind zentrale Bezüge (z. B. welche Kategorien/Zeitspanne/Vergleiche die Aussage betreffen)
        vorhanden, sodass die Kernaussage ohne Blick auf das Bild nachvollziehbar ist?
    - Details, kleine Ausnahmen oder alle Einzelwerte müssen NICHT enthalten sein.
    - Bewerte NICHT:
      - sprachliche Schönheit oder Stil
      - Vollständigkeit aller Diagrammelemente (dafür gibt es „Vollständigkeit“)
      - Textlänge an sich (dafür gibt es „Kürze“)
    - Fokussiere NUR darauf, ob die Hauptaussage des Diagramms verständlich und korrekt vermittelt wird.

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Reason: <1-2 Sätze>
    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_neutralität_reason(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Beauty Bias (wohlklingende Texte
    nicht bevorzugen), Emotion Bias, Positional Bias, Overconfidence Bias und stilistische Vorlieben.
    Nutze das Diagramm-Bild und die Datentabelle ausschließlich als groben Kontext – deine Bewertung
    bezieht sich hier **nur auf die Sprachwahl**, nicht auf die inhaltliche Korrektheit der Daten.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Neutralität und Abwesenheit von Wertungen in der SPRACHWAHL**

    Definition:
    Ein neutral formulierter Alt-Text verwendet eine sachliche, beschreibende Sprache.
    Er enthält keine wertenden, emotionalen oder suggerierenden Ausdrücke.
    Es geht hier ausschließlich um die Wortwahl, nicht darum, ob Zahlen, Kategorien
    oder Inhalte faktisch korrekt sind.

    Bewertungsskala (1–5):
    1 = deutlich wertend / emotional:
        - mehrere klar wertende, emotionale oder suggestive Formulierungen
    2 = teilweise wertend:
        - einzelne deutlich wertende Ausdrücke oder emotionale Sprache
    3 = überwiegend neutral:
        - überwiegend sachliche Sprache, aber kleinere wertende oder suggestive Elemente
    4 = neutral:
        - klar sachliche, nüchterne Sprache ohne erkennbare Wertungen
    5 = vollständig neutral:
        - ausschließlich deskriptive, objektive Formulierungen; keinerlei Bewertungen oder
          implizit wertende Sprache

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Prüfe nur die Wortwahl, NICHT die inhaltliche Richtigkeit der Aussagen.
    - Wertende / emotionale Begriffe:
      - z. B. „stark“, „schwach“, „erfreulich“, „alarmierend“, „beeindruckend“, „problematisch“,
        „schlimm“, „gut“, „traurig“
    - Suggestive Worte:
      - z. B. „nur“, „kaum“, „offensichtlich“, „typischerweise“, „ausgerechnet“
    - Implizite Wertungen:
      - z. B. „bereits“, „immer noch“, „endlich“, wenn sie eine Haltung transportieren
    - Bewerte NICHT:
      - ob Werte, Trends oder Kategorien korrekt beschrieben sind
      - Klarheit, Textlänge, Vollständigkeit oder Struktur

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Reason: <1-2 Sätze>
    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """


def prompt_faktenkorrektheit_reason(alt_text: str, csv_text: str) -> str:
    return f"""
    Du agierst als neutraler, unparteiischer Evaluator für Alternativtexte von Diagrammen.
    Folge strikt den Anweisungen. Vermeide Verzerrungen wie Beauty Bias (wohlklingende Texte
    nicht bevorzugen), Emotion Bias, Positional Bias, Overconfidence Bias und stilistische Vorlieben.
    Nutze das Diagramm-Bild und die Datentabelle ausschließlich als Faktenbasis – ohne zusätzliche
    Vermutungen oder Erklärungsversuche.

    Dir werden drei Inputs bereitgestellt:
    1. Ein Diagramm als Bild (separater Input, nicht im Text dargestellt).
    2. Ein vorgeschlagener Alt-Text.
    3. Ein Auszug der zugrunde liegenden Daten:

    Datentabelle:
    {csv_text}

    ============================================================
    AUFGABE
    Bewerte den Alt-Text ausschließlich nach dem Kriterium:
    **Faktenkorrektheit (Faktentreue / Halluzinationsfreiheit / keine Selbstinterpretationen)**

    Definition:
    Ein faktentreuer Alt-Text beschreibt ausschließlich Inhalte, die im Diagramm oder
    in der Datentabelle eindeutig vorhanden und direkt ablesbar sind. Er enthält keine
    erfundenen Werte, Kategorien, Achsentitel, Farben, Trends, Ursachenbehauptungen
    oder darüber hinausgehende Interpretationen des Diagramms.

    Bewertungsskala (1–5):
    1 = stark fehlerhaft:
        - mehrere falsche oder erfundene Inhalte; deutliche Abweichung von Diagramm/Daten;
          starke eigene Interpretation oder Erklärung
    2 = fehlerhaft:
        - einige falsche Elemente oder klare Halluzinationen; einzelne Interpretationen,
          die nicht direkt aus den Daten hervorgehen
    3 = teilweise korrekt:
        - Mehrheit der Inhalte korrekt, aber einzelne Ungenauigkeiten, überzogene
          Interpretationen oder kleinere Halluzinationen
    4 = weitgehend korrekt:
        - überwiegend faktentreu; nur minimale, nicht schwerwiegende Abweichungen,
          kaum bis keine Interpretation über direkt Erkennbares hinaus
    5 = vollständig korrekt:
        - keine Halluzinationen; alle genannten Fakten stimmen mit Bild und Daten überein;
          es werden nur Informationen beschrieben, die eindeutig aus Diagramm oder CSV
          ablesbar sind, ohne eigene Deutung oder Erklärung.

    Analyseleitfaden (intern anwenden, NICHT im Output ausgeben):
    - Prüfe ausschließlich inhaltliche Korrektheit, NICHT die Art der Formulierung oder Tonalität.
    - Prüfe, ob der Alt-Text erfundene:
      - Kategorien, Werte, Einheiten, Zeiträume, Achsentitel, Farben oder Beziehungen nennt.
    - Prüfe, ob Interpretationen oder Erklärungen vorkommen, die über das direkt Sichtbare
      hinausgehen, z. B.:
      - Ursachen („weil“, „aufgrund“, „infolge“)
      - Vermutungen („vermutlich“, „scheint“, „wahrscheinlich“)
      - Bewertungen von Trends („nimmt stark zu“, „bricht ein“), wenn diese nicht klar
        aus den Daten ableitbar sind.
    - Vergleiche alle genannten Fakten mit Bild und CSV:
      - Stimmen Größenordnungen, Richtungen, Kategorien, Zeiträume und Bezeichnungen?
    - Bewerte NICHT:
      - sprachliche Qualität, Neutralität der Wortwahl, Textlänge oder Struktur

    Alt-Text zur Bewertung:
    \"\"\"{alt_text}\"\"\"


    ============================================================
    AUSGABEFORMAT (STRICT)
    Gib ausschließlich Folgendes aus:

    Reason: <1-2 Sätze>
    Score: <Zahl von 1–5>

    Keine Erklärungen oder Texte.
    """



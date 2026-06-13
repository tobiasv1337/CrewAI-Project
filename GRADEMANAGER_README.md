# TU Notenmanager

Interaktives Python-Tool zur Planung, Verwaltung und Berechnung von Abschlussnoten (Bachelor/Master).  
Der Fokus liegt auf sauberer Trennung zwischen Frontend (Streamlit) und Studiengangslogik sowie einem modularen Aufbau für zukünftige Studiengänge.

## Highlights
- Modulverwaltung (anlegen, bearbeiten, löschen)
- Detailansicht pro Modul (Beschreibung, Uni-Link, Uploads, GitHub-Link)
- Regelprüfung pro Studiengang (anpassbar)
- Notenberechnung inkl. 30-LP-Streichliste
- Szenarien: Current, Forecast, Best Case, Worst Case
- Visualisierungen (Notenvergleich, LP-Verteilung, Semesterplan, Sensitivität)
- Studienplan-Export als Markdown und PDF

## Schnellstart
```bash
# Projekt initialisieren (erstellt .venv und installiert Abhängigkeiten)
uv sync

# Anwendung starten
uv run streamlit run app.py
```

## Projektstruktur
```
app.py                 # Streamlit Einstiegspunkt
assets/style.css       # UI Theme
core/                  # Studiengangslogik (backend)
  calculations.py      # Notenberechnung + Szenarien
  analytics.py         # Daten für Visualisierungen
  interfaces.py        # Typen & Protokolle
  manager.py           # Facade für Programme
  models.py            # Datenmodell (Module)
  rules.py             # Regel- und Streichlisten-Modelle
  registry.py          # Studiengang-Registry
  impl/tu_berlin.py     # Beispielprogramm TU Berlin M.Sc.
ui/                    # Streamlit Seiten
data/                  # Persistenz (modules.json, uploads/)
```

## Datenmodell (Module)
Jedes Modul wird über `core/models.py` definiert:
- `name`: Modulname
- `state`: Status (`Completed`, `In Progress`, `Planned`)
- `area`: Abschluss-Bucket / Bereich (z. B. `Electives`, `Free Choice`, `Master Thesis`, `Additional Courses`)
- `program_key`: Studiengang-Schlüssel (ermöglicht Master/Bachelor parallel)
- `cp`: Leistungspunkte
- `grade`: Note (falls abgeschlossen)
- `estimated_grade`: Geschätzte Note (Planung)
- `is_graded`: Benotet / unbenotet
- `term`: Semesterlabel wie `WS 25/26` oder `SS 26`
- `catalogs`: Kataloge / Prüfungsordnungskategorien (z. B. `Cognitive Systems`, `Information Systems`)
- `catalog_mode`: `Auto` nutzt MOSES-Kataloge für den jeweiligen Studiengang, `Manual` nutzt die gespeicherte Auswahl
- `extra_registrations`: weitere Studiengänge, für die dasselbe Modul zählt; jede Registrierung hat eigene `area`- und `catalogs`-Werte
- `description`: Beschreibungstext
- `url`: Uni-Modulbeschreibung
- `github_url`: Link zum Code
- `notes`: Kommentare
- `attachments`: Uploads (PDF/ZIP/…)
- optional: `start_date` / `end_date` (YYYY-MM-DD), z. B. für Thesis-Deadline Checks

## Notenberechnung (TU Berlin Beispiel)
Die Logik ist in `core/calculations.py` implementiert und pro Studiengang konfigurierbar:
- LP-gewichtetes Mittel aller benoteten Module
- Trunkierung auf eine Dezimalstelle **ohne Rundung**
- Masterarbeit wird niemals gestrichen
- 30-LP-Streichliste:
  - Maximal 30 LP ausgeschlossen
  - Pflicht: 12 LP aus Freier Wahl
  - Priorität: unbenotete Module, dann schlechteste Noten

### Szenarien
Die Berechnung unterstützt folgende Modi:
- **Current**: Nur abgeschlossene Module zählen
- **Forecast**: Offene Module nutzen `estimated_grade`
- **Best Case**: Offene Module = 1.0
- **Worst Case**: Offene Module = 4.0

## Regelprüfung
Regeln sind in `core/rules.py` modelliert und pro Studiengang definierbar.  
Im TU-Berlin-Beispiel werden u. a. geprüft:
- Gesamt-LP (120)
- Freie Wahl (12 LP)
- Masterarbeit vorhanden
- Projekt/Seminar vorhanden (Warnung)
- Doppelte Module (Warnung)
- Fehlende Noten bei abgeschlossenen Modulen (Warnung)

Hinweis: Die Regeln sind bewusst minimal gehalten und sollen je nach offizieller Prüfungsordnung angepasst werden.

## Persistenz
Module werden in `data/modules.json` gespeichert.  
Uploads landen in `data/uploads/` und werden in `attachments` referenziert.

Hinweis zur Semantik:
- `area` ist das Bucket innerhalb des Abschlusses.
- `catalogs` sind die fachlichen Kataloge / study fields.
- `catalog_mode=Auto` leitet die Kataloge pro Studiengang aus `moses.normalized_catalogs_by_program` ab; `catalog_mode=Manual` schützt die manuelle Katalogauswahl vor MOSES-Refreshes.
- Bei Cross-Degree-Registrierungen gehören `area` und `catalogs` zur jeweiligen Registrierung; Note, LP, Semester, Status und MOSES-Metadaten bleiben ein gemeinsamer Moduldatensatz.
- Im CS-Master wird die primäre Fachrichtung automatisch aus den `catalogs` der Wahlpflichtmodule abgeleitet.
- `Additional Courses` (Zusatzmodule) werden gespeichert, zählen aber nicht für Abschlussregeln oder GPA.

## Erweiterung: Neue Studiengänge
1. Neue Klasse in `core/impl/` anlegen (z. B. `my_degree.py`)
2. Regeln + Streichliste konfigurieren
3. In `core/registry.py` registrieren

Beispiel:
```python
from core.impl.my_degree import MyDegreeProgram
PROGRAM_REGISTRY["My University - Data Science (M.Sc.)"] = MyDegreeProgram
```

## Hinweise
- Die UI ist vollständig getrennt von der Studiengangslogik.
- Das Projekt kann später problemlos in FastAPI + React integriert werden (Core bleibt identisch).

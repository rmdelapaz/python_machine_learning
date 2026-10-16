#!/usr/bin/env python3
"""
Assemble the three combined instructor-kit books from the per-session source docs.

Kit:    Advanced Machine Learning (Python Machine Learning, Modules 7 to 13), 13 sessions x 3 hours.
Reads:  source/module-NN/{facilitator-guide,participant-workbook,answer-key}.html   (NN = 01 ... 13)
        ../../course-config.json   (lesson titles, so the contents page always matches the course)
Writes: ./participant-workbook.html, ./facilitator-guide.html, ./answer-key.html
        (the hand-out documents: a cover, a contents page, then one chapter per session)

A "module" folder is one 3-hour SESSION of the kit (not a course module): module-07 = Session 7.

A session whose source doc is missing (or has no sheet/foot) is skipped with a warning, so a
partly written kit still builds. The exit code is 0 when every session was found, 1 otherwise;
add --strict to stop at the first missing doc instead.

Links: a source doc sits two folders below the kit root, a book sits in the root, so every
href="../../..." or src="../../..." in a source doc is rewritten (the leading "../../" is dropped).

After editing a session's source doc, run this again, then render the PDFs
(python3 ~/projects/_kit_tools/python_ml/render_pdfs_ml.py advanced, or each book's Print button).

Usage:  python3 build-combined.py [--strict]      (works from any folder)
"""
import json
import re
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
COURSE = "Python Machine Learning"          # the course on the site
KIT_TITLE = "Advanced Machine Learning"     # this kit's name: masthead foot lines and book covers
KIT_KEY = "advanced"                        # folder name under instructor-kit/
AUTHOR = "Ray de la Paz"
SESSIONS = 13
HOURS = 39
FORMAT_CHIP = "13 sessions × 3 h · 39 h"
LESSON_CHIP = "23 lessons"
SCOPE = "Modules 7 to 13"

# (session number, session title, the lesson ids taught in it). Lesson TITLES are read from
# ../../course-config.json when the script runs; do not type them here.
MODULES = [
    (1, "Interpreting Models and SHAP", ["7.1", "7.2"]),
    (2, "Local Explanations, Fairness and K-Means", ["7.3", "8.1"]),
    (3, "Choosing k and Hierarchical Clustering", ["8.2", "8.3"]),
    (4, "Density-Based Clustering and Gaussian Mixtures", ["8.4", "8.5"]),
    (5, "Anomaly Detection and PCA", ["8.6", "9.1"]),
    (6, "LDA, t-SNE and UMAP", ["9.2", "9.3"]),
    (7, "Neural Networks and PyTorch", ["10.1", "10.2"]),
    (8, "CNNs and Sequence Models", ["10.3", "10.4"]),
    (9, "Transfer Learning and Text Classification", ["10.5", "11.1"]),
    (10, "Recommendations and Reproducibility", ["11.2", "12.1"]),
    (11, "Serving, Monitoring and the Churn Case Study", ["12.2", "13.1"]),
    (12, "Capstone Kickoff and Studio I", ["13.2"]),
    (13, "Capstone Studio II and Showcase", ["13.2"]),
]

DOCS = {
    "participant-workbook": dict(
        base="participant-workbook.html", role="Participant Workbook",
        eyebrow=f"{COURSE} · {KIT_TITLE} · Complete Workbook",
        sub=f"Your working notebook for all {SESSIONS} sessions of {KIT_TITLE}. Read the key ideas, study the worked examples, do the guided labs, predict before you run, and check yourself with the self-check quizzes.",
        chips=[f'<span class="chip"><b>{SESSIONS}</b> sessions · {LESSON_CHIP}</span>', '<span class="chip">Write-in exercises &amp; self-checks</span>', '<span class="chip">Name: ____________________</span>'],
        toolbar=("Student workbook —", "write in it · bring it to every session")),
    "facilitator-guide": dict(
        base="facilitator-guide.html", role="Facilitator Guide",
        eyebrow=f"{COURSE} · {KIT_TITLE} · Complete Facilitator Guide",
        sub=f"The full teaching manual for the {SESSIONS}-session {KIT_TITLE} kit ({SCOPE}; one 3-hour session, two lessons each). Timing, live-demo scripts, the mistakes to pre-empt, lab facilitation, and checks for understanding, for every session.",
        chips=[f'<span class="chip"><b>{SESSIONS}</b> sessions · {LESSON_CHIP}</span>', f'<span class="chip">{FORMAT_CHIP}</span>', '<span class="chip">Instructor copy · not for students</span>'],
        toolbar=("Instructor copy —", "contains answers &amp; timing · not for students")),
    "answer-key": dict(
        base="answer-key.html", role="Answer Key",
        eyebrow=f"{COURSE} · {KIT_TITLE} · Complete Answer Key",
        sub=f"Every self-check item answered and explained, plus reference notes for each lab, for all {SESSIONS} sessions. Use for grading and reteaching. Confidential instructor material.",
        chips=[f'<span class="chip"><b>{SESSIONS}</b> sessions</span>', f'<span class="chip">{SESSIONS * 12} self-check items + lab solutions</span>', '<span class="chip">Confidential · do not distribute</span>'],
        toolbar=("Instructor copy —", "answers &amp; solutions · not for students")),
}

SHEET_OPEN = '<div class="sheet">'
FOOT_OPEN = '<div class="foot">'


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def folder(num):
    """Session 7 -> 'module-07'."""
    return f"module-{num:02d}"


def lesson_titles():
    """{'7.1': 'Interpreting Models', ...} from the course's course-config.json."""
    for cfg in (KIT.parent.parent / "course-config.json", KIT.parent / "course-config.json", KIT / "course-config.json"):
        if cfg.is_file():
            break
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"Cannot read lesson titles from {cfg}: {e}")
    return {les["id"]: les["title"] for mod in data["modules"] for les in mod["lessons"]}


def lessons_for(num, titles=None):
    """[('7.1', 'Interpreting Models'), ('7.2', 'SHAP Values')] for a session (used by the checker too)."""
    titles = titles or lesson_titles()
    ids = next(ids for n, _, ids in MODULES if n == num)
    missing = [i for i in ids if i not in titles]
    if missing:
        raise SystemExit(f"Lesson id(s) {missing} are not in course-config.json")
    return [(i, titles[i]) for i in ids]


def extract_body(path):
    """Inner content of .sheet minus the trailing .foot block; None (with a reason) if it cannot be cut."""
    try:
        html = path.read_text(encoding="utf-8")
    except OSError:
        return None, "file not found"
    if html.count(SHEET_OPEN) != 1 or html.count(FOOT_OPEN) != 1:
        return None, f'needs exactly one {SHEET_OPEN} and one {FOOT_OPEN}'
    i = html.index(SHEET_OPEN) + len(SHEET_OPEN)
    j = html.index(FOOT_OPEN)
    if j < i:
        return None, "the foot comes before the sheet"
    body = html[i:j].strip()
    body = re.sub(r'\b(href|src)="\.\./\.\./', r'\1="', body)
    return body, ""


def build(key, titles, strict=False):
    d = DOCS[key]
    toc_rows, chapters, skipped = [], [], []
    for num, title, _ in MODULES:
        path = KIT / "source" / folder(num) / d["base"]
        body, why = extract_body(path)
        if body is None:
            if strict:
                raise SystemExit(f"Session {num:02d}: {path.relative_to(KIT)}: {why}")
            if why != "file not found":       # a doc that exists but cannot be cut deserves its own line
                print(f"WARNING: Session {num:02d} skipped in {d['base']}: {path.relative_to(KIT)}: {why}")
            skipped.append(num)
            toc_rows.append(f'<li class="mod missing"><span class="mnum">S{num:02d}</span>{esc(title)} (not written yet)</li>')
            continue
        toc_rows.append(f'<li class="mod"><span class="mnum">S{num:02d}</span>{esc(title)}</li>')
        for lc, lt in lessons_for(num, titles):
            toc_rows.append(f'<li class="les"><span class="lc">{lc}</span><span>{esc(lt)}</span></li>')
        chapters.append(f'<section class="modwrap">\n{body}\n</section>')
    toc = "\n        ".join(toc_rows)
    mods = "\n\n".join(chapters)
    scope = f"all {SESSIONS} sessions" if not skipped else f"{len(chapters)} of {SESSIONS} sessions (draft)"

    out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{d['role']} | {KIT_TITLE}</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=Source+Sans+3:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@500;600&display=swap">
<link rel="stylesheet" href="kit.css">
</head>
<body>

<div class="toolbar">
  <span class="lbl">{d['toolbar'][0]}</span>
  <button type="button" onclick="window.print()">🖨 Print / Save as PDF</button>
  <span class="hint">{d['toolbar'][1]}</span>
</div>

<div class="sheet">

  <header class="cover">
    <div class="kit-eyebrow">{d['eyebrow']}</div>
    <h1>{KIT_TITLE}</h1>
    <p class="sub">{d['sub']}</p>
    <div class="meta-row">{''.join(d['chips'])}</div>
  </header>

  <nav class="toc">
    <h2>Contents</h2>
    <ol>
        {toc}
    </ol>
  </nav>

{mods}

  <div class="foot">
    <div class="who"><span class="bio">{KIT_TITLE} · {d['role']}</span></div>
    <div class="ver">Complete {d['role']} · {scope} · v1.0 · &copy; 2026 {AUTHOR}</div>
  </div>

</div>
</body>
</html>
"""
    (KIT / d["base"]).write_text(out, encoding="utf-8")
    if skipped:
        print(f"WARNING: {d['base']}: {len(skipped)} session(s) skipped (source doc missing or not cuttable): "
              + ", ".join(f"{n:02d}" for n in skipped))
    print(f"wrote {d['base']}  ({len(out) // 1024} KB, {len(chapters)} of {len(MODULES)} sessions)")
    return skipped


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        return 0
    strict = "--strict" in argv
    if not (KIT / "source").is_dir():
        raise SystemExit(f"source/ folder not found next to {Path(__file__).name}")
    titles = lesson_titles()
    for num, _, _ in MODULES:
        lessons_for(num, titles)          # fail early on a lesson id that left the course
    skipped = set()
    for k in DOCS:
        skipped |= set(build(k, titles, strict))
    if skipped:
        print(f"Built with {len(skipped)} session(s) missing at least one doc: "
              + ", ".join(f"{n:02d}" for n in sorted(skipped)))
        return 1
    print(f"Done: all {SESSIONS} sessions in all three books. Now render the PDFs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Release gates: checks the published repo must satisfy.

Unresolved release blockers are reported as pytest WARNINGS (visible in the
run's warnings summary) and printed, so the suite stays green while every
blocker remains loudly visible until resolved:

- TODO markers anywhere in tracked files
- README/CITATION placeholders: the arXiv id (XXXX.XXXXX), placeholder BibTeX key
"""
import re
import subprocess
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]

# vendored third-party code is the ONLY exclusion (upstream TODOs are theirs);
# this file itself is skipped only because it spells the pattern it greps for
VENDORED_PREFIXES = ('src/mamba_tiny/', 'src/mqar_zoology/')
THIS_FILE = 'test/system/dry/test_release_gates.py'


def _todo_hits(text, name):
    """(name, lineno, trimmed line) for every TODO marker, for a readable report."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if re.search(r'(?i)\btodo\b', line):
            hits.append((name, lineno, line.strip()[:100]))
    return hits


def _warn(title, hits):
    print(f'{title} ({len(hits)} finding(s), release blockers):')
    for name, lineno, line in hits:  # one warning per finding, so pytest counts each
        print(f'  {name}:{lineno}: {line}')
        warnings.warn(f'{title}: {name}:{lineno}: {line}', stacklevel=2)


def test_no_todo_markers_in_any_tracked_file():
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=PROJECT_DIR, text=True).split()
    hits = []
    for name in tracked:
        if name.startswith(VENDORED_PREFIXES) or name == THIS_FILE:
            continue
        path = PROJECT_DIR / name
        if not path.is_file():  # e.g. the paper submodule gitlink
            continue
        data = path.read_bytes()
        if b'\0' in data[:8192]:  # binary
            continue
        hits += _todo_hits(data.decode('utf-8', errors='ignore'), name)
    if hits:
        _warn('Unresolved TODO markers in tracked files', hits)


def _placeholder_hits(name, patterns):
    text = (PROJECT_DIR / name).read_text()
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if any(re.search(p, line) for p in patterns):
            hits.append((name, lineno, line.strip()[:100]))
    return hits


def test_arxiv_id_placeholders_warn():
    """arXiv id still unset: badge/link/BibTeX in README, journal in CITATION.cff."""
    hits = _placeholder_hits('README.md', [r'XXXX\.XXXXX'])
    hits += _placeholder_hits('CITATION.cff', [r'XXXX\.XXXXX'])
    if hits:
        _warn('arXiv id placeholders still unset - these XXXX.XXXXX links are broken/temporary until the arXiv id is assigned', hits)


def test_name_placeholders_warn():
    """Placeholder names: the BibTeX citation key and the temporary author homepage link."""
    hits = _placeholder_hits('README.md', [r'koren2026x{3,}'])
    if hits:
        _warn('Name placeholders / temporary links still in place', hits)

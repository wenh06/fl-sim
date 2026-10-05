"""Build the multi-version, multi-language documentation.

Builds the documentation for ``latest`` (the current working tree, usually
the master branch) and for every version declared in ``versions.yaml``, in
every configured language, into a directory layout ready to be published on
GitHub Pages:

    pages/
    ├── index.html            (redirect to latest/<language>)
    ├── .nojekyll
    └── <version>/
        ├── en/
        └── zh_CN/

Each build keeps ``conf.py`` and ``versions.yaml`` from the starting commit
(usually master), so that the switcher logic stays up to date even when
building old tags.

Usage::

    python build_docs.py [--pages-root URL] [--output DIR] [--latest-only]

"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent
SOURCE_DIR = DOCS_ROOT / "source"
PAGES_ROOT_DEFAULT = "https://wenh06.github.io/fl-sim"
OUTPUT_DEFAULT = DOCS_ROOT / "pages"
LANGUAGES = ["en", "zh_CN"]
REDIRECT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>fl-sim documentation</title>
  <script>
    // Redirect to the latest docs, preferring the browser language.
    (function () {
      var lang = (navigator.language || navigator.userLanguage || "en").toLowerCase();
      var target = lang.indexOf("zh") === 0 ? "latest/zh_CN/" : "latest/en/";
      window.location.replace(target);
    })();
  </script>
  <noscript><meta http-equiv="refresh" content="0; url=latest/en/"></noscript>
</head>
<body>
  <p>Redirecting to the documentation...</p>
  <p><a href="latest/en/">English</a> | <a href="latest/zh_CN/">简体中文</a></p>
</body>
</html>
"""


def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def build_one(version: str, language: str, output_dir: Path, pages_root: str) -> None:
    """Build one (version, language) pair and move the result into place."""
    env = os.environ.copy()
    env.update(
        {
            "SPHINX_LANGUAGE": language,
            "CURRENT_VERSION": version,
            "PAGES_ROOT": pages_root,
            "BUILD_ALL_DOCS": "1",
        }
    )
    tmp_dir = DOCS_ROOT / "build" / f"pages-{version}-{language}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    run([sys.executable, "-m", "sphinx", "-b", "html", SOURCE_DIR, tmp_dir], env=env)
    dst = output_dir / version / language
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_dir), str(dst))
    print(f"built {version}/{language}", flush=True)


def write_redirect(output_dir: Path) -> None:
    (output_dir / "index.html").write_text(REDIRECT_PAGE, encoding="utf-8")
    # keep GitHub Pages from running Jekyll (which would drop `_static` etc.)
    (output_dir / ".nojekyll").touch()


COMPAT_REDIRECT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>fl-sim documentation</title>
  <script>
    (function () {
      var lang = (navigator.language || navigator.userLanguage || "en").toLowerCase();
      var target = lang.indexOf("zh") === 0 ? "zh_CN/" : "en/";
      window.location.replace(target);
    })();
  </script>
  <noscript><meta http-equiv="refresh" content="0; url=en/"></noscript>
</head>
<body>
  <p>Redirecting to the documentation...</p>
  <p><a href="en/">English</a> | <a href="zh_CN/">简体中文</a></p>
</body>
</html>
"""


def refresh_compat_layer(output_dir: Path, languages: list) -> None:
    """Maintain the compatibility layer in ``docs/build``.

    A web server may serve ``docs/build`` directly (e.g. via a symlink like
    ``/var/www/html/fl-sim -> docs/build``, with the old flat layout
    ``build/en`` + ``build/zh_CN`` + ``build/index.html``). After each build,
    recreate that layer as symlinks into ``pages/latest/<lang>`` plus a
    language-aware redirect page, so the served URLs keep working and always
    reflect the latest build. ``make clean`` may wipe ``docs/build`` freely;
    this function brings the layer back.
    """
    build_dir = DOCS_ROOT / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    for language in languages:
        link = build_dir / language
        target = os.path.relpath(output_dir / "latest" / language, build_dir)
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        os.symlink(target, link)
    (build_dir / "index.html").write_text(COMPAT_REDIRECT_PAGE, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the multi-version, multi-language documentation.")
    parser.add_argument("--pages-root", default=PAGES_ROOT_DEFAULT, help="Root URL of the pages site")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT, help="Output directory")
    parser.add_argument("--latest-only", action="store_true", help="Skip tagged versions (quick local build of latest only)")
    args = parser.parse_args()

    import yaml

    versions_file = DOCS_ROOT / "versions.yaml"
    versions_cfg = {}
    if versions_file.exists():
        versions_cfg = yaml.safe_load(versions_file.read_text()) or {}

    start_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "docs/source/conf.py", "docs/versions.yaml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty and not args.latest_only:
        print(
            "WARNING: uncommitted changes in docs/source/conf.py or docs/versions.yaml; "
            "tagged versions will be built with the committed versions.",
            flush=True,
        )

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # "latest" is built from the current working tree, no checkout needed
    for language in LANGUAGES:
        build_one("latest", language, output_dir, args.pages_root)

    # tagged versions: checkout the tag, keep conf.py/versions.yaml from the starting commit
    if not args.latest_only:
        for version, cfg in versions_cfg.items():
            tag = cfg.get("tag", version)
            languages = cfg.get("languages", LANGUAGES)
            run(["git", "checkout", tag], cwd=REPO_ROOT)
            run(["git", "checkout", start_rev, "--", "docs/source/conf.py", "docs/versions.yaml"], cwd=REPO_ROOT)
            for language in languages:
                build_one(version, language, output_dir, args.pages_root)
            run(["git", "checkout", start_rev], cwd=REPO_ROOT)
            run(["git", "checkout", start_rev, "--", "docs/source/conf.py", "docs/versions.yaml"], cwd=REPO_ROOT)

    write_redirect(output_dir)
    refresh_compat_layer(output_dir, LANGUAGES)
    print(f"Done. Pages layout at {output_dir}", flush=True)


if __name__ == "__main__":
    main()

"""
Builds the final clean source code ZIP for academic submission.
Includes: all .py, .yaml, .md, .txt, .csv, .json, .html, .pptx files
Excludes: .git/, __pycache__/, .pytest_cache/, .pyc, video files, model weights
"""
import os, zipfile

BASE = r"d:\sports analytics"
OUT_ZIP = r"d:\sports analytics\outputs\reports\SportsVision_AI_SourceCode.zip"

# File extensions to include
INCLUDE_EXTS = {
    ".py", ".yaml", ".yml", ".md", ".txt", ".csv", ".json",
    ".html", ".pptx", ".gitignore"
}

# Directories / path fragments to skip entirely
SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache",
    "outputs\\videos",       # large video outputs
    "data\\sample",          # large .mp4 source videos
    "models\\weights",       # large .pt model files
    "brain",                 # gemini internal
}

def should_skip(rel_path: str) -> bool:
    parts = rel_path.replace("/", "\\").split("\\")
    for skip in SKIP_DIRS:
        if any(skip in p for p in parts):
            return True
    return False

files_added = []
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(BASE):
        rel_root = os.path.relpath(root, BASE)
        if should_skip(rel_root):
            dirs.clear()
            continue
        # Prune subdirs in-place to avoid walking into skipped dirs
        dirs[:] = [d for d in dirs if not should_skip(os.path.join(rel_root, d))]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in INCLUDE_EXTS:
                continue
            full_path = os.path.join(root, fname)
            arc_name = os.path.join("SportsVision-ai", rel_root, fname)
            arc_name = arc_name.replace("\\", "/")
            zf.write(full_path, arc_name)
            files_added.append(arc_name)
            print(f"  Added: {arc_name}")

print(f"\n✅ ZIP created: {OUT_ZIP}")
print(f"   Total files: {len(files_added)}")
# Show size
size_mb = os.path.getsize(OUT_ZIP) / (1024 * 1024)
print(f"   ZIP size: {size_mb:.2f} MB")

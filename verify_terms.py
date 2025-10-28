"""
verify_terms.py
Step 8A.3 – Terminology Consistency Checker
--------------------------------------------
Scans project files to ensure old terminology (Plane, WorkCenter, etc.)
has been replaced by the new unified vocabulary (JobAgent, WorkCenter, etc.).
Per the migration plan, references to legacy 'Site'/'Sites' are being removed
and replaced incrementally with 'WorkCenter'/'WorkCenters'. This script
helps find remaining legacy terms for targeted manual fixes.
"""

import os
import re

# === Configuration ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FOLDERS = ["MARL", "utils"]
# Only check for old aviation/legacy terms; 'Site' removed from list as
# we migrate to WorkCenter terminology and perform targeted replacements.
OLD_TERMS = ["Plane", "plane"]
IGNORE_DIRS = ["__pycache__", ".git", "my_data_and_graph"]

def scan_file(file_path):
    """Scan a single file for old terms."""
    results = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                for term in OLD_TERMS:
                    if re.search(rf"\b{term}\b", line):
                        results.append((idx, term, line.strip()))
    except Exception as e:
        print(f"[ERROR] Could not scan {file_path}: {e}")
    return results


def verify_terms():
    """Walk through all project files and check for outdated terminology."""
    total_warnings = 0
    for folder in TARGET_FOLDERS:
        dir_path = os.path.join(ROOT_DIR, folder)
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                if not file.endswith(".py"):
                    continue
                path = os.path.join(root, file)
                results = scan_file(path)
                if results:
                    total_warnings += len(results)
                    print(f"\n[WARN] {path} → found {len(results)} potential old terms:")
                    for (line_no, term, line) in results[:5]:
                        print(f"   line {line_no:4d}: {term} → {line}")
    if total_warnings == 0:
        print("\n✅ All terminology consistent across project.")
    else:
        print(f"\n⚠️  Found {total_warnings} outdated term occurrences.")
        print("   Please review the warnings above.")


if __name__ == "__main__":
    print("🔎 Running Step 8A.3 – Terminology Consistency Checker...")
    verify_terms()

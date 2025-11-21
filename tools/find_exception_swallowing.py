#!/usr/bin/env python3
"""
Detect catch-all exception handlers in codebase.

Usage:
    python tools/find_exception_swallowing.py
    python tools/find_exception_swallowing.py --verbose
    python tools/find_exception_swallowing.py --json output.json
"""
import re
import os
import json
import argparse


def find_exception_swallowing(root_dir, verbose=False):
    """Find catch-all exception handlers."""
    
    # Pattern 1: except Exception: pass
    pattern1 = re.compile(
        r'except\s+Exception\s*:\s*\n\s*pass\b',
        re.MULTILINE
    )
    
    # Pattern 2: except Exception: return/continue
    pattern2 = re.compile(
        r'except\s+Exception\s*:\s*\n\s*(return|continue)',
        re.MULTILINE
    )
    
    # Pattern 3: except: (no exception type)
    pattern3 = re.compile(
        r'except\s*:\s*\n\s*(pass|return|continue)',
        re.MULTILINE
    )
    
    patterns = [
        (pattern1, 'except Exception: pass'),
        (pattern2, 'except Exception: return/continue'),
        (pattern3, 'bare except'),
    ]
    
    results = []
    for root, dirs, files in os.walk(root_dir):
        # Skip non-code directories
        skip_dirs = ['archive', 'backup', '__pycache__', '.git', 'my_data_and_graph', 'venv']
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if not file.endswith('.py'):
                continue
            
            path = os.path.join(root, file)
            rel_path = os.path.relpath(path, root_dir)
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, pattern_name in patterns:
                    matches = pattern.finditer(content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Extract context
                        lines = content.split('\n')
                        start_line = max(0, line_num - 4)
                        end_line = min(len(lines), line_num + 3)
                        context = '\n'.join(lines[start_line:end_line])
                        
                        results.append({
                            'file': rel_path,
                            'line': line_num,
                            'pattern': pattern_name,
                            'snippet': match.group(0),
                            'context': context if verbose else None,
                        })
            except Exception as e:
                print(f"Warning: Failed to process {path}: {e}")
    
    return results


def categorize_results(results):
    """Categorize results by priority."""
    
    high_priority = ['rollout.py', 'environment.py', 'runner.py']
    medium_priority = ['policy', 'qmix', 'epsilon', 'actor_critic']
    
    categories = {
        'high': [],
        'medium': [],
        'low': [],
    }
    
    for r in results:
        file = r['file']
        
        if any(hp in file for hp in high_priority):
            categories['high'].append(r)
        elif any(mp in file for mp in medium_priority):
            categories['medium'].append(r)
        else:
            categories['low'].append(r)
    
    return categories


def print_summary(results, categories):
    """Print summary report."""
    
    print(f"\n{'='*70}")
    print(f"EXCEPTION SWALLOWING DETECTION REPORT")
    print(f"{'='*70}\n")
    
    print(f"Total locations found: {len(results)}\n")
    
    # By priority
    print("By Priority:")
    print(f"  🔴 HIGH:   {len(categories['high'])} locations")
    print(f"  🟡 MEDIUM: {len(categories['medium'])} locations")
    print(f"  🟢 LOW:    {len(categories['low'])} locations")
    print()
    
    # By file
    by_file = {}
    for r in results:
        file = r['file']
        if file not in by_file:
            by_file[file] = []
        by_file[file].append(r)
    
    print(f"By File ({len(by_file)} files):")
    for file, matches in sorted(by_file.items(), key=lambda x: -len(x[1])):
        priority = '🔴' if any(r in categories['high'] for r in matches) else \
                   '🟡' if any(r in categories['medium'] for r in matches) else '🟢'
        print(f"  {priority} {file}: {len(matches)} locations")
        for m in matches[:3]:  # Show first 3
            print(f"      Line {m['line']}: {m['pattern']}")
        if len(matches) > 3:
            print(f"      ... and {len(matches) - 3} more")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Find exception swallowing in code')
    parser.add_argument('--verbose', '-v', action='store_true', help='Include context')
    parser.add_argument('--json', '-j', type=str, help='Output JSON file')
    parser.add_argument('--root', '-r', type=str, default='.', help='Root directory')
    
    args = parser.parse_args()
    
    print(f"Scanning {args.root} for exception swallowing...")
    results = find_exception_swallowing(args.root, verbose=args.verbose)
    categories = categorize_results(results)
    
    print_summary(results, categories)
    
    # JSON output
    if args.json:
        output = {
            'total': len(results),
            'categories': {k: len(v) for k, v in categories.items()},
            'results': results,
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Detailed results written to: {args.json}")


if __name__ == "__main__":
    main()

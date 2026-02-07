#!/usr/bin/env python3
"""
Script to analyze relationships between JSONL and TXT files across all splits, languages, and dialects.
Generates statistics showing how example_ids link between files.
"""

import json
import sys
import csv
import gzip
from pathlib import Path
from collections import defaultdict

def load_txt_ids(txt_file):
    """Load all IDs from a txt file (first column, tab-separated)."""
    ids = set()
    if not txt_file.exists():
        return ids
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if parts:
                    ids.add(parts[0])
    return ids

def load_jsonl_example_ids(jsonl_file):
    """Load all example_ids from a JSONL file (handles both .jsonl and .jsonl.gz)."""
    example_ids = []
    if not jsonl_file.exists():
        return example_ids
    
    # Determine if file is gzipped
    open_func = gzip.open if jsonl_file.suffix == '.gz' or '.gz' in jsonl_file.name else open
    mode = 'rt' if jsonl_file.suffix == '.gz' or '.gz' in jsonl_file.name else 'r'
    
    try:
        with open_func(jsonl_file, mode, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    example_id = data.get('example_id')
                    if example_id is not None:
                        example_ids.append(str(example_id))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Warning: Failed to read {jsonl_file}: {e}", file=sys.stderr)
    
    return example_ids

def load_metadata_ids(metadata_file):
    """Load all example_ids from metadata.csv file."""
    ids = set()
    if not metadata_file.exists():
        return ids
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                example_id = row.get('example_id', '').strip()
                if example_id:
                    ids.add(example_id)
    except Exception:
        pass
    return ids

def load_audio_file_ids(audio_dir):
    """Load all example_ids from audio file names."""
    ids = set()
    if not audio_dir.exists() or not audio_dir.is_dir():
        return ids
    for audio_file in audio_dir.glob('*.wav'):
        ids.add(audio_file.stem)
    return ids

def find_audio_dir(dialect_dir, language):
    """Find the audio directory (wav_eng, wav_ara, etc.) based on language."""
    # Map language codes to audio directory names
    audio_dir_map = {
        'eng': 'wav_eng',
        'ara': 'wav_ara',
        'ben': 'wav_ben',
        'kor': 'wav_kor',
        'swa': 'wav_swa',
    }
    
    audio_dir_name = audio_dir_map.get(language, f'wav_{language}')
    audio_dir = dialect_dir / audio_dir_name
    
    # If the expected directory doesn't exist, try to find any wav_* directory
    if not audio_dir.exists():
        for wav_dir in dialect_dir.glob('wav_*'):
            if wav_dir.is_dir():
                return wav_dir
    
    return audio_dir

def find_jsonl_files(dialect_dir):
    """Find all JSONL files in a dialect directory."""
    jsonl_files = {}
    for pattern in ['*.jsonl', '*.jsonl.gz']:
        for jsonl_file in dialect_dir.glob(pattern):
            # Extract variant from filename
            # e.g., english.test.gbr-en-GB.jsonl -> 'GB'
            name = jsonl_file.stem if jsonl_file.suffix != '.gz' else jsonl_file.stem.replace('.jsonl', '')
            parts = name.split('-')
            if len(parts) >= 2:
                variant = parts[-1]  # GB, US, AU, etc.
                jsonl_files[variant] = jsonl_file
    return jsonl_files

def find_txt_files(dialect_dir):
    """Find all TXT files in a dialect directory."""
    txt_files = {}
    for txt_file in dialect_dir.glob('*.txt'):
        # Extract variant from filename
        name = txt_file.stem
        parts = name.split('-')
        if len(parts) >= 2:
            variant = parts[-1]  # GB, US, AU, etc.
            txt_files[variant] = txt_file
    return txt_files

def analyze_dialect(dialect_dir, language):
    """Analyze a single dialect directory."""
    dialect_name = dialect_dir.name
    
    # Find files
    jsonl_files = find_jsonl_files(dialect_dir)
    txt_files = find_txt_files(dialect_dir)
    metadata_file = dialect_dir / "metadata.csv"
    audio_dir = find_audio_dir(dialect_dir, language)
    
    # Load data
    metadata_ids = load_metadata_ids(metadata_file)
    audio_ids = load_audio_file_ids(audio_dir)
    
    results = {
        'dialect': dialect_name,
        'jsonl_files': {},
        'txt_files': {},
        'metadata_count': len(metadata_ids),
        'audio_count': len(audio_ids),
        'issues': []
    }
    
    # Analyze each JSONL file
    for variant, jsonl_file in jsonl_files.items():
        jsonl_ids = set(load_jsonl_example_ids(jsonl_file))
        results['jsonl_files'][variant] = {
            'file': jsonl_file.name,
            'count': len(jsonl_ids),
            'ids': jsonl_ids
        }
    
    # Analyze each TXT file
    for variant, txt_file in txt_files.items():
        txt_ids = set(load_txt_ids(txt_file))
        results['txt_files'][variant] = {
            'file': txt_file.name,
            'count': len(txt_ids),
            'ids': txt_ids
        }
    
    # Check for issues
    all_jsonl_ids = set()
    all_txt_ids = set()
    
    for variant_data in results['jsonl_files'].values():
        all_jsonl_ids.update(variant_data['ids'])
    
    for variant_data in results['txt_files'].values():
        all_txt_ids.update(variant_data['ids'])
    
    # Check for count mismatches between corresponding JSONL and TXT files
    for variant in set(list(results['jsonl_files'].keys()) + list(results['txt_files'].keys())):
        jsonl_count = results['jsonl_files'].get(variant, {}).get('count', 0)
        txt_count = results['txt_files'].get(variant, {}).get('count', 0)
        
        if jsonl_count != txt_count:
            results['issues'].append(f"{variant}: Count mismatch - JSONL has {jsonl_count}, TXT has {txt_count}")
    
    # Check JSONL vs TXT matches
    for variant, jsonl_data in results['jsonl_files'].items():
        jsonl_ids = jsonl_data['ids']
        
        # Check against all TXT files
        found_in_txt = False
        for txt_variant, txt_data in results['txt_files'].items():
            if jsonl_ids & txt_data['ids']:
                found_in_txt = True
                break
        
        if not found_in_txt:
            missing_count = len(jsonl_ids - all_txt_ids)
            if missing_count > 0:
                results['issues'].append(f"{variant} JSONL: {missing_count} IDs not in any TXT file")
    
    # Check TXT vs JSONL matches
    for variant, txt_data in results['txt_files'].items():
        txt_ids = txt_data['ids']
        
        if txt_ids - all_jsonl_ids:
            missing_count = len(txt_ids - all_jsonl_ids)
            results['issues'].append(f"{variant} TXT: {missing_count} IDs not in any JSONL file")
    
    # Check metadata and audio
    if metadata_ids and all_jsonl_ids:
        missing_metadata = len(all_jsonl_ids - metadata_ids)
        if missing_metadata > 0:
            results['issues'].append(f"Metadata: {missing_metadata} JSONL IDs missing from metadata.csv")
    
    if audio_ids and all_jsonl_ids:
        missing_audio = len(all_jsonl_ids - audio_ids)
        if missing_audio > 0:
            results['issues'].append(f"Audio: {missing_audio} JSONL IDs missing audio files")
    
    # Calculate common IDs and missing IDs for JSON output
    # Common IDs: IDs present in all three sources (JSONL, TXT, WAV)
    common_ids = sorted(list(all_jsonl_ids & all_txt_ids & audio_ids))
    
    # Missing IDs in each source
    missing_in_jsonl = sorted(list((all_txt_ids | audio_ids) - all_jsonl_ids))
    missing_in_txt = sorted(list((all_jsonl_ids | audio_ids) - all_txt_ids))
    missing_in_wav = sorted(list((all_jsonl_ids | all_txt_ids) - audio_ids))
    
    # Store for JSON output
    results['common_ids'] = common_ids
    results['missing_in_jsonl'] = missing_in_jsonl
    results['missing_in_txt'] = missing_in_txt
    results['missing_in_wav'] = missing_in_wav
    results['all_jsonl_ids'] = sorted(list(all_jsonl_ids))
    results['all_txt_ids'] = sorted(list(all_txt_ids))
    results['all_audio_ids'] = sorted(list(audio_ids))
    
    return results

def main():
    base_dir = Path(__file__).parent
    
    print("="*90)
    print("DATA LINK ANALYSIS - ALL SPLITS, LANGUAGES, AND DIALECTS")
    print("="*90)
    print()
    
    # Process both test and dev splits
    splits = ['test', 'dev']
    all_results = defaultdict(lambda: defaultdict(list))
    
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory not found, skipping...")
            continue
        
        print(f"Processing {split.upper()} split...")
        
        # Find all language directories
        lang_dirs = [d for d in split_dir.iterdir() 
                    if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__']
        lang_dirs.sort()
        
        for lang_dir in lang_dirs:
            language = lang_dir.name
            print(f"  Language: {language}")
            
            # Find all dialect directories
            dialect_dirs = [d for d in lang_dir.iterdir() 
                           if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__']
            dialect_dirs.sort()
            
            for dialect_dir in dialect_dirs:
                print(f"    Analyzing {dialect_dir.name}...", end=' ')
                try:
                    results = analyze_dialect(dialect_dir, language)
                    results['split'] = split
                    results['language'] = language
                    all_results[split][language].append(results)
                    print("✓")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue
        
        print()
    
    # Print summary statistics
    print("="*90)
    print("SUMMARY STATISTICS")
    print("="*90)
    print()
    
    for split in splits:
        if split not in all_results:
            continue
        
        print(f"{split.upper()} SPLIT:")
        print("-" * 90)
        print(f"{'Language':<10} {'Dialect':<12} {'JSONL':<20} {'TXT':<20} {'Metadata':<10} {'Audio':<10} {'Issues':<10}")
        print("-" * 90)
        
        for language in sorted(all_results[split].keys()):
            for r in all_results[split][language]:
                jsonl_info = []
                for variant, data in sorted(r['jsonl_files'].items()):
                    jsonl_info.append(f"{variant}:{data['count']}")
                jsonl_str = ", ".join(jsonl_info) if jsonl_info else "None"
                
                txt_info = []
                for variant, data in sorted(r['txt_files'].items()):
                    txt_info.append(f"{variant}:{data['count']}")
                txt_str = ", ".join(txt_info) if txt_info else "None"
                
                issues_count = len(r['issues'])
                issues_str = f"{issues_count}" if issues_count > 0 else "0"
                
                print(f"{language:<10} {r['dialect']:<12} {jsonl_str:<20} {txt_str:<20} {r['metadata_count']:<10} {r['audio_count']:<10} {issues_str:<10}")
        
        print()
    
    # Print detailed issues
    print("="*90)
    print("DETAILED ISSUES")
    print("="*90)
    print()
    
    total_issues = 0
    for split in splits:
        if split not in all_results:
            continue
        
        for language in sorted(all_results[split].keys()):
            for r in all_results[split][language]:
                if r['issues']:
                    print(f"{split.upper()}/{language}/{r['dialect']}:")
                    for issue in r['issues']:
                        print(f"  - {issue}")
                    print()
                    total_issues += len(r['issues'])
    
    if total_issues == 0:
        print("No issues found! All data links are consistent.")
    else:
        print(f"Total issues found: {total_issues}")
    
    print()
    print("="*90)
    
    # Generate JSON output with common IDs and missing IDs
    json_output = {}
    for split in splits:
        if split not in all_results:
            continue
        
        json_output[split] = {}
        for language in sorted(all_results[split].keys()):
            json_output[split][language] = {}
            
            # Language-level aggregation
            all_dialect_common_ids = []
            all_dialect_jsonl_ids = set()
            all_dialect_txt_ids = set()
            all_dialect_audio_ids = set()
            all_dialect_missing_jsonl = set()
            all_dialect_missing_txt = set()
            all_dialect_missing_wav = set()
            
            # First pass: collect all IDs from all dialects
            for r in all_results[split][language]:
                all_dialect_common_ids.append(set(r.get('common_ids', [])))
                all_dialect_jsonl_ids.update(r.get('all_jsonl_ids', []))
                all_dialect_txt_ids.update(r.get('all_txt_ids', []))
                all_dialect_audio_ids.update(r.get('all_audio_ids', []))
                all_dialect_missing_jsonl.update(r.get('missing_in_jsonl', []))
                all_dialect_missing_txt.update(r.get('missing_in_txt', []))
                all_dialect_missing_wav.update(r.get('missing_in_wav', []))
            
            # Calculate language-level common IDs (present in ALL dialects)
            if all_dialect_common_ids:
                language_common_ids = set.intersection(*all_dialect_common_ids) if len(all_dialect_common_ids) > 1 else all_dialect_common_ids[0]
            else:
                language_common_ids = set()
            
            # Language-level missing IDs
            language_missing_jsonl = sorted(list(all_dialect_missing_jsonl))
            language_missing_txt = sorted(list(all_dialect_missing_txt))
            language_missing_wav = sorted(list(all_dialect_missing_wav))
            
            # Language-level statistics
            lang_total_common = len(language_common_ids)
            lang_total_jsonl = len(all_dialect_jsonl_ids)
            lang_total_txt = len(all_dialect_txt_ids)
            lang_total_audio = len(all_dialect_audio_ids)
            
            lang_jsonl_coverage = (lang_total_common / lang_total_jsonl * 100) if lang_total_jsonl > 0 else 0
            lang_txt_coverage = (lang_total_common / lang_total_txt * 100) if lang_total_txt > 0 else 0
            lang_audio_coverage = (lang_total_common / lang_total_audio * 100) if lang_total_audio > 0 else 0
            lang_overall_coverage = (lang_total_common / max(lang_total_jsonl, lang_total_txt, lang_total_audio) * 100) if max(lang_total_jsonl, lang_total_txt, lang_total_audio) > 0 else 0
            
            # Count dialects and total issues
            total_dialects = len(all_results[split][language])
            total_lang_issues = sum(len(r.get('issues', [])) for r in all_results[split][language])
            
            # Add language-level summary
            json_output[split][language]['_language_summary'] = {
                'common_ids': sorted(list(language_common_ids)),
                'missing_in_jsonl': language_missing_jsonl,
                'missing_in_txt': language_missing_txt,
                'missing_in_wav': language_missing_wav,
                'statistics': {
                    'counts': {
                        'common_ids': lang_total_common,
                        'missing_in_jsonl': len(language_missing_jsonl),
                        'missing_in_txt': len(language_missing_txt),
                        'missing_in_wav': len(language_missing_wav),
                        'total_jsonl': lang_total_jsonl,
                        'total_txt': lang_total_txt,
                        'total_audio': lang_total_audio,
                        'total_unique_ids': len(all_dialect_jsonl_ids | all_dialect_txt_ids | all_dialect_audio_ids),
                        'total_dialects': total_dialects
                    },
                    'coverage_percentages': {
                        'jsonl_coverage': round(lang_jsonl_coverage, 2),
                        'txt_coverage': round(lang_txt_coverage, 2),
                        'audio_coverage': round(lang_audio_coverage, 2),
                        'overall_coverage': round(lang_overall_coverage, 2)
                    },
                    'total_issues': total_lang_issues,
                    'has_issues': total_lang_issues > 0
                }
            }
            
            # Process each dialect
            for r in all_results[split][language]:
                dialect = r['dialect']
                
                # Calculate statistics
                common_ids = r.get('common_ids', [])
                missing_in_jsonl = r.get('missing_in_jsonl', [])
                missing_in_txt = r.get('missing_in_txt', [])
                missing_in_wav = r.get('missing_in_wav', [])
                all_jsonl_ids = r.get('all_jsonl_ids', [])
                all_txt_ids = r.get('all_txt_ids', [])
                all_audio_ids = r.get('all_audio_ids', [])
                
                total_common = len(common_ids)
                total_jsonl = len(all_jsonl_ids)
                total_txt = len(all_txt_ids)
                total_audio = len(all_audio_ids)
                
                # Calculate coverage percentages
                jsonl_coverage = (total_common / total_jsonl * 100) if total_jsonl > 0 else 0
                txt_coverage = (total_common / total_txt * 100) if total_txt > 0 else 0
                audio_coverage = (total_common / total_audio * 100) if total_audio > 0 else 0
                overall_coverage = (total_common / max(total_jsonl, total_txt, total_audio) * 100) if max(total_jsonl, total_txt, total_audio) > 0 else 0
                
                # Variant-level statistics
                variant_stats = {}
                audio_ids_set = set(all_audio_ids)
                for variant, jsonl_data in r.get('jsonl_files', {}).items():
                    variant_jsonl_ids = set(jsonl_data.get('ids', set()))
                    variant_txt_data = r.get('txt_files', {}).get(variant, {})
                    variant_txt_ids = set(variant_txt_data.get('ids', set()))
                    variant_common = sorted(list(variant_jsonl_ids & variant_txt_ids & audio_ids_set))
                    
                    variant_jsonl_count = jsonl_data.get('count', 0)
                    variant_txt_count = variant_txt_data.get('count', 0)
                    variant_common_count = len(variant_common)
                    
                    variant_stats[variant] = {
                        'jsonl_count': variant_jsonl_count,
                        'txt_count': variant_txt_count,
                        'common_count': variant_common_count,
                        'jsonl_file': jsonl_data.get('file', ''),
                        'txt_file': variant_txt_data.get('file', ''),
                        'coverage_percentage': round((variant_common_count / max(variant_jsonl_count, variant_txt_count, 1) * 100), 2) if max(variant_jsonl_count, variant_txt_count) > 0 else 0,
                        'count_match': variant_jsonl_count == variant_txt_count
                    }
                
                json_output[split][language][dialect] = {
                    'common_ids': common_ids,
                    'missing_in_jsonl': missing_in_jsonl,
                    'missing_in_txt': missing_in_txt,
                    'missing_in_wav': missing_in_wav,
                    'all_jsonl_ids': all_jsonl_ids,
                    'all_txt_ids': all_txt_ids,
                    'all_audio_ids': all_audio_ids,
                    'statistics': {
                        'counts': {
                            'common_ids': total_common,
                            'missing_in_jsonl': len(missing_in_jsonl),
                            'missing_in_txt': len(missing_in_txt),
                            'missing_in_wav': len(missing_in_wav),
                            'total_jsonl': total_jsonl,
                            'total_txt': total_txt,
                            'total_audio': total_audio,
                            'total_unique_ids': len(set(all_jsonl_ids + all_txt_ids + all_audio_ids))
                        },
                        'coverage_percentages': {
                            'jsonl_coverage': round(jsonl_coverage, 2),
                            'txt_coverage': round(txt_coverage, 2),
                            'audio_coverage': round(audio_coverage, 2),
                            'overall_coverage': round(overall_coverage, 2)
                        },
                        'variant_breakdown': variant_stats,
                        'metadata_count': r.get('metadata_count', 0),
                        'issues_count': len(r.get('issues', [])),
                        'has_issues': len(r.get('issues', [])) > 0
                    }
                }
    
    # Save JSON file
    json_file = base_dir / "data_link_ids.json"
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print(f"\nJSON file with IDs saved to: {json_file}")
    except PermissionError:
        print(f"\nWarning: Could not write JSON file to {json_file}")
    
    # Save detailed report
    report_file = base_dir / "data_link_report.txt"
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("DETAILED DATA LINK REPORT - ALL SPLITS, LANGUAGES, AND DIALECTS\n")
            f.write("="*90 + "\n\n")
            
            for split in splits:
                if split not in all_results:
                    continue
                
                f.write(f"{split.upper()} SPLIT\n")
                f.write("="*90 + "\n\n")
                
                for language in sorted(all_results[split].keys()):
                    f.write(f"Language: {language}\n")
                    f.write("-" * 90 + "\n")
                    
                    for r in all_results[split][language]:
                        f.write(f"  Dialect: {r['dialect']}\n")
                        
                        f.write("  JSONL Files:\n")
                        for variant, data in sorted(r['jsonl_files'].items()):
                            f.write(f"    {variant}: {data['file']} - {data['count']} IDs\n")
                        
                        f.write("  TXT Files:\n")
                        for variant, data in sorted(r['txt_files'].items()):
                            f.write(f"    {variant}: {data['file']} - {data['count']} IDs\n")
                        
                        f.write(f"  Metadata: {r['metadata_count']} IDs\n")
                        f.write(f"  Audio: {r['audio_count']} files\n")
                        
                        if r['issues']:
                            f.write("  Issues:\n")
                            for issue in r['issues']:
                                f.write(f"    - {issue}\n")
                        else:
                            f.write("  Issues: None\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
                
                f.write("\n")
            
            print(f"\nDetailed report saved to: {report_file}")
    except PermissionError:
        print(f"\nWarning: Could not write report file to {report_file}")
        print("Statistics are available in the console output above.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


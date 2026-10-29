"""
Shared utilities for mentee and mentor processing scripts.
Includes caching, validation, reporting, and file I/O helpers.
"""

import hashlib
import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd

# llm_client.py lives in the repo root, one level above Normalization/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import ClaudeClient

NORMALIZATION_MODEL = "claude-haiku-5-5"


def setup_claude() -> ClaudeClient:
    """Claude Haiku client for the normalization scripts; key from ANTHROPIC_API_KEY."""
    return ClaudeClient(NORMALIZATION_MODEL, os.getenv("ANTHROPIC_API_KEY", ""))


# ============================================================================
# CACHING
# ============================================================================

class AICache:
    """In-memory cache for AI API results."""
    
    def __init__(self):
        self.cache: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
    
    def get_cache_key(self, input_text: str) -> str:
        """Generate cache key from input text."""
        normalized = str(input_text).strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Any | None:
        """Get cached result."""
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        self.misses += 1
        return None
    
    def set(self, cache_key: str, result: Any):
        """Cache a result."""
        self.cache[cache_key] = result
    
    def get_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total': total,
            'hit_rate': hit_rate,
            'cached_items': len(self.cache)
        }


# ============================================================================
# FILE I/O
# ============================================================================

def find_column_flexible(df: pd.DataFrame, search_terms: list[str]) -> str | None:
    """
    Find a column in dataframe using flexible matching.
    Case-insensitive, handles trailing spaces.
    
    Args:
        df: DataFrame to search
        search_terms: List of terms to search for (any match)
    
    Returns:
        Column name if found, None otherwise
    """
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for term in search_terms:
            if term.lower() in col_lower:
                return col
    return None


def read_input_sheet(input_file: str) -> pd.DataFrame:
    """Read a responses sheet, locating the header row.

    These files sometimes carry one or more blank/metadata rows before the header
    (e.g. the sample files have two), so a fixed ``header=1`` lands on a blank row and
    yields unnamed columns. Sniff the first row that looks like the header; fall back to
    the historical ``header=1`` if none is recognized."""
    hints = ("code mentor", "code mentee", "field of expertise", "affiliation")
    raw = pd.read_excel(input_file, header=None)
    for i in range(min(8, len(raw))):
        cells = raw.iloc[i].astype(str).str.lower()
        if any(cells.str.contains(h, na=False).any() for h in hints):
            df = pd.read_excel(input_file, header=i)
            return df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return pd.read_excel(input_file, header=1)


def generate_output_filename(input_file: str, prefix: str = "POST_") -> str:
    """
    Generate output filename with prefix and timestamp.
    
    Args:
        input_file: Input file path
        prefix: Prefix to add (default: "POST_")
    
    Returns:
        Output file path
    """
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{prefix}{name}_{timestamp}{ext}"
    return os.path.join(os.path.dirname(input_file), output_name)


# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe: convert NaN to "N/A", trim whitespace, clean column names.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Clean column names (remove trailing spaces)
    df_clean.columns = [str(col).rstrip() for col in df_clean.columns]
    
    # Convert NaN to "N/A" for object columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna("N/A")
            # Trim whitespace from string values
            df_clean[col] = df_clean[col].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x) != "N/A" else x
            )
    
    return df_clean


def find_missing_data(df: pd.DataFrame, required_columns: dict[str, str]) -> list[tuple[int, str]]:
    """
    Find rows with missing data (N/A or empty) in required columns.
    
    Args:
        df: DataFrame to check
        required_columns: Dictionary mapping internal keys to column names
    
    Returns:
        List of (row_index, column_name) tuples for missing data
    """
    missing = []
    for idx, row in df.iterrows():
        for col_name in required_columns.values():
            if col_name in df.columns:
                value = row[col_name]
                if pd.isna(value) or str(value).strip() == "" or str(value).strip() == "N/A":
                    missing.append((idx + 1, col_name))  # +1 for 1-based row numbering
    return missing


# ============================================================================
# VALIDATION
# ============================================================================

def validate_required_columns(df: pd.DataFrame, required_columns: dict[str, str]) -> tuple[bool, list[str]]:
    """
    Validate that all required columns are present in dataframe.
    
    Args:
        df: DataFrame to validate
        required_columns: Dictionary mapping internal keys to column names
    
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = []
    for col_name in required_columns.values():
        # Check exact match first
        if col_name not in df.columns:
            # Check flexible match (case-insensitive, trailing spaces)
            found = False
            for col in df.columns:
                if str(col).strip().lower() == col_name.strip().lower():
                    found = True
                    break
            if not found:
                missing.append(col_name)
    
    return (len(missing) == 0, missing)


# ============================================================================
# REPORTING
# ============================================================================

class TransformationReport:
    """Track and report transformation statistics."""
    
    def __init__(self, script_name: str):
        self.script_name = script_name
        self.stats = {
            'total_rows': 0,
            'successful_rows': 0,
            'failed_rows': 0,
            'education_levels_normalized': 0,
            'education_programs_split': 0,
            'fields_classified': 0,
            'skills_merged': 0,
            'institution_types_normalized': 0,
            'missing_data_locations': []
        }
        self.errors = []
    
    def record_success(self, row_num: int):
        """Record successful row processing."""
        self.stats['successful_rows'] += 1
    
    def record_failure(self, row_num: int, operation: str, error: str):
        """Record failed row processing."""
        self.stats['failed_rows'] += 1
        self.errors.append(f"Row {row_num}: {operation} - {error}")
    
    def record_operation(self, operation: str, count: int = 1):
        """Record a transformation operation."""
        if operation in self.stats:
            self.stats[operation] += count
    
    def set_missing_data(self, missing: list[tuple[int, str]]):
        """Set missing data locations."""
        self.stats['missing_data_locations'] = missing
    
    def generate_summary(self, output_file: str, cache_stats: dict | None = None) -> str:
        """Generate summary report text."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"{self.script_name} - Transformation Summary")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total rows processed: {self.stats['total_rows']}")
        lines.append(f"Successfully processed: {self.stats['successful_rows']}")
        lines.append(f"Failed: {self.stats['failed_rows']}")
        lines.append("")
        
        if self.stats['education_levels_normalized'] > 0:
            lines.append(f"Education levels normalized: {self.stats['education_levels_normalized']}")
        if self.stats['education_programs_split'] > 0:
            lines.append(f"Education programs split: {self.stats['education_programs_split']}")
        if self.stats['fields_classified'] > 0:
            lines.append(f"Fields of expertise classified: {self.stats['fields_classified']}")
        if self.stats['skills_merged'] > 0:
            lines.append(f"Skills merged: {self.stats['skills_merged']}")
        if self.stats['institution_types_normalized'] > 0:
            lines.append(f"Institution types normalized: {self.stats['institution_types_normalized']}")
        
        if cache_stats:
            lines.append("")
            lines.append("Cache Statistics:")
            lines.append(f"  Hits: {cache_stats['hits']}")
            lines.append(f"  Misses: {cache_stats['misses']}")
            lines.append(f"  Hit Rate: {cache_stats['hit_rate']:.1f}%")
            lines.append(f"  Cached Items: {cache_stats['cached_items']}")
        
        if self.stats['missing_data_locations']:
            lines.append("")
            lines.append("Missing Data Found (converted to N/A):")
            # Group by column
            by_column = {}
            for row_num, col_name in self.stats['missing_data_locations']:
                if col_name not in by_column:
                    by_column[col_name] = []
                by_column[col_name].append(row_num)
            
            for col_name, rows in sorted(by_column.items()):
                rows_str = ", ".join(map(str, sorted(rows)))
                lines.append(f"  {col_name}: Rows {rows_str}")
        
        if self.errors:
            lines.append("")
            lines.append("Errors Encountered:")
            for error in self.errors:
                lines.append(f"  {error}")
        
        lines.append("")
        lines.append(f"Output file: {output_file}")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def write_error_log(errors: list[str], output_dir: str, script_name: str) -> str:
    """
    Write error log to file.
    
    Args:
        errors: List of error messages
        output_dir: Directory to write log
        script_name: Name of script (for filename)
    
    Returns:
        Path to error log file
    """
    if not errors:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"error_log_{script_name}_{timestamp}.txt"
    log_path = os.path.join(output_dir, log_filename)
    
    with open(log_path, 'w') as f:
        f.write(f"Error Log - {script_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.writelines(f"{error}\n" for error in errors)
    
    return log_path


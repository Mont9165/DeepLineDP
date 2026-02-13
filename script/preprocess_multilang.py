"""
Multi-language preprocessor for DeepLineDP using tree-sitter.

Replaces the Java-only preprocess_data.py with a language-agnostic version
that uses tree-sitter AST parsing for accurate tokenization across 12+ languages.

Output format is identical to the original preprocessor:
    filename, is_test_file, code_line, line_number, is_comment, is_blank,
    file-label, line-label

Usage:
    python preprocess_multilang.py --input dataset.csv --output preprocessed/
    python preprocess_multilang.py --input dataset.csv --output preprocessed/ --language python
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# Add parent project to path for language_config access
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from language_config import (
    get_language_from_filename,
)

# Tree-sitter node types that represent comments across languages
_COMMENT_NODE_TYPES = {
    "comment",
    "line_comment",
    "block_comment",
    "documentation_comment",
    "doc_comment",
}

# Tree-sitter node types that represent string/char literals
_STRING_NODE_TYPES = {
    "string",
    "string_literal",
    "interpreted_string_literal",
    "raw_string_literal",
    "template_string",
    "string_content",
    "char_literal",
    "character_literal",
    "rune_literal",
    "heredoc_body",
    "regex_literal",
}

# Tree-sitter node types that represent numbers
_NUMBER_NODE_TYPES = {
    "integer_literal",
    "float_literal",
    "decimal_literal",
    "decimal_integer_literal",
    "hex_integer_literal",
    "octal_integer_literal",
    "binary_integer_literal",
    "integer",
    "float",
    "number",
    "numeric_literal",
}


def _get_language(language_name: str):
    """Load a tree-sitter Language object."""
    try:
        from tree_sitter_language_pack import get_language
        return get_language(language_name)
    except Exception:
        return None


def _get_parser(language_name: str):
    """Create a tree-sitter Parser for the given language."""
    try:
        import tree_sitter
        lang = _get_language(language_name)
        if lang is None:
            return None
        parser = tree_sitter.Parser(lang)
        return parser
    except Exception:
        return None


def _collect_node_ranges(node, target_types: set):
    """
    Walk the AST and collect (start_byte, end_byte) ranges for nodes
    matching any of the target_types.
    """
    ranges = []
    if node.type in target_types:
        ranges.append((node.start_byte, node.end_byte))
    else:
        for child in node.children:
            ranges.extend(_collect_node_ranges(child, target_types))
    return ranges


def _replace_ranges(code_bytes: bytes, ranges, replacement: bytes) -> bytes:
    """
    Replace byte ranges in source code with a replacement token.
    Ranges must not overlap. Processes from end to start to preserve offsets.
    """
    sorted_ranges = sorted(ranges, key=lambda r: r[0], reverse=True)
    result = bytearray(code_bytes)
    for start, end in sorted_ranges:
        result[start:end] = replacement
    return bytes(result)


def _normalize_line(line: str) -> str:
    """
    Normalize a preprocessed code line: collapse whitespace, strip punctuation
    and operators to produce a token sequence.
    """
    # Remove common punctuation/operators
    line = re.sub(r"[.,;:{}()\[\]<>+\-*/=!&|^~%?@#\\]", " ", line)
    # Collapse whitespace
    line = re.sub(r"\s+", " ", line).strip()
    return line


def tokenize_with_treesitter(
    code: str, language_name: str,
) -> pd.DataFrame:
    """
    Tokenize source code using tree-sitter AST analysis.

    For each line, determines:
    - is_comment: whether the line is entirely a comment
    - is_blank: whether the line is blank after preprocessing
    - code_line: preprocessed token string (literals replaced, punctuation removed)

    Args:
        code: Source code string
        language_name: tree-sitter language name (e.g., "python", "java")

    Returns:
        DataFrame with columns: code_line, line_number, is_comment, is_blank
    """
    code_lines = code.splitlines()
    n_lines = len(code_lines)

    if n_lines == 0:
        return pd.DataFrame(
            columns=["code_line", "line_number", "is_comment", "is_blank"]
        )

    parser = _get_parser(language_name)

    # Track which lines are comments
    comment_lines = set()
    processed_code = code

    if parser is not None:
        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)
        root = tree.root_node

        # Collect comment ranges
        comment_ranges = _collect_node_ranges(root, _COMMENT_NODE_TYPES)
        for start_byte, end_byte in comment_ranges:
            # Mark all lines spanned by this comment
            start_line = code_bytes[:start_byte].count(b"\n")
            end_line = code_bytes[:end_byte].count(b"\n")
            for line_no in range(start_line, end_line + 1):
                comment_lines.add(line_no)

        # Collect all literal ranges from the original AST before any
        # modifications, then replace in a single end-to-start pass so
        # byte offsets stay valid.
        string_ranges = _collect_node_ranges(root, _STRING_NODE_TYPES)
        number_ranges = _collect_node_ranges(root, _NUMBER_NODE_TYPES)

        tagged_ranges = [(s, e, b"<str>") for s, e in string_ranges]
        tagged_ranges += [(s, e, b"<num>") for s, e in number_ranges]
        # Sort end-to-start so replacements don't shift earlier offsets
        tagged_ranges.sort(key=lambda r: r[0], reverse=True)

        result = bytearray(code_bytes)
        for start, end, replacement in tagged_ranges:
            # Preserve newlines in multi-line literals so line count stays aligned
            original = result[start:end]
            newline_count = original.count(ord(b"\n"))
            result[start:end] = replacement + b"\n" * newline_count
        code_bytes = bytes(result)

        processed_code = code_bytes.decode("utf-8", errors="replace")
    else:
        # Fallback: regex-based preprocessing when tree-sitter is unavailable
        processed_code = _regex_preprocess(code, language_name)

    processed_lines = processed_code.splitlines()
    # Ensure same number of lines (replacement might change line count if
    # multi-line strings/comments are replaced with single tokens)
    while len(processed_lines) < n_lines:
        processed_lines.append("")

    result_lines = []
    is_comments = []
    is_blanks = []

    for i in range(n_lines):
        is_comment = i in comment_lines
        is_comments.append(is_comment)

        if is_comment:
            # Keep original comment text for reference, but mark as comment
            result_lines.append(code_lines[i].strip())
        else:
            line = processed_lines[i] if i < len(processed_lines) else ""
            line = _normalize_line(line)
            result_lines.append(line)

        is_blanks.append(len(result_lines[-1].strip()) == 0)

    df = pd.DataFrame({
        "code_line": result_lines,
        "line_number": np.arange(1, n_lines + 1),
        "is_comment": is_comments,
        "is_blank": is_blanks,
    })

    return df


def _regex_preprocess(code: str, language_name: str) -> str:
    """
    Fallback regex-based preprocessing when tree-sitter is unavailable.
    Handles basic string/number replacement for common language patterns.
    """
    # Replace double-quoted strings
    code = re.sub(r'"(?:[^"\\]|\\.)*"', "<str>", code)
    # Replace single-quoted strings
    code = re.sub(r"'(?:[^'\\]|\\.)*'", "<str>", code)
    # Replace backtick template strings (JS/TS)
    if language_name in ("javascript", "typescript"):
        code = re.sub(r"`(?:[^`\\]|\\.)*`", "<str>", code)
    # Replace numbers
    code = re.sub(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b", "<num>", code)
    return code


def preprocess_file(
    code: str,
    filename: str,
    file_label: bool = False,
    buggy_lines: list = None,
    language: str = "",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single source file.

    Args:
        code: Source code string
        filename: File path/name
        file_label: True if file contains a bug (file-level label)
        buggy_lines: List of line numbers that are buggy (1-based)
        language: Override language detection (optional)

    Returns:
        DataFrame with DeepLineDP-compatible columns:
            filename, is_test_file, code_line, line_number,
            is_comment, is_blank, file-label, line-label
    """
    if not language:
        language = get_language_from_filename(filename)
    if not language:
        return pd.DataFrame()

    code_df = tokenize_with_treesitter(code, language)

    if len(code_df) == 0:
        return pd.DataFrame()

    # Check only the file path portion (after optional repo_name/ prefix)
    # to avoid false positives from repository names containing "test"
    file_part = filename.split("/", 1)[1] if "/" in filename else filename
    is_test = "test" in file_part.lower()

    code_df["filename"] = filename
    code_df["is_test_file"] = is_test
    code_df["file-label"] = file_label
    code_df["line-label"] = False

    if buggy_lines:
        code_df["line-label"] = code_df["line_number"].isin(buggy_lines)

    # Reorder columns to match DeepLineDP format
    return code_df[
        [
            "filename",
            "is_test_file",
            "code_line",
            "line_number",
            "is_comment",
            "is_blank",
            "file-label",
            "line-label",
        ]
    ]


def preprocess_dataset(
    input_csv: str,
    output_dir: str,
    release_name: str = "default",
    language_filter: str = "",
):
    """
    Preprocess a dataset CSV (file-level + optional line-level) into
    DeepLineDP format.

    Expected input CSV columns:
        File-level: File, Bug, SRC [, Language]
        Line-level (optional): File, Line_number, SRC

    Args:
        input_csv: Path to file-level CSV
        output_dir: Directory for output CSVs
        release_name: Name for this release/split
        language_filter: Only process files of this language (empty = all)
    """
    os.makedirs(output_dir, exist_ok=True)

    file_level_data = pd.read_csv(input_csv, encoding="latin")
    file_level_data = file_level_data.fillna("")

    # Check for line-level CSV (same directory, different suffix)
    line_level_path = input_csv.replace(
        "_ground-truth-files_dataset.csv", "_defective_lines_dataset.csv"
    )
    line_level_data = None
    if os.path.exists(line_level_path) and line_level_path != input_csv:
        line_level_data = pd.read_csv(line_level_path, encoding="latin")

    # Check for line-level column names (for future use)
    _ = "Line_number" in file_level_data.columns

    buggy_files = set()
    if line_level_data is not None:
        buggy_files = set(line_level_data["File"].unique())

    preprocessed_dfs = []

    for _, row in file_level_data.iterrows():
        filename = str(row["File"])
        language = get_language_from_filename(filename)

        if not language:
            continue
        if language_filter and language != language_filter:
            continue

        code = str(row.get("SRC", ""))
        if not code:
            continue

        label = bool(row.get("Bug", False))

        buggy_lines = None
        if filename in buggy_files and line_level_data is not None:
            buggy_lines = list(
                line_level_data[line_level_data["File"] == filename]["Line_number"]
            )

        code_df = preprocess_file(
            code, filename, file_label=label, buggy_lines=buggy_lines, language=language,
        )

        if len(code_df) > 0:
            preprocessed_dfs.append(code_df)

    if preprocessed_dfs:
        all_df = pd.concat(preprocessed_dfs, ignore_index=True)
        output_path = os.path.join(output_dir, f"{release_name}.csv")
        all_df.to_csv(output_path, index=False)
        print(f"Preprocessed {len(preprocessed_dfs)} files -> {output_path}")
    else:
        print(f"Warning: No files preprocessed for {release_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-language preprocessor for DeepLineDP using tree-sitter"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV path (file-level dataset)",
    )
    parser.add_argument(
        "--output", "-o",
        default="../datasets/preprocessed_data/",
        help="Output directory for preprocessed CSVs",
    )
    parser.add_argument(
        "--release", "-r",
        default="default",
        help="Release/split name for output file",
    )
    parser.add_argument(
        "--language", "-l",
        default="",
        help="Filter to process only this language (e.g., 'python', 'java')",
    )

    args = parser.parse_args()
    preprocess_dataset(args.input, args.output, args.release, args.language)


if __name__ == "__main__":
    main()

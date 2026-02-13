"""
Build DeepLineDP-format datasets from LLM4SZZ bug-inducing commit results.

Uses buggy_stmts (file_name, lineno) as line-level ground truth and
retrieves source code from BIC commits via git.

Produces:
    - agent_data/   — BICs authored by AI agents
    - human_data/   — BICs authored by humans
    - mixed_data/   — all BICs combined
    Each directory has cross-project train/eval/test split CSVs.
    A global repo-to-split assignment is computed once from all entries
    so no repository appears in different splits across groups.

Usage:
    python build_dataset_from_llm4szz.py --output ../datasets/preprocessed_data/
    python build_dataset_from_llm4szz.py --output ../datasets/preprocessed_data/ --negative-ratio 3
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root and script dir to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from classify_commit_author import get_commit_author, is_ai_agent
from extract_results import extract_all_results
from language_config import is_supported_language
from preprocess_multilang import preprocess_file

REPOS_DIR = os.path.join(_PROJECT_ROOT, "repos")


def _git_show_file(repo_path: str, commit_hash: str, file_path: str) -> Optional[str]:
    """Retrieve file content at a specific commit."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{file_path}"],
            cwd=repo_path,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8", errors="replace")
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _git_commit_date(repo_path: str, commit_hash: str) -> Optional[datetime]:
    """Get the author date of a commit."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%aI", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            date_str = result.stdout.strip()
            # Parse ISO 8601 date, handle timezone
            return datetime.fromisoformat(date_str)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return None


def _git_list_files(repo_path: str, commit_hash: str) -> List[str]:
    """List all files in the repository at a specific commit."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, OSError):
        pass
    return []


def collect_bic_entries(results: List[Dict]) -> List[Dict]:
    """
    Extract BIC entries with buggy statement info from LLM4SZZ results.

    Returns list of dicts with:
        repo_name, bfc, bic, buggy_stmts [{file_name, lineno}],
        author_name, author_type, commit_date
    """
    entries = []
    seen = set()

    for r in results:
        repo_name = r.get("repo_name", "")
        bfc = r.get("bug_fixing_commit", "")
        result_data = r.get("results", {})
        bics = result_data.get("bug_inducing_commits", [])
        buggy_stmts = result_data.get("buggy_stmts", [])

        if not bics or not buggy_stmts:
            continue

        # Group buggy_stmts by induce_cid
        stmts_by_bic: Dict[str, List[Dict]] = {}
        for stmt in buggy_stmts:
            bic = stmt.get("induce_cid", "")
            if bic:
                stmts_by_bic.setdefault(bic, []).append({
                    "file_name": stmt.get("file_name", ""),
                    "lineno": stmt.get("lineno", -1),
                })

        repo_path = os.path.join(REPOS_DIR, repo_name)

        for bic in bics:
            if not bic:
                continue

            key = (repo_name, bic)
            if key in seen:
                continue
            seen.add(key)

            stmts = stmts_by_bic.get(bic, [])
            # Filter stmts with valid file and line info
            stmts = [s for s in stmts if s["file_name"] and s["lineno"] > 0]

            if not stmts:
                continue

            # Get author info
            author_name = get_commit_author(repo_name, bic)
            if author_name:
                is_agent, agent_pattern = is_ai_agent(author_name)
                author_type = agent_pattern if is_agent else "human"
            else:
                author_type = "unknown"

            # Get commit date
            commit_date = None
            if os.path.isdir(repo_path):
                commit_date = _git_commit_date(repo_path, bic)

            entries.append({
                "repo_name": repo_name,
                "bfc": bfc,
                "bic": bic,
                "buggy_stmts": stmts,
                "author_name": author_name or "unknown",
                "author_type": author_type,
                "commit_date": commit_date,
            })

    return entries


def build_file_data(
    entry: Dict,
    negative_ratio: int = 3,
) -> List[pd.DataFrame]:
    """
    Build preprocessed DataFrames for a single BIC entry.

    For each buggy file: preprocess and set line-level labels.
    For negative examples: sample clean files from the same commit.

    Args:
        entry: BIC entry from collect_bic_entries()
        negative_ratio: Number of clean files per buggy file

    Returns:
        List of DataFrames in DeepLineDP format
    """
    repo_name = entry["repo_name"]
    bic = entry["bic"]
    repo_path = os.path.join(REPOS_DIR, repo_name)

    if not os.path.isdir(repo_path):
        return []

    # Group buggy stmts by file
    stmts_by_file: Dict[str, List[int]] = {}
    for stmt in entry["buggy_stmts"]:
        fname = stmt["file_name"]
        stmts_by_file.setdefault(fname, []).append(stmt["lineno"])

    dfs = []

    # Process buggy files
    buggy_files_processed = set()
    for file_name, line_numbers in stmts_by_file.items():
        if not is_supported_language(file_name):
            continue

        code = _git_show_file(repo_path, bic, file_name)
        if not code:
            continue

        # Prefix with repo_name to disambiguate identical paths across repos
        prefixed_name = f"{repo_name}/{file_name}"
        df = preprocess_file(
            code, prefixed_name,
            file_label=True,
            buggy_lines=line_numbers,
        )
        if len(df) > 0:
            dfs.append(df)
            buggy_files_processed.add(file_name)

    if not dfs:
        return []

    # Sample negative (clean) files from same commit
    if negative_ratio > 0:
        all_files = _git_list_files(repo_path, bic)
        clean_candidates = [
            f for f in all_files
            if f not in buggy_files_processed
            and is_supported_language(f)
        ]

        n_clean = min(
            len(buggy_files_processed) * negative_ratio,
            len(clean_candidates),
        )

        if n_clean > 0:
            rng = np.random.default_rng(hash(bic) % (2**32))
            sampled = rng.choice(clean_candidates, size=n_clean, replace=False)

            for file_name in sampled:
                code = _git_show_file(repo_path, bic, file_name)
                if not code or len(code) > 100_000:  # Skip very large files
                    continue

                prefixed_name = f"{repo_name}/{file_name}"
                df = preprocess_file(code, prefixed_name, file_label=False)
                if len(df) > 0:
                    dfs.append(df)

    return dfs


def assign_repo_splits(
    entries: List[Dict],
    train_ratio: float = 0.6,
    eval_ratio: float = 0.2,
) -> Dict[str, str]:
    """
    Compute a global repository-to-split assignment from ALL entries.

    Returns a dict mapping repo_name → "train" | "eval" | "test".

    Repositories are sorted by their median commit date so that the
    training set contains generally older projects and the test set
    contains newer ones.  The assignment is computed once from the full
    (mixed) entry list and is then applied identically to every author
    group, guaranteeing that no repository appears in different splits
    across cross-domain scenarios (e.g. Human→Agent).

    Requires at least 3 distinct repositories so that each split can
    contain at least one repo.
    """
    if not (0 < train_ratio < 1 and 0 < eval_ratio < 1 and train_ratio + eval_ratio < 1):
        raise ValueError(f"Invalid ratios: train={train_ratio}, eval={eval_ratio}")

    # Group entries by repository
    repo_entries: Dict[str, List[Dict]] = {}
    for e in entries:
        repo_entries.setdefault(e["repo_name"], []).append(e)

    if len(repo_entries) < 3:
        raise ValueError(
            f"Cross-project split requires at least 3 repositories, "
            f"but only {len(repo_entries)} found: {list(repo_entries.keys())}"
        )

    # Sort repos by median commit date, breaking ties by name for determinism.
    # Repos without dates go first → train.
    def _median_date(elist):
        dates = [e["commit_date"] for e in elist if e["commit_date"] is not None]
        if not dates:
            return datetime.min
        dates.sort()
        return dates[len(dates) // 2]

    sorted_repos = sorted(repo_entries.keys(), key=lambda r: (_median_date(repo_entries[r]), r))

    # Assign repos to splits so cumulative entry count approximates the ratios
    total = len(entries)
    train_target = int(total * train_ratio)
    eval_target = int(total * (train_ratio + eval_ratio))

    repo_split: Dict[str, str] = {}
    running = 0
    for repo in sorted_repos:
        if running < train_target:
            repo_split[repo] = "train"
        elif running < eval_target:
            repo_split[repo] = "eval"
        else:
            repo_split[repo] = "test"
        running += len(repo_entries[repo])

    # Ensure eval and test are non-empty by reassigning the last train repo
    splits_used = set(repo_split.values())
    if "eval" not in splits_used:
        train_repos = [r for r in sorted_repos if repo_split[r] == "train"]
        if len(train_repos) > 1:
            repo_split[train_repos[-1]] = "eval"
    splits_used = set(repo_split.values())
    if "test" not in splits_used:
        eval_repos = [r for r in sorted_repos if repo_split[r] == "eval"]
        if len(eval_repos) > 1:
            repo_split[eval_repos[-1]] = "test"

    # Summary
    split_counts = {"train": 0, "eval": 0, "test": 0}
    for s in repo_split.values():
        split_counts[s] += 1
    print(f"  Global repo assignment: {split_counts['train']} train repos, "
          f"{split_counts['eval']} eval repos, {split_counts['test']} test repos")

    return repo_split


def apply_repo_splits(
    entries: List[Dict],
    repo_splits: Dict[str, str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Partition entries into (train, eval, test) using a pre-computed repo assignment.

    Entries whose repo_name is absent from *repo_splits* are silently dropped.
    """
    train, eval_, test = [], [], []
    for e in entries:
        split = repo_splits.get(e["repo_name"])
        if split == "train":
            train.append(e)
        elif split == "eval":
            eval_.append(e)
        elif split == "test":
            test.append(e)
    return train, eval_, test


def build_split_csv(
    entries: List[Dict],
    split_name: str,
    output_dir: str,
    negative_ratio: int = 3,
) -> Optional[str]:
    """
    Build a preprocessed CSV for a dataset split.

    Args:
        entries: BIC entries for this split
        split_name: Name for the output file (e.g., "train", "eval", "test")
        output_dir: Output directory
        negative_ratio: Clean-to-buggy file ratio

    Returns:
        Path to output CSV, or None if no data
    """
    all_dfs = []
    for i, entry in enumerate(entries):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(entries)} entries...")

        file_dfs = build_file_data(entry, negative_ratio=negative_ratio)
        all_dfs.extend(file_dfs)

    if not all_dfs:
        print(f"  Warning: No data for split '{split_name}'")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    output_path = os.path.join(output_dir, f"{split_name}.csv")
    combined.to_csv(output_path, index=False)

    n_files = combined["filename"].nunique()
    n_buggy_files = combined[combined["file-label"]]["filename"].nunique()
    n_buggy_lines = combined["line-label"].sum()
    print(
        f"  {split_name}: {len(combined)} lines, "
        f"{n_files} files ({n_buggy_files} buggy), "
        f"{n_buggy_lines} buggy lines -> {output_path}"
    )
    return output_path


def build_datasets(
    output_base: str,
    negative_ratio: int = 3,
    min_entries: int = 10,
    input_file: str = "",
):
    """
    Build all DeepLineDP datasets from LLM4SZZ results.

    Creates:
        {output_base}/agent_data/{train,eval,test}.csv
        {output_base}/human_data/{train,eval,test}.csv
        {output_base}/mixed_data/{train,eval,test}.csv
        {output_base}/dataset_info.json

    Args:
        input_file: Path to pre-extracted results JSON. If empty, scans save_logs.
    """
    if input_file and os.path.exists(input_file):
        print(f"Loading pre-extracted results from {input_file}...")
        with open(input_file) as f:
            data = json.load(f)
        results = data.get("results", data) if isinstance(data, dict) else data
        if isinstance(results, dict):
            results = results.get("results", [])
    else:
        print("Extracting LLM4SZZ results from save_logs...")
        results = extract_all_results("llm4szz")
    print(f"  Found {len(results)} commit results")

    print("Collecting BIC entries with buggy statements...")
    entries = collect_bic_entries(results)
    print(f"  Found {len(entries)} BIC entries with valid buggy stmts")

    if not entries:
        print("Error: No valid BIC entries found")
        return

    # Classify entries by author type
    agent_entries = [e for e in entries if e["author_type"] != "human" and e["author_type"] != "unknown"]
    human_entries = [e for e in entries if e["author_type"] == "human"]
    unknown_entries = [e for e in entries if e["author_type"] == "unknown"]

    print(f"  Agent: {len(agent_entries)}, Human: {len(human_entries)}, Unknown: {len(unknown_entries)}")

    # Print agent type breakdown
    agent_types: Dict[str, int] = {}
    for e in agent_entries:
        agent_types[e["author_type"]] = agent_types.get(e["author_type"], 0) + 1
    if agent_types:
        print(f"  Agent breakdown: {agent_types}")

    dataset_info: Dict[str, Any] = {
        "total_entries": len(entries),
        "agent_entries": len(agent_entries),
        "human_entries": len(human_entries),
        "unknown_entries": len(unknown_entries),
        "agent_breakdown": agent_types,
        "negative_ratio": negative_ratio,
        "datasets": {},
    }

    # Compute a single global repo-to-split mapping from ALL entries.
    # This ensures no repository appears in different splits across
    # cross-domain scenarios (e.g. Human→Agent), preventing data leaking.
    print("\nComputing global cross-project repo assignment...")
    repo_splits = assign_repo_splits(entries)

    # Build datasets for each author group
    groups = {
        "agent_data": agent_entries,
        "human_data": human_entries,
        "mixed_data": entries,  # all entries including unknown
    }

    for group_name, group_entries in groups.items():
        print(f"\n{'='*60}")
        print(f"Building {group_name} ({len(group_entries)} entries)...")

        if len(group_entries) < min_entries:
            print(f"  Skipping: fewer than {min_entries} entries")
            continue

        group_dir = os.path.join(output_base, group_name)
        os.makedirs(group_dir, exist_ok=True)

        train, eval_, test = apply_repo_splits(group_entries, repo_splits)
        print(f"  Cross-project split: train={len(train)}, eval={len(eval_)}, test={len(test)}")

        if not train or not eval_ or not test:
            empty = [s for s, v in [("train", train), ("eval", eval_), ("test", test)] if not v]
            print(f"  WARNING: {group_name} has empty split(s): {empty}. "
                  "Cross-domain evaluation for this group may not be possible.")

        split_info = {}
        for split_name, split_entries in [("train", train), ("eval", eval_), ("test", test)]:
            if not split_entries:
                print(f"  Warning: Empty {split_name} split")
                continue

            path = build_split_csv(
                split_entries, split_name, group_dir,
                negative_ratio=negative_ratio,
            )
            if path:
                split_info[split_name] = {
                    "path": os.path.relpath(path, output_base),
                    "n_entries": len(split_entries),
                }

        dataset_info["datasets"][group_name] = split_info

    # Save dataset info
    info_path = os.path.join(output_base, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2, default=str)
    print(f"\nDataset info saved to {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build DeepLineDP datasets from LLM4SZZ results"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(_SCRIPT_DIR, "..", "datasets", "preprocessed_data"),
        help="Output base directory",
    )
    parser.add_argument(
        "--input", "-i",
        default="",
        help="Pre-extracted results JSON (e.g., extracted_results.json). "
             "If empty, scans save_logs directly.",
    )
    parser.add_argument(
        "--negative-ratio", "-n",
        type=int,
        default=3,
        help="Number of clean files per buggy file (default: 3)",
    )
    parser.add_argument(
        "--min-entries",
        type=int,
        default=10,
        help="Minimum BIC entries to build a dataset (default: 10)",
    )

    args = parser.parse_args()
    build_datasets(args.output, args.negative_ratio, args.min_entries, args.input)


if __name__ == "__main__":
    main()

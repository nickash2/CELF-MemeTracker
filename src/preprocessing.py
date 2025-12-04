from __future__ import annotations

import gzip
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

from .celf import InfluenceGraph


@dataclass
class MemeTrackerDocument:
    url: str
    timestamp: datetime
    phrases: List[str]
    links: List[str]

    def to_node_id(self) -> str:
        """Extract domain/site identifier from URL for graph nodes."""
        url_clean = self.url.replace("http://", "").replace("https://", "")
        domain = url_clean.split("/")[0]
        return domain


def load_graph_from_file(
    path: str,
    default_prob: float = 0.1,
    delimiter: Optional[str] = None,
    skip_header: bool = False,
) -> InfluenceGraph:
    """Build an InfluenceGraph from an edge list file.

    File format: source target [probability]
    """
    graph = InfluenceGraph(default_prob=default_prob)
    with open(path, "r", encoding="utf-8") as handle:
        if skip_header:
            handle.readline()
        for line_no, raw in enumerate(handle, start=1 if not skip_header else 2):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter) if delimiter is not None else line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Line {line_no}: expected at least 2 columns, found {parts}."
                )
            src, dst = parts[0], parts[1]
            prob: Optional[float] = None
            if len(parts) >= 3:
                try:
                    prob = float(parts[2])
                except ValueError as exc:
                    raise ValueError(
                        f"Line {line_no}: invalid probability '{parts[2]}'."
                    ) from exc
            graph.add_edge(src, dst, prob)
    return graph


def load_costs_from_file(path: str) -> Dict[str, float]:
    """Load node costs from a two-column text file: node cost"""
    costs: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_no}: expected 'node cost' but found {parts}."
                )
            node, raw_cost = parts
            try:
                costs[node] = float(raw_cost)
            except ValueError as exc:
                raise ValueError(f"Line {line_no}: invalid cost '{raw_cost}'.") from exc
    return costs


def build_graph_from_cascades(
    cascades: List[List[Tuple[str, float]]],
    min_prob: float = 0.01,
) -> InfluenceGraph:
    """Build influence graph from temporal cascades using frequency-based static probs.

    Each cascade: list of (node, timestamp_hours)
    """
    graph = InfluenceGraph(default_prob=min_prob)
    edge_counts: Dict[Tuple[str, str], int] = {}
    node_out_counts: Dict[str, int] = {}

    for cascade in cascades:
        sorted_cascade = sorted(cascade, key=lambda x: x[1])
        for i, (source, t1) in enumerate(sorted_cascade):
            node_out_counts[source] = (
                node_out_counts.get(source, 0) + len(sorted_cascade) - i - 1
            )
            for target, t2 in sorted_cascade[i + 1 :]:
                time_diff = t2 - t1
                if time_diff <= 0:
                    continue
                edge = (source, target)
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

    for (src, dst), count in edge_counts.items():
        if node_out_counts.get(src, 0) > 0:
            prob = count / node_out_counts[src]
        else:
            prob = min_prob
        if prob >= min_prob:
            graph.add_edge(src, dst, prob)

    return graph


# -----------------------------------------------------------------------------
# Streaming MemeTracker parser utilities
# -----------------------------------------------------------------------------


def _open_memetracker_file(path: str):
    return (
        gzip.open(path, "rt", encoding="utf-8", errors="ignore")
        if path.endswith(".gz")
        else open(path, "rt", encoding="utf-8", errors="ignore")
    )


def iter_memetracker_documents(
    path: str, max_documents: Optional[int] = None
) -> Generator[MemeTrackerDocument, None, None]:
    """Stream MemeTracker documents as MemeTrackerDocument objects.

    Yields documents one at a time. Stops early if max_documents is a positive int.
    If max_documents is None, streams the whole file.
    """
    # Normalize unlimited sentinel: if negative, treat as None
    if max_documents is not None and max_documents < 0:
        max_documents = None

    try:
        from tqdm import tqdm

        use_tqdm = True
    except Exception:

        def tqdm(x, **kwargs):
            return x

        use_tqdm = False

    with _open_memetracker_file(path) as f:
        iterator = tqdm(f, desc="  Parsing documents", disable=not use_tqdm)

        current_url: Optional[str] = None
        current_time: Optional[datetime] = None
        current_phrases: List[str] = []
        current_links: List[str] = []
        documents_yielded = 0

        for raw_line in iterator:
            line = raw_line.strip()

            # Document boundary
            if not line:
                if current_url and current_time:
                    yield MemeTrackerDocument(
                        url=current_url,
                        timestamp=current_time,
                        phrases=current_phrases,
                        links=current_links,
                    )
                    documents_yielded += 1
                    if max_documents is not None and documents_yielded >= max_documents:
                        return
                # reset
                current_url = None
                current_time = None
                current_phrases = []
                current_links = []
                continue

            # Lines expected to be like: 'P\thttp...'
            if len(line) < 2 or line[1] != "\t":
                # malformed line -> skip
                continue

            line_type = line[0]
            content = line[2:].strip()

            if line_type == "P":
                current_url = content
            elif line_type == "T":
                try:
                    current_time = datetime.strptime(content, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    # Silently skip invalid timestamps; doc will be dropped if no valid time
                    current_time = None
            elif line_type == "Q":
                current_phrases.append(content)
            elif line_type == "L":
                current_links.append(content)

        # final document if file doesn't end with blank line
        if current_url and current_time:
            yield MemeTrackerDocument(
                url=current_url,
                timestamp=current_time,
                phrases=current_phrases,
                links=current_links,
            )


# -----------------------------------------------------------------------------
# High-level MemeTracker graph builder (streaming & memory-efficient)
# -----------------------------------------------------------------------------


def build_memetracker_cascade(
    documents: List[MemeTrackerDocument],
    meme_phrase: str,
    case_sensitive: bool = False,
) -> List[Tuple[str, float]]:
    """Build a single cascade from a list of MemeTrackerDocument objects (in-memory list).

    This helper is retained for compatibility with places that already have a list of docs.
    """
    cascade_entries: List[Tuple[str, datetime, str]] = []
    search_phrase = meme_phrase if case_sensitive else meme_phrase.lower()

    for doc in documents:
        doc_phrases = (
            doc.phrases if case_sensitive else [p.lower() for p in doc.phrases]
        )
        if any(search_phrase in phrase for phrase in doc_phrases):
            site_id = doc.to_node_id()
            cascade_entries.append((site_id, doc.timestamp, doc.url))

    if not cascade_entries:
        return []

    cascade_entries.sort(key=lambda x: x[1])
    start_time = cascade_entries[0][1]
    cascade: List[Tuple[str, float]] = []
    seen_sites = set()
    for site_id, timestamp, _ in cascade_entries:
        if site_id in seen_sites:
            continue
        seen_sites.add(site_id)
        time_diff = (timestamp - start_time).total_seconds() / 3600.0
        cascade.append((site_id, time_diff))
    return cascade


def build_graph_from_memetracker(
    path: str,
    top_memes: int = 100,
    min_prob: float = 0.01,
    max_documents: Optional[int] = None,
) -> Tuple[InfluenceGraph, Dict[str, List[Tuple[str, float]]]]:
    """Build influence graph from MemeTracker file with memory-efficient streaming.

    Behavior changes:
    - If `max_documents` or `top_memes` is negative, treat as unlimited (None).
    - Two-pass streaming: first pass counts meme frequencies and determines top memes;
      second pass extracts occurrences only for those top memes.
    - Avoids storing all documents in memory when not necessary.
    """

    # Normalize "unlimited" sentinel: negative -> None
    if max_documents is not None and max_documents < 0:
        max_documents = None
    if top_memes is not None and top_memes < 0:
        top_memes = None

    # Compose cache file names based on input and parameters
    cache_prefix = f"{os.path.splitext(os.path.basename(path))[0]}_tm{top_memes if top_memes is not None else 'all'}_md{max_documents if max_documents is not None else 'all'}_mp{min_prob}"
    cache_dir = os.path.join(os.path.dirname(path), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cascades_cache = os.path.join(cache_dir, cache_prefix + "_cascades.pkl")
    graph_cache = os.path.join(cache_dir, cache_prefix + "_graph.pkl")

    # Try to load from cache
    if os.path.exists(cascades_cache) and os.path.exists(graph_cache):
        print(f"Loading cascades and graph from cache: {cache_dir}")
        with open(cascades_cache, "rb") as f:
            cascades_dict = pickle.load(f)
        with open(graph_cache, "rb") as f:
            graph = pickle.load(f)
        print(f"  Loaded {len(cascades_dict)} cascades, {len(graph.nodes)} nodes")
        return graph, cascades_dict

    print(f"Parsing MemeTracker file: {path}")

    # ---------------------------
    # Pass 1: count meme frequencies
    # ---------------------------
    try:
        from tqdm import tqdm

        use_tqdm = True
    except Exception:

        def tqdm(x, **kwargs):
            return x

        use_tqdm = False

    meme_counts: Dict[str, int] = defaultdict(int)
    parsed_docs = 0

    for doc in iter_memetracker_documents(path, max_documents=max_documents):
        parsed_docs += 1
        for phrase in doc.phrases:
            phrase_clean = phrase.lower().strip()
            # simple filter to avoid extremely short phrases
            if len(phrase_clean) > 5:
                meme_counts[phrase_clean] += 1

    print(f"  Parsed {parsed_docs} documents")

    # Determine top memes list based on top_memes limit
    sorted_memes = sorted(meme_counts.items(), key=lambda x: x[1], reverse=True)
    if top_memes is None:
        meme_list = [m for m, _ in sorted_memes]
    else:
        meme_list = [m for m, _ in sorted_memes[:top_memes]]

    if sorted_memes:
        print(f"  Top meme: '{sorted_memes[0][0]}' ({sorted_memes[0][1]} mentions)")
    else:
        print("  No memes found.")

    # If no memes to process, return early with informative message
    if not meme_list:
        print("[ERROR] No candidate memes after filtering. Exiting.")
        return InfluenceGraph(default_prob=min_prob), {}

    # ---------------------------
    # Pass 2: extract occurrences only for top memes (streaming)
    # ---------------------------
    print("  Indexing meme occurrences (streaming extraction)...")
    meme_set = set(meme_list)
    meme_occurrences: Dict[str, List[Tuple[str, datetime, str]]] = defaultdict(list)

    docs_seen = 0
    for doc in iter_memetracker_documents(path, max_documents=max_documents):
        docs_seen += 1
        site_id = doc.to_node_id()
        # check each phrase and if it's one of the top memes, record occurrence
        for phrase in doc.phrases:
            phrase_clean = phrase.lower().strip()
            if phrase_clean in meme_set:
                meme_occurrences[phrase_clean].append((site_id, doc.timestamp, doc.url))

    # Build cascades for each meme and assemble all cascades
    cascades_dict: Dict[str, List[Tuple[str, float]]] = {}
    all_cascades: List[List[Tuple[str, float]]] = []

    print("  Building cascades from indexed occurrences...")
    for meme in meme_list:
        entries = meme_occurrences.get(meme, [])
        if not entries:
            continue
        entries.sort(key=lambda x: x[1])
        start_time = entries[0][1]
        cascade: List[Tuple[str, float]] = []
        seen_sites = set()
        for site_id, timestamp, _ in entries:
            if site_id in seen_sites:
                continue
            seen_sites.add(site_id)
            time_diff = (timestamp - start_time).total_seconds() / 3600.0
            cascade.append((site_id, time_diff))
        if len(cascade) >= 2:
            cascades_dict[meme] = cascade
            all_cascades.append(cascade)

    print(
        f"  Extracted {len(all_cascades)} valid cascades out of {len(meme_list)} memes."
    )
    if len(all_cascades) == 0:
        print(
            "[ERROR] No valid cascades found after extraction. Check filters or input data."
        )
        # Helpful debug snippets
        print("[DEBUG] Example meme counts (top 5):", sorted_memes[:5])
        return InfluenceGraph(default_prob=min_prob), {}

    # Build graph and cache results
    print("  Constructing influence graph...")
    graph = build_graph_from_cascades(all_cascades, min_prob)
    print(f"  Graph: {len(graph.nodes)} nodes")

    # Save to cache (best-effort)
    try:
        with open(cascades_cache, "wb") as f:
            pickle.dump(cascades_dict, f)
        with open(graph_cache, "wb") as f:
            pickle.dump(graph, f)
        print(f"  Saved cascades and graph to cache: {cache_dir}")
    except Exception as exc:
        print(f"  Warning: failed to save cache: {exc}")

    return graph, cascades_dict

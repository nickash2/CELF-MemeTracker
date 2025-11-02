"""
Data preprocessing utilities for CELF algorithm.

Provides loaders for graphs, node costs, and MemeTracker-specific cascade data.
MemeTracker format: http://snap.stanford.edu/data/memetracker9.html
"""

from __future__ import annotations

import gzip
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .celf import InfluenceGraph


@dataclass
class MemeTrackerDocument:
    """Represents a single document from MemeTracker dataset."""

    url: str
    timestamp: datetime
    phrases: List[str]
    links: List[str]

    def to_node_id(self) -> str:
        """Extract domain/site identifier from URL for graph nodes."""
        # Extract domain: http://blog.example.com/path -> blog.example.com
        url_clean = self.url.replace("http://", "").replace("https://", "")
        domain = url_clean.split("/")[0]
        return domain


def load_graph_from_file(
    path: str,
    default_prob: float = 0.1,
    delimiter: Optional[str] = None,
    skip_header: bool = False,
) -> InfluenceGraph:
    """Builds an InfluenceGraph from an edge list on disk.

    Args:
        path: Path to edge list file
        default_prob: Default edge probability when not specified
        delimiter: Field delimiter (None = whitespace)
        skip_header: Whether to skip first line

    Returns:
        Constructed InfluenceGraph

    File format:
        source target [probability]
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
    """Loads node costs from a simple two-column text file.

    Args:
        path: Path to costs file

    Returns:
        Dictionary mapping node -> cost

    File format:
        node cost
    """

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


def estimate_propagation_probability(
    time_diff_hours: float,
    decay_rate: float = 0.01,
) -> float:
    """Estimates influence probability from temporal cascade data.

    Uses exponential decay model: p(t) = exp(-α * t)

    Args:
        time_diff_hours: Time difference between source and target mention
        decay_rate: Decay parameter α (default: 0.01)

    Returns:
        Propagation probability in [0, 1]
    """

    if time_diff_hours < 0:
        return 0.0
    prob = math.exp(-decay_rate * time_diff_hours)
    return min(1.0, max(0.0, prob))


def build_graph_from_cascades(
    cascades: list[list[tuple[str, float]]],
    decay_rate: float = 0.01,
    min_prob: float = 0.01,
) -> InfluenceGraph:
    """Constructs an influence graph from temporal cascades.

    Each cascade is a list of (node_id, timestamp) pairs showing when
    nodes adopted a meme/information. Creates directed edges from earlier
    to later adopters with probabilities based on time differences.

    Args:
        cascades: List of cascades, each cascade is [(node, timestamp), ...]
        decay_rate: Exponential decay parameter for probability estimation
        min_prob: Minimum probability threshold (edges below are discarded)

    Returns:
        InfluenceGraph with estimated edge probabilities

    Example:
        cascades = [
            [("blog_A", 0.0), ("blog_B", 1.5), ("blog_C", 3.0)],
            [("blog_B", 0.0), ("blog_C", 0.5)],
        ]
        graph = build_graph_from_cascades(cascades)
    """

    graph = InfluenceGraph(default_prob=min_prob)
    edge_counts: Dict[tuple[str, str], list[float]] = {}

    for cascade in cascades:
        sorted_cascade = sorted(cascade, key=lambda x: x[1])

        for i, (source, t1) in enumerate(sorted_cascade):
            for target, t2 in sorted_cascade[i + 1 :]:
                time_diff = t2 - t1
                if time_diff <= 0:
                    continue

                prob = estimate_propagation_probability(time_diff, decay_rate)
                if prob < min_prob:
                    continue

                edge = (source, target)
                edge_counts.setdefault(edge, []).append(prob)

    for (src, dst), probs in edge_counts.items():
        avg_prob = sum(probs) / len(probs)
        graph.add_edge(src, dst, avg_prob)

    return graph


def parse_memetracker_file(
    path: str,
    max_documents: Optional[int] = None,
) -> List[MemeTrackerDocument]:
    """Parse MemeTracker format file (plain text or .gz compressed).

    File format from http://snap.stanford.edu/data/memetracker9.html:
        P       http://blogs.example.com/post.html
        T       2008-09-09 22:35:24
        Q       phrase extracted from text
        Q       another phrase
        L       http://linked-site.com/article.html
        L       http://another-link.com
        <blank line separates documents>

    Args:
        path: Path to MemeTracker data file (.txt or .txt.gz)
        max_documents: Optional limit on number of documents to parse

    Returns:
        List of parsed MemeTrackerDocument objects
    """
    documents: List[MemeTrackerDocument] = []
    current_url: Optional[str] = None
    current_time: Optional[datetime] = None
    current_phrases: List[str] = []
    current_links: List[str] = []

    open_fn = gzip.open if path.endswith(".gz") else open

    with open_fn(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Empty line signals end of document
            if not line:
                if current_url and current_time:
                    documents.append(
                        MemeTrackerDocument(
                            url=current_url,
                            timestamp=current_time,
                            phrases=current_phrases,
                            links=current_links,
                        )
                    )
                    if max_documents and len(documents) >= max_documents:
                        break

                # Reset for next document
                current_url = None
                current_time = None
                current_phrases = []
                current_links = []
                continue

            # Parse line type
            if len(line) < 2 or line[1] != "\t":
                continue  # Skip malformed lines

            line_type = line[0]
            content = line[2:].strip()

            if line_type == "P":
                current_url = content
            elif line_type == "T":
                try:
                    current_time = datetime.strptime(content, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass  # Skip documents with invalid timestamps
            elif line_type == "Q":
                current_phrases.append(content)
            elif line_type == "L":
                current_links.append(content)

        # Handle last document if file doesn't end with blank line
        if current_url and current_time:
            documents.append(
                MemeTrackerDocument(
                    url=current_url,
                    timestamp=current_time,
                    phrases=current_phrases,
                    links=current_links,
                )
            )

    return documents


def build_memetracker_cascade(
    documents: List[MemeTrackerDocument],
    meme_phrase: str,
    case_sensitive: bool = False,
) -> List[Tuple[str, float]]:
    """Extract a single meme cascade from documents.

    Args:
        documents: List of parsed MemeTracker documents
        meme_phrase: The phrase/meme to track
        case_sensitive: Whether phrase matching is case-sensitive

    Returns:
        Cascade as list of (site_id, timestamp_hours) tuples, sorted by time

    Example:
        docs = parse_memetracker_file("data/quotes_2008-08.txt.gz")
        cascade = build_memetracker_cascade(docs, "lipstick on a pig")
    """
    cascade_entries: List[Tuple[str, datetime, str]] = []
    search_phrase = meme_phrase if case_sensitive else meme_phrase.lower()

    for doc in documents:
        # Check if document contains the meme
        doc_phrases = (
            doc.phrases if case_sensitive else [p.lower() for p in doc.phrases]
        )

        if any(search_phrase in phrase for phrase in doc_phrases):
            site_id = doc.to_node_id()
            cascade_entries.append((site_id, doc.timestamp, doc.url))

    if not cascade_entries:
        return []

    # Sort by timestamp
    cascade_entries.sort(key=lambda x: x[1])

    # Convert to relative hours from first mention
    start_time = cascade_entries[0][1]
    cascade = []
    seen_sites = set()

    for site_id, timestamp, _ in cascade_entries:
        # Only include first mention per site
        if site_id in seen_sites:
            continue
        seen_sites.add(site_id)

        time_diff = (timestamp - start_time).total_seconds() / 3600.0  # hours
        cascade.append((site_id, time_diff))

    return cascade


def build_graph_from_memetracker(
    path: str,
    top_memes: int = 100,
    decay_rate: float = 0.01,
    min_prob: float = 0.01,
    max_documents: Optional[int] = None,
) -> Tuple[InfluenceGraph, Dict[str, List[Tuple[str, float]]]]:
    """Build influence graph from MemeTracker data file.

    Extracts cascades for top memes and constructs a graph where edges
    represent influence relationships between sites (blogs/news).

    Args:
        path: Path to MemeTracker file (.txt or .txt.gz)
        top_memes: Number of top memes to track
        decay_rate: Exponential decay for probability estimation
        min_prob: Minimum edge probability threshold
        max_documents: Optional limit on documents to parse

    Returns:
        Tuple of (InfluenceGraph, dict of meme -> cascade)

    Example:
        graph, cascades = build_graph_from_memetracker(
            "data/quotes_2008-08.txt.gz",
            top_memes=50,
            max_documents=100000
        )
    """
    print(f"Parsing MemeTracker file: {path}")
    documents = parse_memetracker_file(path, max_documents=max_documents)
    print(f"  Parsed {len(documents)} documents")

    # Count meme frequencies
    meme_counts: Dict[str, int] = defaultdict(int)
    for doc in documents:
        for phrase in doc.phrases:
            phrase_clean = phrase.lower().strip()
            if len(phrase_clean) > 5:  # Filter very short phrases
                meme_counts[phrase_clean] += 1

    # Get top memes
    top_meme_phrases = sorted(meme_counts.items(), key=lambda x: x[1], reverse=True)[
        :top_memes
    ]
    print(f"  Top meme: '{top_meme_phrases[0][0]}' ({top_meme_phrases[0][1]} mentions)")

    # Extract cascades
    cascades_dict: Dict[str, List[Tuple[str, float]]] = {}
    all_cascades: List[List[Tuple[str, float]]] = []

    print("  Building cascades...")
    for meme, count in top_meme_phrases:
        cascade = build_memetracker_cascade(documents, meme, case_sensitive=False)
        if len(cascade) >= 2:  # Need at least 2 sites for an edge
            cascades_dict[meme] = cascade
            all_cascades.append(cascade)

    print(f"  Extracted {len(all_cascades)} valid cascades")

    # Build graph from cascades
    print("  Constructing influence graph...")
    graph = build_graph_from_cascades(all_cascades, decay_rate, min_prob)
    print(f"  Graph: {len(graph.nodes)} nodes")

    return graph, cascades_dict

"""
Module: whisperx_corrector_core.py

Purpose:
    Provides core functions for advanced correction of WhisperX JSON outputs,
    including a simplified adapter function for pipeline integration. Retains
    sophisticated correction logic while removing standalone application features.

Key Features (Preserved):
    - Explicit config passing (using frozen dataclass).
    - SeverityLevel Enum for anomaly classification.
    - Robust edge-case handling (e.g., zero duration segments).
    - Memory-efficient file discovery using generators.
    - Consistent use of Path objects internally.
    - Tiered anomaly detection, statistical analysis, splitting/merging logic.
    - Safe file writing.
    - Configurable logging level.
TODO:
    - make language option
"""

import os
import json
import numpy as np
import statistics
from dataclasses import dataclass, field
from enum import Enum, auto
import re
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, TypedDict, Union, Generator, Set
from tqdm import tqdm
import sys
from numbers import Number  # For type checking
from types import (
    MappingProxyType,
)  # For immutable default config view (if needed, though frozen dataclass handles it)


# --- Type Definitions & Enums ---
# (TypedDicts remain suitable here for static analysis benefits)
class WordToken(TypedDict, total=False):
    """Represents a single word with timing information."""

    word: str
    start: float
    end: float
    score: Optional[float]


class Segment(TypedDict):
    """Represents a segment of speech with timing and words."""

    start: float
    end: float
    text: str
    words: List[WordToken]


class SegmentStats(TypedDict, total=False):
    """Statistical metrics calculated for a segment."""

    valid: bool
    reason: Optional[str]
    total_words: int
    segment_duration: float
    word_count_with_timing: int
    avg_word_duration: float
    max_word_duration: float
    min_word_duration: float
    word_duration_variance: float
    avg_gap: float
    max_gap: float
    total_gap_time: float
    speech_rate: float
    time_utilization: float
    density: float
    gap_stddev: Optional[float]
    duration_stddev: Optional[float]
    time_span: float


class ProcessedStats(TypedDict):
    """Statistics summarizing the processing of a single file."""

    anomalies_detected: int
    severe_anomalies: int
    moderate_anomalies: int
    splits_performed: int
    merges_performed: int
    original_segments: int
    final_segments: int
    error: Optional[str]


class SeverityLevel(Enum):
    """Enumeration for anomaly severity levels."""

    NORMAL = auto()
    MODERATE = auto()
    SEVERE = auto()


# --- Configuration ---
@dataclass(frozen=True)
class WhisperXCorrectorConfig:
    """Configuration settings for the WhisperX Corrector logic."""

    max_word_duration: float = 2.0
    min_word_duration: float = 0.05
    typical_word_duration: float = 0.3
    max_gap_duration: float = 2.0
    hard_gap_duration: float = 5.0
    min_gap_duration: float = 0.02
    max_segment_duration: float = 15.0
    max_merged_word_count: int = 35
    suspicious_density_threshold: float = 0.3
    suspicious_max_word_duration: float = 5.0
    suspicious_max_gap: float = 7.0
    suspicious_gap_stddev: float = 2.0
    typical_speech_rate: float = 3.0
    max_speech_rate_deviation: float = 3.0
    min_time_utilization: float = 0.3
    safe_write: bool = True
    debug: bool = False
    merge_time_gap_threshold: float = 0.5
    merge_min_word_duration_for_split: float = 1.0
    merge_min_word_count_for_split: int = 3
    sentence_boundary_split_score: float = 5.0
    hard_gap_split_score: float = 20.0
    split_score_threshold: float = 1.0

    def __post_init__(self):
        # Basic validation of config values
        if self.min_word_duration >= self.max_word_duration:
            raise ValueError("min_word_duration >= max_word_duration")
        if self.min_word_duration <= 0:
            raise ValueError("min_word_duration must be positive")
        if self.min_gap_duration >= self.max_gap_duration:
            raise ValueError("min_gap_duration >= max_gap_duration")
        if self.max_gap_duration >= self.hard_gap_duration:
            raise ValueError("max_gap_duration >= hard_gap_duration")
        if self.max_merged_word_count <= 0:
            raise ValueError("max_merged_word_count must be positive")
        if not (0 < self.suspicious_density_threshold < 1):
            raise ValueError("suspicious_density_threshold must be between 0 and 1")
        # Add more comprehensive checks if needed


# Default immutable config instance
DEFAULT_CONFIG: WhisperXCorrectorConfig = WhisperXCorrectorConfig()

# --- Constants ---
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
VALID_LOG_LEVELS: Set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
PUNCTUATION_NO_SPACE_BEFORE: frozenset[str] = frozenset(
    {",", ".", "!", "?", ";", ":", "'", '"', ")", "]", "}"}
)
PUNCTUATION_NO_SPACE_AFTER: frozenset[str] = frozenset(
    {"(", "[", "{", '"', "'"}
)  # Note: Quote is tricky, might need refinement if used differently
# --- NEW SET ---
PUNCTUATION_SPACE_AFTER: frozenset[str] = frozenset(
    {".", ",", "!", "?", ":", ";"}
)  # Punctuation that usually needs a space after it
# --- END NEW SET ---
SENTENCE_ENDING_PUNCTUATION: frozenset[str] = frozenset({".", "!", "?"})
ALL_PUNCTUATION: frozenset[str] = PUNCTUATION_NO_SPACE_BEFORE.union(
    PUNCTUATION_NO_SPACE_AFTER
)

# --- Global Logger ---
log = logging.getLogger(__name__)

# --- Core Logic Functions (Accept config object) ---

# Inside whisperx_corrector_core.py
import re

# Use the previous sets again, maybe quote handling was the issue
PUNCTUATION_NO_SPACE_BEFORE: frozenset[str] = frozenset(
    {",", ".", "!", "?", ";", ":", "'", '"', ")", "]", "}"}
)
PUNCTUATION_NO_SPACE_AFTER: frozenset[str] = frozenset(
    {"(", "[", "{", '"', "'"}
)  # Includes quote as opening

# Inside whisperx_corrector_core.py
import re  # Keep import just in case needed later

# --- Corrected Constant Sets ---
PUNCTUATION_NO_SPACE_BEFORE: frozenset[str] = frozenset(
    {",", ".", "!", "?", ";", ":", ")", "]", "}", "'s", "'m", "'re", "'ve", "'ll", "'d"}
)  # NO quote
PUNCTUATION_NO_SPACE_AFTER: frozenset[str] = frozenset({"(", "[", "{"})  # NO quote

# ... other constants ...


def build_text_with_proper_spacing(words: List[WordToken]) -> str:
    """
    Rebuilds text from word tokens with smart spacing (Iterative approach v8).
    Uses corrected punctuation sets and specific quote handling.
    """
    if not words:
        return ""
    output_str = ""
    # Simple state: Are we likely inside opening quotes? This isn't foolproof for nested quotes.
    is_likely_inside_quotes = False

    for i, token in enumerate(words):
        current_word = str(token.get("word", "")).strip()
        if not current_word:
            continue

        needs_space_before = True

        # --- Determine if space needed BEFORE current word ---
        if not output_str:  # First actual word
            needs_space_before = False
        else:
            last_char = output_str[-1]  # Check last character added

            # 1. Don't add space if current word attaches backward
            if current_word in PUNCTUATION_NO_SPACE_BEFORE:
                needs_space_before = False
            # 2. Don't add space if previous char needs attachment forward
            elif last_char in PUNCTUATION_NO_SPACE_AFTER:
                needs_space_before = False
            # 3. Specific Quote Logic:
            elif current_word == '"':
                if is_likely_inside_quotes:  # Likely a closing quote
                    needs_space_before = False  # Attach to previous word
                # else: opening quote, default needs_space = True applies (unless prev was opening bracket)
            elif last_char == '"':
                # If the last char was a quote, need to know if it was opening or closing
                if (
                    not is_likely_inside_quotes
                ):  # If we just ADDED a closing quote, the next word needs space
                    needs_space_before = True
                else:  # If we just ADDED an opening quote, the next word attaches
                    needs_space_before = False

        # --- Append space and word ---
        if needs_space_before:
            output_str += " "
        output_str += current_word

        # --- Update Quote State ---
        if current_word == '"':
            is_likely_inside_quotes = not is_likely_inside_quotes

    return (
        output_str  # No strip() needed if logic is perfect (but maybe keep for safety?)
    )


def is_sentence_boundary(word: Union[str, WordToken]) -> bool:
    """Determine if a word likely represents a sentence boundary."""
    text: str = ""
    if isinstance(word, dict):
        text = str(word.get("word", "")).strip()
    elif isinstance(word, str):
        text = word.strip()
    return bool(text and text[-1] in SENTENCE_ENDING_PUNCTUATION)


# --- Segment Statistics Calculation ---


def _calculate_basic_stats(segment: Segment, words: List[WordToken]) -> SegmentStats:
    """Calculates initial stats like duration and word counts."""
    segment_start = segment.get("start", 0.0)
    segment_end = segment.get("end", 0.0)
    # Ensure start/end are numeric before calculating duration
    if not isinstance(segment_start, Number) or not isinstance(segment_end, Number):
        segment_duration = 0.0  # Treat as invalid if types are wrong
        reason = "non_numeric_segment_times"
    else:
        segment_duration = max(0.0, segment_end - segment_start)
        reason = "unknown"  # Default reason if duration is okay so far
    base_stats = SegmentStats(
        valid=False,
        reason=reason,
        total_words=len(words),
        segment_duration=segment_duration,
        word_count_with_timing=0,
    )
    if not words:
        base_stats["reason"] = "empty_segment"
    # Update reason if duration was non-positive, but don't override if already set
    elif segment_duration <= 0 and base_stats["reason"] == "unknown":
        base_stats["reason"] = "zero_or_negative_duration"
    return base_stats


def _extract_timing_data(
    words: List[WordToken], config: WhisperXCorrectorConfig
) -> Tuple[List[float], List[float], List[float], float]:
    """Extracts durations, gaps, time points, and total word time from valid words."""
    word_durations: List[float] = []
    word_gaps: List[float] = []
    time_points: List[float] = []
    total_word_time: float = 0.0
    # Filter for valid numeric start/end times BEFORE sorting
    valid_words = [
        w
        for w in words
        if isinstance(w.get("start"), Number)
        and isinstance(w.get("end"), Number)
        and w["end"] > w["start"]
    ]
    valid_words.sort(key=lambda w: w["start"])  # Sort only valid words

    for i, word in enumerate(valid_words):
        # We know start/end exist and are Numbers here due to filtering
        start, end = word["start"], word["end"]
        time_points.extend([start, end])
        duration = end - start
        word_durations.append(duration)
        total_word_time += duration
        if i < len(valid_words) - 1:
            next_word = valid_words[i + 1]
            # We know start/end exist for next_word too
            gap = next_word["start"] - end
            if gap >= config.min_gap_duration:  # Use config
                word_gaps.append(gap)
    return word_durations, word_gaps, time_points, total_word_time


# --- Main calculate_segment_statistics Function (Corrected) ---


def calculate_segment_statistics(
    segment: Segment, config: WhisperXCorrectorConfig
) -> SegmentStats:
    """
    Calculate comprehensive statistical properties of a segment using provided config.

    Args:
        segment: The segment dictionary.
        config: The configuration settings to use.

    Returns:
        A SegmentStats dictionary with calculated metrics.
    """
    words = segment.get("words", [])
    stats = _calculate_basic_stats(segment, words)
    # Return early if basic checks (empty words, bad duration, non-numeric times) failed
    if stats["reason"] not in [None, "unknown"]:
        stats["valid"] = False  # Ensure valid is False
        return stats

    # Extract timing data from potentially valid words
    word_durations, word_gaps, time_points, total_word_time = _extract_timing_data(
        words, config
    )
    stats["word_count_with_timing"] = len(word_durations)

    # --- Corrected Logic Block ---
    if stats["word_count_with_timing"] < 2:
        # If insufficient timed words, set reason and provide defaults
        stats["reason"] = "insufficient_timed_words"
        defaults = {
            "avg_word_duration": 0.0,
            "max_word_duration": 0.0,
            "min_word_duration": 0.0,
            "word_duration_variance": 0.0,
            "avg_gap": 0.0,
            "max_gap": 0.0,
            "time_span": 0.0,
            "speech_rate": 0.0,
            "time_utilization": 0.0,
            "total_gap_time": 0.0,
            "density": 0.0,
            "gap_stddev": 0.0,
            "duration_stddev": 0.0,
        }
        stats.update({k: v for k, v in defaults.items() if k not in stats})
        stats["valid"] = False  # Ensure valid is explicitly False
        return stats  # Return the invalid stats dict
    else:
        # --- Detailed calculations execute ONLY IF word_count_with_timing >= 2 ---
        stats["valid"], stats["reason"] = True, None  # Mark as valid
        segment_duration = stats["segment_duration"]  # Cache for use below

        # Calculate stats (use .item() to convert numpy types to Python floats)
        stats["avg_word_duration"] = np.mean(word_durations).item()
        stats["max_word_duration"] = np.max(word_durations).item()
        stats["min_word_duration"] = np.min(word_durations).item()
        stats["word_duration_variance"] = np.var(word_durations).item()

        if word_gaps:
            stats["avg_gap"] = np.mean(word_gaps).item()
            stats["max_gap"] = np.max(word_gaps).item()
            stats["total_gap_time"] = sum(word_gaps)
            try:
                stats["gap_stddev"] = (
                    statistics.stdev(word_gaps) if len(word_gaps) > 1 else 0.0
                )
            except statistics.StatisticsError:
                stats["gap_stddev"] = (
                    0.0  # Handle case with single gap after filtering? Unlikely.
                )
        else:
            (
                stats["avg_gap"],
                stats["max_gap"],
                stats["total_gap_time"],
                stats["gap_stddev"],
            ) = (0.0, 0.0, 0.0, 0.0)

        stats["time_span"] = max(time_points) - min(time_points) if time_points else 0.0

        # Ensure segment_duration is positive before dividing
        if segment_duration > 0:
            stats["speech_rate"] = stats["total_words"] / segment_duration
            stats["time_utilization"] = total_word_time / segment_duration
            stats["density"] = stats["time_utilization"]
        else:
            # Should be unreachable due to earlier checks, but defensive coding
            stats["speech_rate"], stats["time_utilization"], stats["density"] = (
                0.0,
                0.0,
                0.0,
            )

        try:
            stats["duration_stddev"] = (
                statistics.stdev(word_durations) if len(word_durations) > 1 else 0.0
            )
        except statistics.StatisticsError:
            stats["duration_stddev"] = 0.0

        return stats  # Return the valid stats dict-


# --- Anomaly Detection ---


def detect_anomalies(
    stats: SegmentStats, config: WhisperXCorrectorConfig
) -> Tuple[bool, SeverityLevel, List[str]]:
    """Detect timing anomalies based on statistics, returning severity and reasons."""
    if not stats.get("valid", False):
        return True, SeverityLevel.SEVERE, [stats.get("reason", "invalid_stats")]

    reasons: List[str] = []
    severity: SeverityLevel = SeverityLevel.NORMAL
    density = stats.get("density", 1.0)
    max_word_dur = stats.get("max_word_duration", 0.0)
    max_gap = stats.get("max_gap", 0.0)
    gap_stddev = stats.get("gap_stddev")
    time_util = stats.get("time_utilization", 1.0)
    speech_rate = stats.get("speech_rate", config.typical_speech_rate)
    seg_dur = stats.get("segment_duration", 0.0)

    # Severe conditions first
    if max_word_dur > config.suspicious_max_word_duration:
        reasons.append(f"extreme_word_duration:{max_word_dur:.2f}s")
        severity = SeverityLevel.SEVERE
    if max_gap > config.suspicious_max_gap:
        reasons.append(f"extreme_gap:{max_gap:.2f}s")
        severity = SeverityLevel.SEVERE
    # Moderate conditions if not severe
    if severity != SeverityLevel.SEVERE:
        mod_reasons = []
        if density < config.suspicious_density_threshold:
            mod_reasons.append(f"low_density:{density:.2f}")
        if max_word_dur > config.max_word_duration:
            mod_reasons.append(f"long_word_duration:{max_word_dur:.2f}s")
        if max_gap > config.max_gap_duration:
            mod_reasons.append(f"large_gap:{max_gap:.2f}s")
        if gap_stddev is not None and gap_stddev > config.suspicious_gap_stddev:
            mod_reasons.append(f"high_gap_variance:{gap_stddev:.2f}")
        if time_util < config.min_time_utilization:
            mod_reasons.append(f"low_time_utilization:{time_util:.2f}")
        if speech_rate < config.typical_speech_rate / config.max_speech_rate_deviation:
            mod_reasons.append(f"slow_speech_rate:{speech_rate:.2f}wps")
        if seg_dur > config.max_segment_duration:
            mod_reasons.append(f"segment_too_long:{seg_dur:.2f}s")
        if mod_reasons:
            severity = SeverityLevel.MODERATE
            reasons.extend(mod_reasons)
    return severity != SeverityLevel.NORMAL, severity, reasons


# --- Timestamp Correction ---


def fix_basic_word_timestamps(
    words: List[WordToken],
    segment_start: float,
    segment_end: float,
    config: WhisperXCorrectorConfig,
) -> List[WordToken]:
    """
    Fix missing, invalid, or inconsistent timestamps in words within segment boundaries.
    Handles missing start/end, end <= start, duration < min_duration, and clamps results.
    """
    if not words:
        return []
    fixed_words = []
    # Ensure segment times are valid and have minimum duration space
    segment_start = max(0.0, segment_start)
    segment_end = max(segment_start + config.min_word_duration, segment_end)
    last_valid_end_time = segment_start

    for i, word_data in enumerate(words):
        word = word_data.copy()
        word_value = word.get("word")  # Get value before check

        # --- Filter invalid word entries ---
        word_text_stripped = str(word_value).strip() if word_value is not None else ""
        if (
            "word" not in word
            or word_value is None
            or not word_text_stripped
            or word_text_stripped.lower() == "none"
        ):
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Skipping invalid word entry: {word_data}")
            continue
        word["word"] = word_text_stripped  # Store stripped version
        # --- End Filter ---

        # Add score if missing
        if "score" not in word:
            word["score"] = None

        # --- Get Initial Start/End ---
        start = word.get("start")
        end = word.get("end")

        # --- Validate/Fix Start Time ---
        if not isinstance(start, Number) or start < 0:
            start = None
        if start is None:
            start = last_valid_end_time
        # Ensure start doesn't overlap significantly with previous word's end
        start = max(start, last_valid_end_time)
        # Clamp start to segment boundaries (allowing space for min duration at the end)
        start = max(segment_start, min(segment_end - config.min_word_duration, start))

        # --- Validate/Fix End Time ---
        # Determine if original end time is valid *and* after start
        original_end_is_valid_and_after_start = isinstance(end, Number) and end > start

        if not original_end_is_valid_and_after_start:
            # CASE 1: Original end is invalid (None, non-numeric, or <= start)
            # We need to *estimate* or *force* a new end time.

            # If the original end was explicitly <= start, we prioritize minimum duration.
            if isinstance(end, Number) and end <= start:
                # Force minimum duration as the primary goal
                end = start + config.min_word_duration
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        f"Word '{word_text_stripped}' end <= start. Forcing min duration end: {end:.3f}"
                    )
            else:
                # Original end was None or non-numeric. Estimate using typical duration.
                estimated_end = start + config.typical_word_duration
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        f"Word '{word_text_stripped}' missing/invalid end. Estimating end: {estimated_end:.3f}"
                    )

                # Find upper bound from next word or segment end
                next_valid_start = None
                for j in range(i + 1, len(words)):
                    next_w_start = words[j].get("start")
                    # Check if next word's start is valid and after current start
                    if isinstance(next_w_start, Number) and next_w_start > start:
                        next_valid_start = next_w_start
                        break

                upper_bound = segment_end
                if next_valid_start is not None:
                    # Clamp by next word start, ensuring minimum gap
                    upper_bound = min(
                        segment_end, next_valid_start - config.min_gap_duration
                    )

                # Choose the smaller of the estimate and the upper bound
                potential_end = min(estimated_end, upper_bound)

                # Ensure the final end respects the minimum duration *when estimating*
                end = max(start + config.min_word_duration, potential_end)
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        f"Word '{word_text_stripped}' potential end after bounds: {potential_end:.3f}, final end after min_duration: {end:.3f}"
                    )

        elif end - start < config.min_word_duration:
            # CASE 2: Original end was valid and after start, but duration too small
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"Word '{word_text_stripped}' duration < min ({end-start:.3f} < {config.min_word_duration}). Enforcing min duration."
                )
            end = start + config.min_word_duration

        # --- Final Clamping and Assignment ---
        # Clamp end time to the overall segment boundary
        end = min(segment_end, end)
        # Final check: Ensure end is *at least* min_duration after start, even after segment clamping
        end = max(start + config.min_word_duration, end)

        word["start"], word["end"] = start, end
        last_valid_end_time = end  # Update for the next iteration's start calculation
        fixed_words.append(word)  # Add the fully fixed word

    # Final sort (should mostly be in order, but ensures correctness)
    fixed_words.sort(key=lambda w: w.get("start", 0.0))
    return fixed_words


def distribute_words_evenly(
    words: List[WordToken],
    segment_start: float,
    segment_end: float,
    config: WhisperXCorrectorConfig,
) -> List[WordToken]:
    """
    Distribute words evenly across a time span, adjusting duration based on word length.
    Used for segments with severe timestamp issues. Includes improved edge case handling.

    Args:
        words: List of word tokens to redistribute.
        segment_start: Start time of the segment.
        segment_end: End time of the segment.
        config: Configuration object.

    Returns:
        List of word tokens with redistributed timestamps.
    """
    if not words:
        return []
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"Applying synthetic timestamps for segment {segment_start:.2f}-{segment_end:.2f}"
        )
    segment_duration = segment_end - segment_start

    # --- Handle zero/negative segment duration ---
    if segment_duration <= 0:
        log.warning(
            f"Segment duration <= 0 ({segment_duration:.2f}s) in distribute_words_evenly. Assigning minimal duration sequentially."
        )
        current_time = segment_start
        for word in words:
            word["start"] = current_time
            word["end"] = current_time + config.min_word_duration
            current_time = word["end"]  # Move time forward based on added duration
        return words

    # --- Handle zero total characters ---
    total_chars = sum(len(w.get("word", "")) for w in words if w.get("word"))
    if total_chars <= 0:
        log.warning(
            "Segment has zero total characters for distribution. Distributing time equally per word."
        )
        total_chars = len(words)
        if total_chars == 0:
            log.error("Cannot distribute time: No words.")
            return words  # Should be caught by outer if

    # --- Calculate time available for words vs gaps ---
    num_gaps = max(0, len(words) - 1)
    min_total_gap_time = num_gaps * config.min_gap_duration

    # Check if minimum gaps already exceed segment duration
    if min_total_gap_time >= segment_duration:
        log.warning(
            f"Min gap time ({min_total_gap_time:.2f}s) >= segment duration ({segment_duration:.2f}s) for {len(words)} words. Assigning only minimal word durations without gaps."
        )
        time_for_words = (
            0.0  # No time left for actual word content duration beyond minimum
        )
    else:
        time_for_words = segment_duration - min_total_gap_time

    # --- Distribute Time ---
    current_time = segment_start
    new_words: List[WordToken] = []
    accumulated_word_duration = 0.0  # Track allocated time *for words*

    for i, word_data in enumerate(words):
        word = word_data.copy()
        char_count = max(
            1, len(str(word.get("word", "")))
        )  # Use str() for safety, max 1 char

        # Calculate proportional duration, handle time_for_words == 0 case
        prop_duration = (
            (char_count / total_chars) * time_for_words if time_for_words > 0 else 0.0
        )
        word_duration = max(config.min_word_duration, prop_duration)

        # Cap duration based on remaining allocatable *word* time
        # Ensure we don't overallocate the time_for_words budget
        max_duration_this_word = max(
            config.min_word_duration, time_for_words - accumulated_word_duration
        )
        word_duration = min(word_duration, max_duration_this_word)
        # Ensure minimum again after capping against remaining budget
        word_duration = max(config.min_word_duration, word_duration)

        # Assign start/end, clamping end to segment boundary
        word_start = current_time
        word_end = min(segment_end, current_time + word_duration)
        # Ensure end is strictly after start, respecting min duration
        word_end = max(word_start + config.min_word_duration, word_end)
        # Final clamp of end time to segment boundary
        word_end = min(segment_end, word_end)

        word["start"] = word_start
        word["end"] = word_end
        new_words.append(word)

        # Update accumulated time and current time cursor
        actual_word_duration = word["end"] - word["start"]
        accumulated_word_duration += (
            actual_word_duration  # Add duration of the word just placed
        )
        current_time = word["end"]  # Move cursor to the end of this word

        # If time_for_words was <= 0, stack words without gaps.
        if i < len(words) - 1 and time_for_words > 0:
            current_time = min(segment_end, current_time + config.min_gap_duration)
        # Ensure current_time doesn't exceed segment end after potentially adding gap
        current_time = min(current_time, segment_end)

    # --- Final Adjustment ---
    # Adjust last word end precisely to segment end if needed,
    # ensuring it still respects min duration relative to its start.
    if new_words:
        last_word = new_words[-1]
        if last_word["end"] < segment_end:
            # Extend end, but ensure it's still valid (>= start + min_duration)
            last_word["end"] = max(
                last_word["start"] + config.min_word_duration, segment_end
            )
        # Ensure last word doesn't exceed boundary after adjustment
        last_word["end"] = min(segment_end, last_word["end"])
        # If clamping end made duration too small, adjust start backward (optional, might cause overlap if first word)
        if i == 0 and last_word["end"] - last_word["start"] < config.min_word_duration:
            last_word["start"] = max(
                segment_start, last_word["end"] - config.min_word_duration
            )

    return new_words


def apply_local_fixes(
    words: List[WordToken], config: WhisperXCorrectorConfig
) -> List[WordToken]:
    """Apply local fixes: primarily cap excessive word durations."""
    if not words:
        return []
    fixed_words = [w.copy() for w in words]
    for word in fixed_words:
        start, end = word.get("start"), word.get("end")
        if isinstance(start, Number) and isinstance(end, Number):
            duration = end - start
            if duration > config.max_word_duration:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        f"Capping duration for word '{word.get('word','')}' from {duration:.2f}s"
                    )
                word["end"] = start + config.max_word_duration
    return fixed_words


# --- Segment Splitting/Merging ---

# Define type for split candidate dictionary for clarity
SplitCandidate = TypedDict(
    "SplitCandidate", {"index": int, "score": float, "gap": float, "is_hard": bool}
)


def _get_split_candidates(
    words: List[WordToken], config: WhisperXCorrectorConfig
) -> List[SplitCandidate]:
    """Identifies potential split points based on gaps and sentence boundaries."""
    split_candidates: List[SplitCandidate] = []
    for i in range(len(words) - 1):
        curr_w, next_w = words[i], words[i + 1]
        end_time, start_time = curr_w.get("end"), next_w.get("start")
        if not isinstance(end_time, Number) or not isinstance(start_time, Number):
            continue

        gap = start_time - end_time
        score = 0.0
        is_hard = False
        if is_sentence_boundary(curr_w):
            score += config.sentence_boundary_split_score
        if gap >= config.hard_gap_duration:
            score += config.hard_gap_split_score
            is_hard = True
        elif gap >= config.max_gap_duration:
            score += config.sentence_boundary_split_score * (
                gap / config.max_gap_duration
            )
        if score >= config.split_score_threshold or is_hard:
            split_candidates.append(
                SplitCandidate(index=i, score=score, gap=gap, is_hard=is_hard)
            )
    return split_candidates


def split_segment_at_optimal_points(
    segment: Segment, stats: SegmentStats, config: WhisperXCorrectorConfig
) -> List[Segment]:
    """Split a segment if too long or contains hard gaps, prioritizing linguistic boundaries."""
    words = segment.get("words", [])
    # Determine the initial reason(s) for potentially splitting
    duration_exceeded = stats.get("segment_duration", 0) > config.max_segment_duration
    hard_gap_exists = stats.get("max_gap", 0) > config.hard_gap_duration

    # Only proceed if there's a reason to split
    if not words or len(words) < 2 or not (duration_exceeded or hard_gap_exists):
        return [segment]

    split_candidates = _get_split_candidates(words, config)
    if not split_candidates:
        # Cannot split if no candidates found, even if duration is too long
        log.warning(
            f"Segment {segment['start']:.2f}-{segment['end']:.2f} exceeds max duration but no suitable split points found."
        )
        return [segment]

    selected_indices: Set[int] = set()
    # Always consider the best candidate first (highest score, prioritizing hard gaps)
    best_candidate = sorted(
        split_candidates, key=lambda x: (-x["is_hard"], -x["score"])
    )[0]

    # Reason 1: Hard gap always forces a split at that point (if candidates exist)
    if hard_gap_exists:
        # Find all hard gap candidates and select them
        hard_gap_indices = {
            cand["index"] for cand in split_candidates if cand["is_hard"]
        }
        selected_indices.update(hard_gap_indices)
        log.debug(f"Selecting hard gap split indices: {selected_indices}")

    # Reason 2: Duration exceeded forces at least one split using the best candidate
    # Add the best candidate *if* duration is exceeded AND no hard gap splits were already selected
    # OR if hard gaps exist but the best candidate offers a better score overall (unlikely but possible)
    if duration_exceeded and not selected_indices:
        log.debug(
            f"Duration exceeded ({stats.get('segment_duration', 0):.2f}s > {config.max_segment_duration}s). Selecting best candidate split at index {best_candidate['index']}."
        )
        selected_indices.add(best_candidate["index"])
    elif duration_exceeded and best_candidate["index"] not in selected_indices:
        # Optional: Consider adding the best non-hard split if segment is still too long between hard splits
        # This adds complexity, let's keep it simple first: Split only at hard gaps OR the single best point if duration is the only issue.
        # For now, if hard gaps exist, we only split there even if duration is long.
        pass

    # If after all checks, no indices were selected (e.g., only moderate gaps and duration OK)
    if not selected_indices:
        # This case should have been caught by the initial check, but double-check
        return [segment]

    # --- Segment Construction (remains the same) ---
    final_segments: List[Segment] = []
    start_idx = 0
    sorted_indices = sorted(list(selected_indices))
    for split_idx in sorted_indices:
        sub_words = words[start_idx : split_idx + 1]
        if not sub_words:
            continue
        sub_start = sub_words[0].get("start", segment["start"] if start_idx == 0 else 0)
        sub_end = sub_words[-1].get("end", segment["end"])
        final_segments.append(
            Segment(
                start=sub_start,
                end=sub_end,
                words=sub_words,
                text=build_text_with_proper_spacing(sub_words),
            )
        )
        start_idx = split_idx + 1
    if start_idx < len(words):  # Add remaining part
        sub_words = words[start_idx:]
        if sub_words:
            sub_start = sub_words[0].get("start", segment["start"])
            sub_end = segment["end"]
            final_segments.append(
                Segment(
                    start=sub_start,
                    end=sub_end,
                    words=sub_words,
                    text=build_text_with_proper_spacing(sub_words),
                )
            )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"Split segment {segment['start']:.2f}-{segment['end']:.2f} into {len(final_segments)} parts based on selected indices: {sorted_indices}."
        )
    return final_segments if final_segments else [segment]  # Fallback


def should_merge_segments(
    prev_segment: Segment, current_segment: Segment, config: WhisperXCorrectorConfig
) -> bool:
    """Determine if two adjacent segments should be merged (uses config)."""
    prev_end = prev_segment.get("end", 0.0)
    prev_start = prev_segment.get("start", 0.0)
    curr_start = current_segment.get("start", prev_end)
    curr_end = current_segment.get("end", curr_start)
    prev_words = prev_segment.get("words", [])
    curr_words = current_segment.get("words", [])
    prev_text = prev_segment.get("text", "").strip()

    if curr_start - prev_end > config.merge_time_gap_threshold:
        return False
    if prev_text and is_sentence_boundary(prev_text):
        return False
    if (
        curr_end - prev_start > config.max_segment_duration
        or len(prev_words) + len(curr_words) > config.max_merged_word_count
    ):
        return False
    curr_duration = curr_end - curr_start
    if (
        curr_duration < config.merge_min_word_duration_for_split
        and len(curr_words) < config.merge_min_word_count_for_split
    ):
        return True
    return True


def merge_segments(prev_segment: Segment, current_segment: Segment) -> Segment:
    """Merge two segments into one."""
    merged_words = prev_segment.get("words", []) + current_segment.get("words", [])
    merged_words.sort(key=lambda w: w.get("start", 0.0))
    merged = Segment(
        start=prev_segment.get("start", 0.0),
        end=current_segment.get("end", prev_segment.get("end", 0.0)),
        words=merged_words,
        text=build_text_with_proper_spacing(merged_words),
    )
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"Merged segments ending at {prev_segment.get('end', 0):.2f} and {current_segment.get('end', 0):.2f}"
        )
    return merged


# --- Segment Processing (Refactored, accepts config) ---


def _apply_correction_strategy(
    segment: Segment,
    stats: SegmentStats,
    severity: SeverityLevel,
    reasons: List[str],
    config: WhisperXCorrectorConfig,
) -> List[Segment]:
    """Applies correction based on detected anomaly severity, using config."""
    original_start = segment.get("start", 0.0)
    original_end = segment.get("end", 0.0)

    if severity == SeverityLevel.SEVERE:
        log.warning(
            f"Segment {original_start:.2f}-{original_end:.2f} SEVERE ({reasons}). Applying synthetic timestamps."
        )
        redist_start = (
            original_start if original_start < original_end else segment["start"]
        )
        redist_end = original_end if original_start < original_end else segment["end"]
        corrected_words = distribute_words_evenly(
            segment["words"], redist_start, redist_end, config
        )
        corrected_segment = Segment(
            start=redist_start,
            end=redist_end,
            words=corrected_words,
            text=build_text_with_proper_spacing(corrected_words),
        )
        corrected_stats = calculate_segment_statistics(corrected_segment, config)
        return split_segment_at_optimal_points(
            corrected_segment, corrected_stats, config
        )

    elif severity == SeverityLevel.MODERATE:
        log.info(
            f"Segment {original_start:.2f}-{original_end:.2f} MODERATE ({reasons}). Applying local fixes & splitting."
        )
        locally_fixed_words = apply_local_fixes(segment["words"], config)
        locally_fixed_segment = Segment(
            start=segment["start"],
            end=segment["end"],
            words=locally_fixed_words,
            text=build_text_with_proper_spacing(locally_fixed_words),
        )
        return split_segment_at_optimal_points(
            locally_fixed_segment, stats, config
        )  # Split based on original moderate stats

    else:  # NORMAL severity (Should be unreachable if called correctly)
        log.error(
            "_apply_correction_strategy called with NORMAL severity (logic error)."
        )
        return [segment]


def process_segment(segment: Segment, config: WhisperXCorrectorConfig) -> List[Segment]:
    """
    Process a single segment: fix timestamps, analyze, apply correction/splitting.

    Args:
        segment: The input segment dictionary.
        config: The configuration settings to use.

    Returns:
        A list of 0, 1, or more processed segment dictionaries. Returns empty
        if the segment becomes invalid during processing.

    Example:
        >>> cfg = WhisperXCorrectorConfig()
        >>> problematic_segment = {'start': 10.0, 'end': 15.0, 'text': 'Word A Word B', 'words': [{'word': 'Word A', 'start': 10.1, 'end': 14.8}, {'word': 'Word B', 'start': 14.8, 'end': 14.9}]}
        >>> corrected = process_segment(problematic_segment, cfg)
        >>> print(len(corrected)) # Might be 1 or more depending on thresholds
    """
    original_start = segment.get("start", 0.0)
    original_end = segment.get("end", 0.0)
    if not segment.get("words"):
        log.warning(f"Seg {original_start:.2f}-{original_end:.2f} has no words.")
        return []

    fixed_words = fix_basic_word_timestamps(
        segment["words"], original_start, original_end, config
    )
    if not fixed_words:
        log.warning(
            f"Seg {original_start:.2f}-{original_end:.2f}: No valid words after fixing."
        )
        return []

    current_segment = Segment(
        start=fixed_words[0].get("start", original_start),
        end=fixed_words[-1].get("end", original_end),
        words=fixed_words,
        text=build_text_with_proper_spacing(fixed_words),
    )
    stats = calculate_segment_statistics(current_segment, config)
    is_anomalous, severity, reasons = detect_anomalies(stats, config)
    processed_segments: List[Segment] = []

    if not is_anomalous:
        processed_segments = split_segment_at_optimal_points(
            current_segment, stats, config
        )
        if log.isEnabledFor(logging.DEBUG) and len(processed_segments) == 1:
            log.debug(f"Seg {original_start:.2f}-{original_end:.2f} passed.")
    else:
        processed_segments = _apply_correction_strategy(
            current_segment, stats, severity, reasons, config
        )

    # Final validation filter
    final_valid_segments = [
        seg
        for seg in processed_segments
        if seg.get("end", 0) > seg.get("start", 0) and seg.get("words")
    ]
    if len(final_valid_segments) != len(processed_segments):
        log.warning(
            f"Dropped {len(processed_segments) - len(final_valid_segments)} invalid segments during processing of {original_start:.2f}-{original_end:.2f}"
        )
    return final_valid_segments


# --- File I/O ---


def safe_write_json(data: Dict[str, Any], output_path: Path) -> None:
    """Safely write JSON data using a temporary file and atomic rename."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Failed creating parent dir for {output_path}: {e}")
        raise IOError(f"Failed creating parent dir for {output_path}") from e
    temp_file_path: Optional[Path] = None
    try:
        fd, temp_file_path_str = tempfile.mkstemp(
            suffix=".json.tmp", dir=output_path.parent
        )
        temp_file_path = Path(temp_file_path_str)
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        shutil.move(str(temp_file_path), output_path)
        temp_file_path = None  # Set to None after successful move
    except Exception as e:
        log.error(f"Error during safe write to {output_path}: {e}")
        raise IOError(f"Error writing safely to {output_path}") from e
    finally:
        if temp_file_path and temp_file_path.exists():  # Cleanup if move failed
            try:
                temp_file_path.unlink()
            except OSError as unlink_err:
                log.error(f"Failed removing temp file {temp_file_path}: {unlink_err}")


def load_json_file(input_path: Path) -> Dict[str, Any]:
    """
    Load and validate the basic structure and types of a WhisperX JSON file.

    Args:
        input_path: Path object for the input JSON file.

    Returns:
        Parsed JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        ValueError: If JSON is invalid, structure is wrong, or essential
                    segment keys/types are incorrect.
        IOError: If the file cannot be read due to other issues.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {input_path}: {e}") from e
    except Exception as e:
        raise IOError(f"Failed to read {input_path}") from e
    # Structure validation
    if (
        not isinstance(data, dict)
        or "segments" not in data
        or not isinstance(data["segments"], list)
    ):
        raise ValueError(f"Invalid JSON structure in {input_path}")
    # Enhanced segment validation
    for i, segment in enumerate(data["segments"]):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {i} in {input_path} is not a dict.")
        # Check required keys and their types more strictly
        if not all(k in segment for k in ["start", "end", "text"]):
            raise ValueError(f"Segment {i} in {input_path} missing required key(s).")
        if not isinstance(segment.get("start"), Number):
            raise ValueError(f"Segment {i} in {input_path} has non-numeric 'start'.")
        if not isinstance(segment.get("end"), Number):
            raise ValueError(f"Segment {i} in {input_path} has non-numeric 'end'.")
        if not isinstance(segment.get("text"), str):
            raise ValueError(f"Segment {i} in {input_path} has non-string 'text'.")
        if "words" in segment and not isinstance(segment.get("words"), list):
            raise ValueError(f"Segment {i} in {input_path} has invalid 'words' type.")
        # Could add validation for word structure too if needed
    return data


# --- Internal Processing Functions (accepts config) ---


def _process_single_file(
    input_file_path: Path, output_file_path: Path, config: WhisperXCorrectorConfig
) -> ProcessedStats:
    """Internal function to process one file sequentially, using provided config."""
    if log.isEnabledFor(logging.INFO):
        log.info(f"Processing file: {input_file_path} -> {output_file_path}")
    stats = ProcessedStats(
        anomalies_detected=0,
        severe_anomalies=0,
        moderate_anomalies=0,
        splits_performed=0,
        merges_performed=0,
        original_segments=0,
        final_segments=0,
        error=None,
    )
    try:
        data = load_json_file(input_file_path)  # Uses enhanced validation
        original_segments = data.get("segments", [])
        stats["original_segments"] = len(original_segments)
        if not original_segments:
            log.warning(f"No segments found in {input_file_path}")
            stats["error"] = "no_segments"
            data["segments"] = []
            # Write empty file
            if config.safe_write:
                safe_write_json(data, output_file_path)
            else:
                output_file_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            return stats

        processed_segments_intermediate: List[Segment] = []
        # Disable tqdm if logging DEBUG messages to avoid mixed output
        segment_iterator = tqdm(
            original_segments,
            desc=f"Correcting {input_file_path.name}",
            leave=False,
            disable=not sys.stdout.isatty() or log.getEffectiveLevel() <= logging.DEBUG,
        )
        for segment in segment_iterator:
            processed_list = process_segment(segment, config)
            processed_segments_intermediate.extend(processed_list)
            # Anomaly counting (approximate on original segment)
            seg_stats = calculate_segment_statistics(segment, config)
            is_anom, severity, _ = detect_anomalies(seg_stats, config)
            if is_anom:
                stats["anomalies_detected"] += 1
            if severity == SeverityLevel.SEVERE:
                stats["severe_anomalies"] += 1
            elif severity == SeverityLevel.MODERATE:
                stats["moderate_anomalies"] += 1
            if len(processed_list) > 1:
                stats["splits_performed"] += len(processed_list) - 1

        # Merge adjacent segments post-processing
        final_segments: List[Segment] = []
        if processed_segments_intermediate:
            processed_segments_intermediate.sort(key=lambda s: s.get("start", 0.0))
            final_segments.append(processed_segments_intermediate[0])
            for i in range(1, len(processed_segments_intermediate)):
                current_segment = processed_segments_intermediate[i]
                prev_segment = final_segments[-1]
                if should_merge_segments(prev_segment, current_segment, config):
                    final_segments[-1] = merge_segments(prev_segment, current_segment)
                    stats["merges_performed"] += 1
                else:
                    final_segments.append(current_segment)

        stats["final_segments"] = len(final_segments)
        data["segments"] = final_segments
        # Write output
        if config.safe_write:
            safe_write_json(data, output_file_path)
        else:
            output_file_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if log.isEnabledFor(logging.INFO):
            log.info(f"Successfully processed and saved {output_file_path}")

    except (ValueError, IOError, FileNotFoundError) as e:
        log.error(f"Error processing file {input_file_path}: {e}")
        stats["error"] = str(e)
    except Exception as e:
        log.exception(f"Unexpected error processing file {input_file_path}")
        stats["error"] = f"Unexpected: {type(e).__name__}"
    return stats


# --- File Iterator ---
def _find_json_files(directory: Path) -> Generator[Path, None, None]:
    """Yields Path objects for all JSON files found recursively in a directory."""
    yield from directory.rglob("*.json")


# --- Pipeline Integration Adapter ---


def correct_whisperx_outputs(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    overwrite: bool = False,
    config: WhisperXCorrectorConfig = DEFAULT_CONFIG,
) -> None:
    """
    Processes and corrects WhisperX JSON files in a directory. Main integration point.

    Args:
        input_dir: Path object or string to the input directory.
        output_dir: Path object or string to the output directory.
        overwrite: If True, overwrite existing files in the output directory.

    Raises:
        TypeError: If input path arguments are not strings or Path objects.
        ValueError: If directory paths are invalid after conversion to Path.
        FileNotFoundError: If input directory does not exist.
        IOError: On directory creation or file writing issues.
    """
    # --- Input Validation ---
    # Check type before converting to Path
    if not isinstance(input_dir, (str, Path)):
        raise TypeError(
            f"input_dir must be a string or Path object, got {type(input_dir)}"
        )
    if not isinstance(output_dir, (str, Path)):
        raise TypeError(
            f"output_dir must be a string or Path object, got {type(output_dir)}"
        )
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")

    # if not isinstance(config, WhisperXCorrectorConfig): raise TypeError("config must be an instance of WhisperXCorrectorConfig") # If config passed

    # Convert to Path objects and resolve EARLY for consistent handling
    try:
        input_path = Path(input_dir).resolve()
        output_path = Path(output_dir).resolve()
    except Exception as e:  # Catch potential errors during Path conversion/resolution
        log.exception(
            f"Failed to resolve input/output paths: {input_dir}, {output_dir}"
        )
        raise ValueError(f"Invalid input/output path provided: {e}") from e

    log.info(f"Input directory: '{input_dir}'")
    log.info(f"Output directory: '{output_dir}'")
    if overwrite:
        log.warning("Overwrite mode enabled.")

    # Validate resolved paths
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Could not create output directory {output_path}: {e}")
        raise IOError(f"Could not create output directory {output_path}") from e

    # Use the default config instance (or passed one if argument added)
    config = DEFAULT_CONFIG  # Modify here if config argument is added

    # --- Processing Loop ---
    success_count, error_count, skipped_count = 0, 0, 0
    file_iterator = _find_json_files(input_path)
    disable_tqdm = not sys.stdout.isatty() or log.getEffectiveLevel() <= logging.DEBUG
    tqdm_iterator = tqdm(
        file_iterator, desc="Correcting JSON files", unit="file", disable=disable_tqdm
    )

    for input_file_path in tqdm_iterator:  # Uses Path objects now
        try:
            relative_path = input_file_path.relative_to(input_path)
            output_file_path = output_path / relative_path
            if not overwrite and output_file_path.exists():
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(f"Skipping existing file: {output_file_path}")
                skipped_count += 1
                continue
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Pass config object to internal processor
            result = _process_single_file(input_file_path, output_file_path, config)
            if result["error"]:
                error_count += 1
                log.error(
                    f"Failed processing {input_file_path.name}: {result['error']}"
                )  # Use .name for brevity
            else:
                success_count += 1
        except Exception as e:
            log.exception(f"Critical error handling file {input_file_path.name}")
            error_count += 1

    # --- Completion Summary ---
    log.info("Correction process finished.")
    total_handled = success_count + error_count + skipped_count
    log.info(
        f"Summary: {success_count} succeeded, {error_count} failed, {skipped_count} skipped (Total files considered: {total_handled})."
    )
    if error_count > 0:
        log.warning(
            f"{error_count} files failed during processing. Please review logs."
        )


# --- Example Usage (Isolated) ---
def run_example_test():
    """Runs a simple test case using example data."""

    test_input_dir = Path("temp_test_input_core_v5")
    test_output_dir = Path("temp_test_output_core_v5")
    try:
        test_input_dir.mkdir(exist_ok=True)
        test_output_dir.mkdir(exist_ok=True)
        example_data = {
            "segments": [
                {
                    "start": 0.031,
                    "end": 11.002,
                    "text": " C'est trop un espion, là.",
                    "words": [
                        {"word": "C'est", "start": 0.031, "end": 2.273, "score": 0.633},
                        {"word": "trop", "start": 2.293, "end": 2.434, "score": 0.303},
                        {"word": "un", "start": 2.634, "end": 10.362, "score": 0.925},
                        {
                            "word": "espion,",
                            "start": 10.402,
                            "end": 10.722,
                            "score": 0.488,
                        },
                        {"word": "là.", "start": 10.762, "end": 11.002, "score": 0.406},
                    ],
                },
                {
                    "start": 11.042,
                    "end": 22.354,
                    "text": "Ouais, ouais, des petits nerds, des chasseurs qui travaillent à Doge, ces petits nerds, c'est ça, hein ?",
                    "words": [
                        {
                            "word": "Ouais,",
                            "start": 11.042,
                            "end": 11.323,
                            "score": 0.433,
                        },
                        {
                            "word": "ouais,",
                            "start": 18.05,
                            "end": 18.55,
                            "score": 0.312,
                        },
                        {"word": "des", "start": 18.59, "end": 18.65, "score": 0.45},
                        {
                            "word": "petits",
                            "start": 18.67,
                            "end": 18.891,
                            "score": 0.503,
                        },
                        {
                            "word": "nerds,",
                            "start": 18.911,
                            "end": 19.131,
                            "score": 0.57,
                        },
                        {"word": "des", "start": 19.151, "end": 19.231, "score": 0.771},
                        {
                            "word": "chasseurs",
                            "start": 19.251,
                            "end": 19.511,
                            "score": 0.529,
                        },
                        {"word": "qui", "start": 19.531, "end": 19.631, "score": 0.941},
                        {
                            "word": "travaillent",
                            "start": 19.672,
                            "end": 19.932,
                            "score": 0.876,
                        },
                        {"word": "à", "start": 20.012, "end": 20.052, "score": 0.327},
                        {
                            "word": "Doge,",
                            "start": 20.112,
                            "end": 20.532,
                            "score": 0.828,
                        },
                        {"word": "ces", "start": 20.572, "end": 20.653, "score": 0.639},
                        {
                            "word": "petits",
                            "start": 20.693,
                            "end": 20.873,
                            "score": 0.666,
                        },
                        {
                            "word": "nerds,",
                            "start": 20.893,
                            "end": 21.073,
                            "score": 0.258,
                        },
                        {
                            "word": "c'est",
                            "start": 21.093,
                            "end": 21.213,
                            "score": 0.653,
                        },
                        {"word": "ça,", "start": 21.253, "end": 21.413, "score": 0.696},
                        {
                            "word": "hein",
                            "start": 21.453,
                            "end": 22.354,
                            "score": 0.402,
                        },
                        {"word": "?"},
                    ],
                },
            ]
        }
        malformed_data_keys = {
            "segments": [{"start": 29.0, "end": 30.0, "words": [{"word": "Error"}]}]
        }  # Missing text
        malformed_data_types = {
            "segments": [
                {"start": "thirty", "end": 31.0, "text": "Bad start", "words": []}
            ]
        }  # Non-numeric start
        file1_path = test_input_dir / "transcript1.json"
        file2_path = test_input_dir / "transcript_malformed_keys.json"
        file3_path = test_input_dir / "transcript_malformed_types.json"
        with open(file1_path, "w", encoding="utf-8") as f:
            json.dump(example_data, f, indent=2)
        with open(file2_path, "w", encoding="utf-8") as f:
            json.dump(malformed_data_keys, f, indent=2)
        with open(file3_path, "w", encoding="utf-8") as f:
            json.dump(malformed_data_types, f, indent=2)

        print(f"\nCreated test files in {test_input_dir}")
        print(
            f"Running correction: {test_input_dir} -> {test_output_dir} (Log Level: DEBUG)"
        )
        correct_whisperx_outputs(
            str(test_input_dir), str(test_output_dir), overwrite=True
        )
        print("\nTest correction finished.")
        print(f"Please inspect contents of '{test_output_dir}'")
        # Basic checks
        if (test_output_dir / "transcript1.json").exists():
            print("- OK: Output for 'transcript1.json' found.")
        else:
            print("- ERROR: Output for 'transcript1.json' NOT found.")
        if not (test_output_dir / "transcript_malformed_keys.json").exists():
            print("- OK: Output for malformed keys file correctly skipped/failed.")
        else:
            print("- ERROR: Output for malformed keys file unexpectedly created.")
        if not (test_output_dir / "transcript_malformed_types.json").exists():
            print("- OK: Output for malformed types file correctly skipped/failed.")
        else:
            print("- ERROR: Output for malformed types file unexpectedly created.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        log.exception("Test execution failed")
    finally:
        pass  # Keep files


if __name__ == "__main__":
    run_example_test()

# tests/test_whisperx_corrector_core.py

import pytest
import statistics
import numpy as np 
from pathlib import Path
from typing import List, Dict, Any
from numbers import Number

# Import the module and specific functions/classes to test
from chronicleweave.modules.whisperx_corrector_core import (
    WhisperXCorrectorConfig,
    WordToken,
    Segment,
    SegmentStats,
    SeverityLevel,
    split_segment_at_optimal_points,
    build_text_with_proper_spacing,
    is_sentence_boundary,
    fix_basic_word_timestamps,
    calculate_segment_statistics, # Function to test
    detect_anomalies,
    distribute_words_evenly,
    apply_local_fixes,
    process_segment,
    should_merge_segments,
    merge_segments,
    # Import constants if needed, or use config
    DEFAULT_CONFIG,
)

# --- Fixtures (Optional reusable setup) ---

@pytest.fixture
def default_config() -> WhisperXCorrectorConfig:
    """Provides a default configuration instance for tests."""
    return WhisperXCorrectorConfig()

# ---------------------- #
# --- Test Functions --- #
# ---------------------- #

# 1. Test build_text_with_proper_spacing
def test_build_text_simple():
    words: List[WordToken] = [
        {"word": "Hello"},
        {"word": "world"},
        {"word": "."}
    ]
    assert build_text_with_proper_spacing(words) == "Hello world."

def test_build_text_punctuation_no_space_before():
    words: List[WordToken] = [
        {"word": "Okay"},
        {"word": ","},
        {"word": "let's"},
        {"word": "go"},
        {"word": "!"}
    ]
    assert build_text_with_proper_spacing(words) == "Okay, let's go!"

def test_build_text_punctuation_no_space_after():
    words: List[WordToken] = [
        {"word": "("},
        {"word": "Hello"},
        {"word": ")"}
    ]
    assert build_text_with_proper_spacing(words) == "(Hello)"

# --- Test Case 1: Mixed Punctuation (Expecting Failure before fix) ---
# This test should now pass after the fix in build_text_with_proper_spacing
def test_build_text_mixed_punctuation():
    words: List[WordToken] = [
        {"word": "He"},
        {"word": "said"},
        {"word": ":"},
        {"word": "\""},
        {"word": "Wait"},
        {"word": "!"},
        {"word": "\""}
    ]
    # Expected: No space after colon, space after closing quote (if followed by word)
    # The previous actual was 'He said:"Wait!"' - missing space after colon
    # Expected after fix: "He said: \"Wait!\"" (Note: repr shows escapes)
    assert build_text_with_proper_spacing(words) == "He said: \"Wait!\""


def test_build_text_empty_list():
    words: List[WordToken] = []
    assert build_text_with_proper_spacing(words) == ""

def test_build_text_with_empty_words():
     words: List[WordToken] = [
        {"word": "Start"},
        {"word": " "}, # Whitespace word
        {"word": ""},  # Empty word
        {"word": "End"},
        {"word": "."}
    ]
     assert build_text_with_proper_spacing(words) == "Start End." # Empty/whitespace words should be ignored

# 2. Test is_sentence_boundary
def test_is_sentence_boundary_true():
    assert is_sentence_boundary("Hello.") is True
    assert is_sentence_boundary("Go!") is True
    assert is_sentence_boundary("Why?") is True
    assert is_sentence_boundary({"word": "Finished."}) is True # Test with WordToken dict

def test_is_sentence_boundary_false():
    assert is_sentence_boundary("Hello") is False
    assert is_sentence_boundary("Go,") is False
    assert is_sentence_boundary("") is False
    assert is_sentence_boundary({"word": "Continue,"}) is False # Test with WordToken dict
    assert is_sentence_boundary({}) is False # Test empty dict
    assert is_sentence_boundary({"word": None}) is False # Test None word


# 3. Test fix_basic_word_timestamps (Basic Cases)

# --- Test Case 2: No Change (Expecting Failure before fix) ---
# This test failed because 'score': None was added by the function.
def test_fix_basic_timestamps_no_change(default_config):
    words: List[WordToken] = [
        {"word": "Word1", "start": 1.0, "end": 1.5},
        {"word": "Word2", "start": 1.6, "end": 2.0}
    ]
    # FIX: Define expected output including the added score
    expected_fixed: List[WordToken] = [
        {"word": "Word1", "start": 1.0, "end": 1.5, 'score': None},
        {"word": "Word2", "start": 1.6, "end": 2.0, 'score': None}
    ]
    fixed = fix_basic_word_timestamps(words, 0.0, 5.0, default_config)
    assert fixed == expected_fixed # Compare to the expected output

# --- Test Case 3: Missing End (Expecting Failure before fix) ---
# This test failed because the second word also got 'score': None added.
def test_fix_basic_timestamps_missing_end(default_config):
    words: List[WordToken] = [
        {"word": "Word1", "start": 1.0}, # Missing end
        {"word": "Word2", "start": 1.6, "end": 2.0}
    ]
    # FIX: Define expected output for the second word including score
    expected_word2_fixed: WordToken = {"word": "Word2", "start": 1.6, "end": 2.0, 'score': None}
    fixed = fix_basic_word_timestamps(words, 0.0, 5.0, default_config)
    assert fixed[0].get("end") is not None
    assert isinstance(fixed[0].get("end"), float)
    expected_end = min(1.0 + default_config.typical_word_duration, 1.6 - default_config.min_gap_duration)
    assert fixed[0]["end"] == pytest.approx(expected_end)
    assert fixed[0]["score"] is None # Check score was added to first word too
    assert fixed[1] == expected_word2_fixed # Compare second word to expected fixed version


# --- Test Case 4: Missing Start (Expecting Failure before fix) ---
# This test failed because the first word also got 'score': None added.
def test_fix_basic_timestamps_missing_start(default_config):
    words: List[WordToken] = [
        {"word": "Word1", "start": 1.0, "end": 1.5},
        {"word": "Word2", "end": 2.0} # Missing start
    ]
    # FIX: Define expected output for the first word including score
    expected_word1_fixed: WordToken = {"word": "Word1", "start": 1.0, "end": 1.5, 'score': None}
    fixed = fix_basic_word_timestamps(words, 0.0, 5.0, default_config)
    assert fixed[1].get("start") is not None
    assert isinstance(fixed[1].get("start"), float)
    assert fixed[1]["start"] == pytest.approx(1.5)
    assert fixed[1]["score"] is None # Check score was added to second word
    assert fixed[0] == expected_word1_fixed # Compare first word to expected fixed version


def test_fix_basic_timestamps_enforce_min_duration(default_config):
    words: List[WordToken] = [
        {"word": "Word1", "start": 1.0, "end": 1.01}, # Duration less than min
    ]
    fixed = fix_basic_word_timestamps(words, 0.0, 5.0, default_config)
    assert fixed[0]["end"] - fixed[0]["start"] == pytest.approx(default_config.min_word_duration)
    assert fixed[0]["start"] == 1.0 # Start shouldn't change
    assert fixed[0]["score"] is None # Check score

# --- Test Case 5: End Before Start (Expecting Failure before fix) ---
# This test failed because the duration was typical_word_duration (0.3), not min_word_duration (0.05)
def test_fix_basic_timestamps_end_before_start(default_config):
    words: List[WordToken] = [
        {"word": "Word1", "start": 1.0, "end": 0.5}, # End before start
    ]
    fixed = fix_basic_word_timestamps(words, 0.0, 5.0, default_config)
    assert fixed[0]["end"] > fixed[0]["start"]
    # FIX: Assert duration matches min_word_duration after fix
    assert fixed[0]["end"] - fixed[0]["start"] == pytest.approx(default_config.min_word_duration)
    assert fixed[0]["start"] == 1.0
    assert fixed[0]["score"] is None


def test_fix_basic_timestamps_clamp_to_segment(default_config):
    words: List[WordToken] = [
        {"word": "Word1", "start": -1.0, "end": 0.5}, # Starts before segment
        {"word": "Word2", "start": 4.8, "end": 5.5}  # Ends after segment
    ]
    fixed = fix_basic_word_timestamps(words, 0.0, 5.0, default_config)
    assert fixed[0]["start"] == 0.0
    assert fixed[0]["end"] == 0.5
    assert fixed[0]["score"] is None
    assert fixed[1]["start"] == 4.8
    assert fixed[1]["end"] == 5.0
    assert fixed[1]["score"] is None


def test_fix_basic_timestamps_empty_list(default_config):
     words: List[WordToken] = []
     fixed = fix_basic_word_timestamps(words, 0.0, 5.0, default_config)
     assert fixed == []

# --- Test Case 6: Invalid Word Entries (Expecting Failure before fix) ---
# This test failed because {"word": None, ...} was not filtered out correctly.
def test_fix_basic_timestamps_invalid_word_entries(default_config):
    words = [
        {"word": "Valid", "start": 1.0, "end": 1.5},
        {"start": 2.0, "end": 2.5}, # Missing 'word' key
        {"word": None, "start": 3.0, "end": 3.5}, # Word is None
        {"word": "  ", "start": 4.0, "end": 4.5}, # Word is whitespace
    ]
    fixed = fix_basic_word_timestamps(words, 0.0, 5.0, default_config)
    # FIX: Expect only the first valid word after fix in whisperx_corrector_core.py
    assert len(fixed) == 1
    assert fixed[0]["word"] == "Valid"
    assert fixed[0]["start"] == 1.0
    assert fixed[0]["end"] == 1.5
    assert fixed[0]["score"] is None

# ---------------------------------------------- #
# --- Tests for calculate_segment_statistics --- #
# ---------------------------------------------- #

def test_calculate_stats_normal_segment(default_config):
    """Test statistics calculation for a well-formed segment."""
    segment: Segment = {
        "start": 10.0, "end": 15.0, "text": "This is normal text.",
        "words": [
            {"word": "This", "start": 10.1, "end": 10.4, "score": 0.9}, # dur=0.3
            {"word": "is", "start": 10.5, "end": 10.7, "score": 0.9},   # dur=0.2, gap=0.1
            {"word": "normal", "start": 11.0, "end": 11.5, "score": 0.9},# dur=0.5, gap=0.3
            {"word": "text", "start": 11.6, "end": 12.0, "score": 0.9}, # dur=0.4, gap=0.1
            {"word": ".", "start": 12.0, "end": 12.1, "score": 0.9}    # dur=0.1, gap=0.0
        ]
    }
    stats = calculate_segment_statistics(segment, default_config)

    # --- Assertions ---
    assert stats["valid"] is True
    assert stats["reason"] is None
    assert stats["total_words"] == 5
    assert stats["word_count_with_timing"] == 5
    assert stats["segment_duration"] == pytest.approx(5.0)

    # Define expected data for clarity
    expected_durations = [0.3, 0.2, 0.5, 0.4, 0.1]
    # Gaps >= min_gap_duration (0.02): [0.1, 0.3, 0.1]
    expected_gaps = [0.1, 0.3, 0.1]
    total_word_time = sum(expected_durations) # 1.5

    # Duration Stats
    assert stats["avg_word_duration"] == pytest.approx(np.mean(expected_durations))
    assert stats["max_word_duration"] == pytest.approx(max(expected_durations))
    assert stats["min_word_duration"] == pytest.approx(min(expected_durations))
    assert stats["word_duration_variance"] == pytest.approx(np.var(expected_durations)) # Population variance from numpy
    # --- Use statistics.stdev for sample standard deviation ---
    assert stats.get("duration_stddev") == pytest.approx(statistics.stdev(expected_durations))

    # Gap Stats
    assert stats["avg_gap"] == pytest.approx(np.mean(expected_gaps))
    assert stats["max_gap"] == pytest.approx(max(expected_gaps))
    assert stats["total_gap_time"] == pytest.approx(sum(expected_gaps))
    # --- Use statistics.stdev for sample standard deviation ---
    assert stats.get("gap_stddev") == pytest.approx(statistics.stdev(expected_gaps))

    # Time Span: 12.1 - 10.1 = 2.0
    assert stats["time_span"] == pytest.approx(2.0)

    # Speech Rate: 5 words / 5.0s = 1.0 wps
    assert stats["speech_rate"] == pytest.approx(1.0)

    # Density / Time Utilization: total_word_time / segment_duration = 1.5 / 5.0 = 0.3
    assert stats["time_utilization"] == pytest.approx(total_word_time / 5.0)
    assert stats["density"] == pytest.approx(total_word_time / 5.0)


# --- Other tests for calculate_segment_statistics ---
# (test_calculate_stats_invalid_cases and test_calculate_stats_specific_features
#  should remain as they were, assuming they passed previously)

def test_calculate_stats_invalid_cases(default_config):
    """Test cases where stats should be marked invalid."""
    # Case 1: Empty words list
    segment1: Segment = {"start": 1.0, "end": 2.0, "text": "", "words": []}
    stats1 = calculate_segment_statistics(segment1, default_config)
    assert stats1["valid"] is False
    assert stats1["reason"] == "empty_segment"

    # Case 2: Zero duration segment
    segment2: Segment = {"start": 1.0, "end": 1.0, "text": "a", "words": [{"word":"a", "start":1.0, "end":1.0}]}
    stats2 = calculate_segment_statistics(segment2, default_config)
    assert stats2["valid"] is False
    assert stats2["reason"] == "zero_or_negative_duration"

    # Case 3: Only one word with timing
    segment3: Segment = {
        "start": 1.0, "end": 3.0, "text": "One word.",
        "words": [{"word": "One", "start": 1.1, "end": 1.5}] # Only one valid timed word
    }
    stats3 = calculate_segment_statistics(segment3, default_config)
    assert stats3["valid"] is False
    assert stats3["reason"] == "insufficient_timed_words"
    assert stats3["word_count_with_timing"] == 1

    # Case 4: Word with invalid time types
    segment4: Segment = {
        "start": 1.0, "end": 3.0, "text": "Bad time",
        "words": [{"word": "Bad", "start": "one", "end": 1.5}, {"word": "time", "start": 1.6, "end": None}]
    }
    stats4 = calculate_segment_statistics(segment4, default_config)
    assert stats4["valid"] is False
    assert stats4["reason"] == "insufficient_timed_words" # No valid timed words found
    assert stats4["word_count_with_timing"] == 0

def test_calculate_stats_specific_features(default_config):
    """Test calculation of specific stats like max_gap, max_word_duration."""
    segment: Segment = {
        "start": 20.0, "end": 30.0, "text": "Long word gap test",
        "words": [
            {"word": "Long", "start": 20.1, "end": 22.5, "score": 0.8},  # Long duration = 2.4s
            {"word": "word", "start": 22.6, "end": 23.0, "score": 0.8},  # Short duration, gap=0.1
            {"word": "gap", "start": 27.0, "end": 27.5, "score": 0.8},   # Large gap = 4.0s
            {"word": "test", "start": 27.6, "end": 28.0, "score": 0.8}   # Short duration, gap=0.1
        ]
    }
    stats = calculate_segment_statistics(segment, default_config)

    assert stats["valid"] is True
    assert stats["word_count_with_timing"] == 4
    assert stats["max_word_duration"] == pytest.approx(2.4)
    assert stats["min_word_duration"] == pytest.approx(0.4) # 23.0 - 22.6
    # Valid Gaps: [0.1, 4.0, 0.1]
    assert stats["max_gap"] == pytest.approx(4.0)
    assert stats["total_gap_time"] == pytest.approx(4.2)

# --- TODO: Add more tests for calculate_segment_statistics ---
# - Segment with only punctuation words?
# - Segment where all gaps are below min_gap_duration?
# - Segment with overlapping word times (how should stats handle this?)

# --- Tests for detect_anomalies ---

# Helper to create stats dict for testing detect_anomalies
def create_test_stats(overrides: Dict[str, Any]) -> SegmentStats:
    base = SegmentStats(
        valid=True, reason=None, total_words=10, segment_duration=10.0,
        word_count_with_timing=10, avg_word_duration=0.3, max_word_duration=0.8,
        min_word_duration=0.1, word_duration_variance=0.01, avg_gap=0.2, max_gap=0.5,
        total_gap_time=1.8, speech_rate=1.0, time_utilization=0.7, density=0.7,
        gap_stddev=0.1, duration_stddev=0.1, time_span=9.0
    )
    base.update(overrides) # Apply specific values for the test
    return base

def test_detect_anomalies_normal(default_config):
    stats = create_test_stats({}) # Use default valid stats
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is False
    assert severity == SeverityLevel.NORMAL
    assert reasons == []

def test_detect_anomalies_invalid_stats(default_config):
    stats = SegmentStats(valid=False, reason="test_reason")
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.SEVERE
    assert reasons == ["test_reason"]

def test_detect_anomalies_severe_duration(default_config):
    stats = create_test_stats({"max_word_duration": 10.0}) # Exceeds suspicious threshold
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.SEVERE
    assert "extreme_word_duration:10.00s" in reasons

def test_detect_anomalies_severe_gap(default_config):
    stats = create_test_stats({"max_gap": 10.0}) # Exceeds suspicious threshold
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.SEVERE
    assert "extreme_gap:10.00s" in reasons

def test_detect_anomalies_moderate_density(default_config):
    stats = create_test_stats({"density": 0.1}) # Below density threshold
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.MODERATE
    assert "low_density:0.10" in reasons

def test_detect_anomalies_moderate_duration(default_config):
    stats = create_test_stats({"max_word_duration": 3.0}) # Above max, below suspicious
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.MODERATE
    assert "long_word_duration:3.00s" in reasons

def test_detect_anomalies_moderate_gap(default_config):
    stats = create_test_stats({"max_gap": 3.0}) # Above max, below suspicious
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.MODERATE
    assert "large_gap:3.00s" in reasons

def test_detect_anomalies_moderate_gap_stddev(default_config):
    stats = create_test_stats({"gap_stddev": 3.0}) # Above suspicious threshold
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.MODERATE
    assert "high_gap_variance:3.00" in reasons

def test_detect_anomalies_moderate_time_util(default_config):
    stats = create_test_stats({"time_utilization": 0.1}) # Below threshold
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.MODERATE
    assert "low_time_utilization:0.10" in reasons

def test_detect_anomalies_moderate_speech_rate(default_config):
    # typical=3.0, deviation=3.0 -> min rate = 3.0 / 3.0 = 1.0
    stats = create_test_stats({"speech_rate": 0.5}) # Below threshold
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.MODERATE
    assert "slow_speech_rate:0.50wps" in reasons

def test_detect_anomalies_moderate_segment_length(default_config):
    stats = create_test_stats({"segment_duration": 20.0}) # Above max length
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.MODERATE
    assert "segment_too_long:20.00s" in reasons

def test_detect_anomalies_multiple_moderate(default_config):
    stats = create_test_stats({"density": 0.1, "max_gap": 3.0}) # Multiple moderate reasons
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.MODERATE
    assert len(reasons) == 2
    assert "low_density:0.10" in reasons
    assert "large_gap:3.00s" in reasons

def test_detect_anomalies_severe_overrides_moderate(default_config):
    # Has both a moderate and a severe condition
    stats = create_test_stats({"density": 0.1, "max_word_duration": 10.0})
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.SEVERE # Severe should take precedence
    assert "extreme_word_duration:10.00s" in reasons
    # It might also contain the moderate reason, depending on implementation,
    # but the severity level is the most important check here.
    # assert "low_density:0.10" in reasons # Check if moderate reasons also included

# ----------------------------------------- #
# --- Tests for distribute_words_evenly --- #
# ----------------------------------------- #

def test_distribute_words_evenly_normal(default_config):
    """Test even distribution based on character length."""
    # Total duration 10s. 3 words, 2 gaps. min_gap=0.02.
    # Time for words = 10.0 - (2 * 0.02) = 9.96s
    # Total chars = 5 + 3 + 4 = 12
    words: List[WordToken] = [
        {"word": "Hello", "score": 0.9}, # 5 chars
        {"word": "cruel", "score": 0.9}, # 5 chars
        {"word": "world", "score": 0.9}, # 5 chars
    ]
    segment_start, segment_end = 5.0, 15.0
    distributed = distribute_words_evenly(words, segment_start, segment_end, default_config)

    assert len(distributed) == 3
    total_duration_words = 0
    # Check start/end times are sequential and within bounds
    last_end = segment_start
    for i, word in enumerate(distributed):
        assert word["start"] == pytest.approx(last_end, abs=1e-3) # Start roughly where last ended (or segment start)
        assert word["end"] > word["start"]
        assert word["start"] >= segment_start
        assert word["end"] <= segment_end
        assert word["end"] - word["start"] >= default_config.min_word_duration
        total_duration_words += (word["end"] - word["start"])
        if i < len(distributed) - 1:
            # Account for the gap added internally
            last_end = word["end"] + default_config.min_gap_duration
        else:
            last_end = word["end"]

    # Check total word duration roughly equals available time
    time_for_words = segment_end - segment_start - (len(words) - 1) * default_config.min_gap_duration
    assert total_duration_words == pytest.approx(time_for_words)
    # Check final word ends exactly at segment end
    assert distributed[-1]["end"] == pytest.approx(segment_end)

def test_distribute_words_evenly_short_words_long_duration(default_config):
    """Test distribution where min duration might dominate."""
    # Duration 10s. 3 words, 2 gaps. min_gap=0.02. min_word_dur=0.05
    # Time for words = 9.96s. Total chars = 1+1+1=3
    # Prop duration = 9.96 / 3 = 3.32s per word. Min duration doesn't dominate.
    words: List[WordToken] = [{"word": "a"}, {"word": "b"}, {"word": "c"}]
    segment_start, segment_end = 0.0, 10.0
    distributed = distribute_words_evenly(words, segment_start, segment_end, default_config)

    assert len(distributed) == 3
    assert distributed[0]["start"] == pytest.approx(0.0)
    # word1 duration approx 3.32
    assert distributed[0]["end"] == pytest.approx(3.32)
    # gap = 0.02
    assert distributed[1]["start"] == pytest.approx(3.32 + 0.02)
    # word2 duration approx 3.32
    assert distributed[1]["end"] == pytest.approx(3.34 + 3.32)
    # gap = 0.02
    assert distributed[2]["start"] == pytest.approx(6.66 + 0.02)
    # word3 duration approx 3.32, but clamped to end
    assert distributed[2]["end"] == pytest.approx(10.0)

def test_distribute_words_evenly_many_words_short_duration(default_config):
    """Test case where min duration forces the timing."""
    # Duration 0.5s. 5 words, 4 gaps. min_gap=0.02, min_word_dur=0.05
    # Min total gap time = 4 * 0.02 = 0.08s
    # Min total word time = 5 * 0.05 = 0.25s
    # Total min time = 0.08 + 0.25 = 0.33s (less than 0.5s duration, so proportional applies)
    # Time for words = 0.5 - 0.08 = 0.42s. Total chars = 5*4 = 20.
    # Prop duration = 0.42 * (4/20) = 0.084s (which is > min_word_dur)
    words: List[WordToken] = [{"word": "word"} for _ in range(5)]
    segment_start, segment_end = 0.0, 0.5
    distributed = distribute_words_evenly(words, segment_start, segment_end, default_config)

    assert len(distributed) == 5
    # Check total duration matches segment duration
    total_calc_duration = (distributed[-1]["end"] - distributed[0]["start"])
    assert total_calc_duration == pytest.approx(0.5)
    # Check individual durations are >= min_word_duration
    for word in distributed:
        assert word["end"] - word["start"] >= default_config.min_word_duration - 1e-9 # Allow for float tolerance

    # Example where min duration *does* dominate:
    segment_start, segment_end = 0.0, 0.3 # Duration 0.3s, min needed is 0.33s
    distributed = distribute_words_evenly(words, segment_start, segment_end, default_config)
    assert len(distributed) == 5
    # Expect words to have roughly minimal duration, potentially slightly less if clamped
    # Check final word ends at boundary
    assert distributed[-1]["end"] == pytest.approx(0.3)


def test_distribute_words_evenly_zero_duration_segment(default_config):
    """Test behavior with zero duration segment."""
    words: List[WordToken] = [{"word": "a"}, {"word": "b"}]
    segment_start, segment_end = 1.0, 1.0
    distributed = distribute_words_evenly(words, segment_start, segment_end, default_config)

    assert len(distributed) == 2
    # Expect words assigned minimal duration starting at segment_start
    assert distributed[0]["start"] == pytest.approx(1.0)
    assert distributed[0]["end"] == pytest.approx(1.0 + default_config.min_word_duration)
    assert distributed[1]["start"] == pytest.approx(1.0 + default_config.min_word_duration)
    assert distributed[1]["end"] == pytest.approx(1.0 + 2 * default_config.min_word_duration)

def test_distribute_words_evenly_gaps_exceed_duration(default_config):
    """Test behavior when min gaps alone exceed segment duration."""
    # Duration 0.1s. 4 words, 3 gaps. min_gap=0.05 (override for test)
    # Min total gap time = 3 * 0.05 = 0.15s > 0.1s duration
    words: List[WordToken] = [{"word": "tst"} for _ in range(4)]
    segment_start, segment_end = 0.0, 0.1
    # Create a temp config with larger min_gap
    test_config = WhisperXCorrectorConfig(min_gap_duration=0.05)
    distributed = distribute_words_evenly(words, segment_start, segment_end, test_config)

    assert len(distributed) == 4
    # Expect words to have minimal duration, clamped within the tiny segment
    assert distributed[0]["start"] == pytest.approx(0.0)
    assert distributed[0]["end"] == pytest.approx(test_config.min_word_duration) # 0.05
    assert distributed[1]["start"] == pytest.approx(0.05) # Starts right after first word
    assert distributed[1]["end"] == pytest.approx(0.05 + test_config.min_word_duration) # 0.1
    assert distributed[2]["start"] == pytest.approx(0.1) # Clamped
    assert distributed[2]["end"] == pytest.approx(0.1) # Clamped end
    assert distributed[3]["start"] == pytest.approx(0.1) # Clamped
    assert distributed[3]["end"] == pytest.approx(0.1) # Clamped end


def test_distribute_words_evenly_empty_list(default_config):
    """Test with an empty list of words."""
    words: List[WordToken] = []
    segment_start, segment_end = 0.0, 1.0
    distributed = distribute_words_evenly(words, segment_start, segment_end, default_config)
    assert distributed == []

# ----------------------------------- #
# --- Tests for apply_local_fixes --- #
# ----------------------------------- #

def test_apply_local_fixes_no_change(default_config):
    """Test with words already within duration limits."""
    words: List[WordToken] = [
        {"word": "Normal1", "start": 1.0, "end": 1.5, "score": 0.9}, # Duration 0.5s
        {"word": "Normal2", "start": 2.0, "end": 3.9, "score": 0.9}, # Duration 1.9s
    ]
    # Create copies for comparison as the function modifies copies internally
    original_words = [w.copy() for w in words]
    fixed_words = apply_local_fixes(words, default_config)
    # Expect no changes as durations are less than max_word_duration (2.0)
    assert fixed_words == original_words

def test_apply_local_fixes_caps_duration(default_config):
    """Test that excessive word duration is capped."""
    words: List[WordToken] = [
        {"word": "Short", "start": 1.0, "end": 1.5, "score": 0.9},      # Duration 0.5s
        {"word": "Looooooooong", "start": 2.0, "end": 5.0, "score": 0.9}, # Duration 3.0s
        {"word": "Normal", "start": 5.5, "end": 6.0, "score": 0.9},     # Duration 0.5s
    ]
    max_dur = default_config.max_word_duration # Should be 2.0s
    fixed_words = apply_local_fixes(words, default_config)

    # Check first word (should be unchanged)
    assert fixed_words[0]["start"] == 1.0
    assert fixed_words[0]["end"] == 1.5

    # Check second word (should be capped)
    assert fixed_words[1]["start"] == 2.0
    assert fixed_words[1]["end"] == pytest.approx(2.0 + max_dur) # End should be start + max_dur
    assert fixed_words[1]["end"] == pytest.approx(4.0)

    # Check third word (should be unchanged)
    assert fixed_words[2]["start"] == 5.5
    assert fixed_words[2]["end"] == 6.0

def test_apply_local_fixes_multiple_caps(default_config):
    """Test capping multiple long words."""
    words: List[WordToken] = [
        {"word": "Long1", "start": 1.0, "end": 4.0, "score": 0.9}, # Duration 3.0s
        {"word": "Long2", "start": 5.0, "end": 9.0, "score": 0.9}, # Duration 4.0s
    ]
    max_dur = default_config.max_word_duration
    fixed_words = apply_local_fixes(words, default_config)

    assert fixed_words[0]["start"] == 1.0
    assert fixed_words[0]["end"] == pytest.approx(1.0 + max_dur) # 3.0
    assert fixed_words[1]["start"] == 5.0
    assert fixed_words[1]["end"] == pytest.approx(5.0 + max_dur) # 7.0

def test_apply_local_fixes_empty_list(default_config):
    """Test with an empty list."""
    words: List[WordToken] = []
    fixed_words = apply_local_fixes(words, default_config)
    assert fixed_words == []

def test_apply_local_fixes_missing_times(default_config):
    """Test words with missing start/end times (should be ignored by this fix)."""
    words: List[WordToken] = [
        {"word": "Word1", "start": 1.0}, # Missing end
        {"word": "Word2", "end": 5.0},   # Missing start
    ]
    original_words = [w.copy() for w in words]
    fixed_words = apply_local_fixes(words, default_config)
    # apply_local_fixes currently only checks if start/end exist and are numbers.
    # It doesn't *fix* missing times, only caps duration if times are valid.
    # Therefore, it should not modify these entries.
    assert fixed_words == original_words


# ------------------------------------------------- #
# --- Tests for split_segment_at_optimal_points --- #
# ------------------------------------------------- #

# Helper to create a basic segment for splitting tests
def create_segment_for_splitting(
    start: float, end: float, words: List[WordToken]
    ) -> Segment:
    return Segment(start=start, end=end, text=" ".join(w.get("word","") for w in words), words=words)

# Helper to quickly create a mock stats object for splitting decisions
def create_mock_stats(duration: float, max_gap: float) -> SegmentStats:
     # Only includes fields relevant to *triggering* the split logic
     return SegmentStats(valid=True, segment_duration=duration, max_gap=max_gap, word_count_with_timing=10) # word count > 2

def test_split_no_split_needed(default_config):
    """Segment within limits, no large gaps."""
    words: List[WordToken] = [
        {"word": "Short", "start": 1.0, "end": 1.5},
        {"word": "segment", "start": 1.7, "end": 2.5}, # Gap = 0.2s
        {"word": "okay.", "start": 2.6, "end": 3.0},   # Gap = 0.1s
    ]
    segment = create_segment_for_splitting(1.0, 3.0, words)
    # Stats indicate no reason to split (duration < max, max_gap < hard)
    stats = create_mock_stats(duration=segment["end"]-segment["start"], max_gap=0.2)
    split_segments = split_segment_at_optimal_points(segment, stats, default_config)
    assert len(split_segments) == 1
    assert split_segments[0] == segment

def test_split_due_to_max_duration(default_config):
    """Segment exceeds max_segment_duration, split should occur at best moderate point."""
    words: List[WordToken] = [
        {"word": "This", "start": 0.0, "end": 0.5},
        {"word": "is", "start": 0.6, "end": 0.8},     # Gap = 0.1
        {"word": "a", "start": 1.0, "end": 1.2},      # Gap = 0.2
        # --- Assume largest moderate gap is here ---
        {"word": "very", "start": 5.2, "end": 5.7},   # Gap = 4.0 (moderate)
        {"word": "long", "start": 6.0, "end": 6.5},   # Gap = 0.3
        {"word": "segment.", "start": 6.8, "end": 7.5} # Gap = 0.3. Total duration 7.5s. Let's make it long...
    ]
    # Adjust times to make total duration > 15s
    words[4]["start"] = 15.0; words[4]["end"] = 15.5
    words[5]["start"] = 15.8; words[5]["end"] = 16.5 # Total duration now 16.5s

    segment = create_segment_for_splitting(0.0, 16.5, words)
    # Stats: duration > max, max_gap is moderate (4.0s after index 3)
    stats = create_mock_stats(duration=16.5, max_gap=4.0)
    split_segments = split_segment_at_optimal_points(segment, stats, default_config)

    assert len(split_segments) > 1 # Should split

    # Logic should split after index 3 ("very") as it has the highest score moderate gap
    assert len(split_segments) == 2 # Expect exactly one split point chosen
    # Segment 1 words: words[0] to words[3] inclusive
    assert split_segments[0]["words"] == words[0:4] # "This", "is", "a", "very"
    assert split_segments[0]["end"] == pytest.approx(5.7) # End time of "very"
    # Segment 2 words: words[4] onwards
    assert split_segments[1]["words"] == words[4:] # "long", "segment."
    assert split_segments[1]["start"] == pytest.approx(15.0) # Start time of "long"
    
def test_split_due_to_hard_gap(default_config):
    """Segment contains a gap larger than hard_gap_duration."""
    words: List[WordToken] = [
        {"word": "Part", "start": 1.0, "end": 1.5},
        {"word": "one.", "start": 1.6, "end": 2.0}, # Ends at 2.0
        # Hard Gap: Starts at 10.0 -> 8.0s > hard_gap_duration (5.0s)
        {"word": "Part", "start": 10.0, "end": 10.5},
        {"word": "two.", "start": 10.6, "end": 11.0},
    ]
    segment = create_segment_for_splitting(1.0, 11.0, words)
    # Stats indicate hard gap present
    stats = create_mock_stats(duration=10.0, max_gap=8.0)
    split_segments = split_segment_at_optimal_points(segment, stats, default_config)
    assert len(split_segments) == 2
    assert split_segments[0]["words"] == words[0:2]
    assert split_segments[0]["end"] == pytest.approx(2.0)
    assert split_segments[1]["words"] == words[2:]
    assert split_segments[1]["start"] == pytest.approx(10.0)

def test_split_due_to_moderate_gap(default_config):
    """Segment has gap > max_gap_duration but < hard_gap_duration."""
    words: List[WordToken] = [
        {"word": "Before", "start": 1.0, "end": 1.5},
        # Moderate Gap: 3.0s (where max=2.0, hard=5.0)
        {"word": "After", "start": 4.5, "end": 5.0},
    ]
    segment = create_segment_for_splitting(1.0, 5.0, words)
    stats = create_mock_stats(duration=4.0, max_gap=3.0)
    split_segments = split_segment_at_optimal_points(segment, stats, default_config)
    # Moderate gap alone might not force a split if total duration is short,
    # but combined with exceeding max duration OR being near sentence end it might.
    # Current logic: Split IS triggered if candidate found and duration>max OR hard gap.
    # Let's refine the trigger: Split if hard_gap OR (duration > max AND moderate_gap/sentence_end exists)
    # For this test, duration is okay (4.0 < 15.0), gap is only moderate. *Should not split*.
    assert len(split_segments) == 1 # Expect NO split for moderate gap alone if duration is ok
    assert split_segments[0] == segment

    # --- Test Case where moderate gap DOES cause split due to duration ---
    long_words: List[WordToken] = [
        {"word": "Start", "start": 0.0, "end": 1.0},
        # Moderate Gap: 3.0s
        {"word": "Middle", "start": 4.0, "end": 10.0},
        # Moderate Gap: 3.0s
        {"word": "End", "start": 13.0, "end": 17.0}, # Total duration 17.0s > 15s
    ]
    long_segment = create_segment_for_splitting(0.0, 17.0, long_words)
    long_stats = create_mock_stats(duration=17.0, max_gap=3.0)
    long_split_segments = split_segment_at_optimal_points(long_segment, long_stats, default_config)
    assert len(long_split_segments) > 1 # Expect split due to duration & available moderate gaps

def test_split_prefers_sentence_boundary(default_config):
    """Test if split prefers sentence end over slightly smaller moderate gap."""
    words: List[WordToken] = [
        {"word": "Sentence", "start": 0.0, "end": 1.0},
        {"word": "one.", "start": 1.1, "end": 1.5}, # Sentence end here. Gap after = 2.6s (moderate)
        {"word": "Middle", "start": 4.1, "end": 4.5}, # Gap after = 3.0s (moderate, slightly larger)
        {"word": "Sentence", "start": 7.5, "end": 8.0},
        {"word": "two.", "start": 8.1, "end": 8.5},
    ]
    # Make segment long enough to force a split
    segment = create_segment_for_splitting(0.0, 20.0, words) # Pad end time
    stats = create_mock_stats(duration=20.0, max_gap=3.0)
    split_segments = split_segment_at_optimal_points(segment, stats, default_config)

    assert len(split_segments) == 2 # Should split once due to duration
    # Expect split after "one." because sentence boundary likely has higher score
    # than the slightly larger gap after "Middle"
    assert split_segments[0]["words"] == words[0:2] # Sentence one.
    assert split_segments[0]["end"] == pytest.approx(1.5)
    assert split_segments[1]["words"] == words[2:] # Middle Sentence two.
    assert split_segments[1]["start"] == pytest.approx(4.1)

def test_split_edge_case_two_words(default_config):
    """Test splitting with only two words - should generally not split."""
    # Case 1: Hard gap
    words_hard: List[WordToken] = [
        {"word": "Word1.", "start": 0.0, "end": 1.0},
        {"word": "Word2", "start": 10.0, "end": 11.0}, # Hard gap
    ]
    segment_hard = create_segment_for_splitting(0.0, 11.0, words_hard)
    stats_hard = create_mock_stats(duration=11.0, max_gap=9.0)
    split_hard = split_segment_at_optimal_points(segment_hard, stats_hard, default_config)
    assert len(split_hard) == 2 # Hard gap should force split even with 2 words

    # Case 2: Moderate gap, short duration
    words_mod: List[WordToken] = [
        {"word": "Word1", "start": 0.0, "end": 1.0},
        {"word": "Word2", "start": 4.0, "end": 5.0}, # Moderate gap
    ]
    segment_mod = create_segment_for_splitting(0.0, 5.0, words_mod)
    stats_mod = create_mock_stats(duration=5.0, max_gap=3.0)
    split_mod = split_segment_at_optimal_points(segment_mod, stats_mod, default_config)
    assert len(split_mod) == 1 # Moderate gap alone shouldn't split short segment

# --------------------------------------- #
# --- Tests for Segment Merging Logic --- #
# --------------------------------------- #

# Helper to create simple segments for merging tests
def create_segment(start: float, end: float, text: str, word_count: int = 5) -> Segment:
    # Generate dummy words - precise timing isn't crucial for merge decision logic testing
    # except for the overall start/end. Word count matters.
    words = [{"word": f"w{i+1}"} for i in range(word_count)]
    if words: # Assign basic times if words exist
        words[0]['start'] = start + 0.1
        words[0]['end'] = start + 0.3
        words[-1]['start'] = max(start + 0.1, end - 0.3)
        words[-1]['end'] = end
    return Segment(start=start, end=end, text=text, words=words)

# --- Tests for should_merge_segments ---

def test_should_merge_true_normal(default_config):
    """Test normal case where segments should merge."""
    seg1 = create_segment(start=1.0, end=5.0, text="Segment one")
    # Gap = 5.1 - 5.0 = 0.1s (< merge_time_gap_threshold=0.5)
    seg2 = create_segment(start=5.1, end=8.0, text="segment two.")
    # Combined duration = 8.0 - 1.0 = 7.0s (< max_segment_duration=15.0)
    # Combined words = 5 + 5 = 10 (< max_merged_word_count=35)
    # Seg1 does not end with sentence punctuation.
    assert should_merge_segments(seg1, seg2, default_config) is True

def test_should_merge_false_time_gap(default_config):
    """Test merging should be false due to large time gap."""
    seg1 = create_segment(start=1.0, end=5.0, text="Segment one")
    # Gap = 6.0 - 5.0 = 1.0s (> merge_time_gap_threshold=0.5)
    seg2 = create_segment(start=6.0, end=8.0, text="segment two.")
    assert should_merge_segments(seg1, seg2, default_config) is False

def test_should_merge_false_sentence_boundary(default_config):
    """Test merging should be false due to sentence boundary."""
    seg1 = create_segment(start=1.0, end=5.0, text="Segment one.") # Ends with period
    seg2 = create_segment(start=5.1, end=8.0, text="Segment two.")
    assert should_merge_segments(seg1, seg2, default_config) is False

def test_should_merge_false_combined_duration(default_config):
    """Test merging should be false due to exceeding max combined duration."""
    seg1 = create_segment(start=1.0, end=10.0, text="Long segment one")
    seg2 = create_segment(start=10.1, end=17.0, text="long segment two.")
    # Combined duration = 17.0 - 1.0 = 16.0s (> max_segment_duration=15.0)
    assert should_merge_segments(seg1, seg2, default_config) is False

def test_should_merge_false_combined_word_count(default_config):
    """Test merging should be false due to exceeding max combined word count."""
    seg1 = create_segment(start=1.0, end=5.0, text="Many words", word_count=20)
    seg2 = create_segment(start=5.1, end=8.0, text="many more words.", word_count=20)
    # Combined words = 20 + 20 = 40 (> max_merged_word_count=35)
    assert should_merge_segments(seg1, seg2, default_config) is False

def test_should_merge_true_short_second_segment(default_config):
    """Test merging is encouraged if second segment is short/simple."""
    seg1 = create_segment(start=1.0, end=5.0, text="Main part")
    # Short duration (< 1.0s) and few words (< 3)
    seg2 = create_segment(start=5.1, end=5.8, text="uh huh.", word_count=2)
    assert should_merge_segments(seg1, seg2, default_config) is True

def test_should_merge_true_short_second_segment_but_sentence_end(default_config):
    """Test sentence boundary still prevents merge even if second segment is short."""
    seg1 = create_segment(start=1.0, end=5.0, text="Main part.") # Ends with period
    seg2 = create_segment(start=5.1, end=5.8, text="OK.", word_count=1)
    assert should_merge_segments(seg1, seg2, default_config) is False


# --- Tests for merge_segments ---

def test_merge_segments_basic(default_config):
    """Test basic merging of two segments."""
    words1 = [{"word": "Seg1", "start": 1.1, "end": 1.5}]
    words2 = [{"word": "Seg2", "start": 2.1, "end": 2.5}]
    seg1 = Segment(start=1.0, end=1.8, text="Seg1", words=words1)
    seg2 = Segment(start=2.0, end=3.0, text="Seg2", words=words2)

    merged = merge_segments(seg1, seg2)

    # Check start/end times
    assert merged["start"] == seg1["start"] # Should take start of first
    assert merged["end"] == seg2["end"]     # Should take end of second

    # Check combined words (ensure order is maintained if timestamps overlap slightly)
    assert merged["words"] == words1 + words2

    # Check rebuilt text
    assert merged["text"] == "Seg1 Seg2" # Based on build_text_with_proper_spacing

def test_merge_segments_with_punctuation(default_config):
    """Test merging handles text spacing with punctuation."""
    words1 = [{"word": "Hello", "start": 1.1, "end": 1.5}]
    words2 = [{"word": ",", "start": 1.6, "end": 1.7}, {"word": "world", "start": 1.8, "end": 2.5}]
    seg1 = Segment(start=1.0, end=1.5, text="Hello", words=words1)
    seg2 = Segment(start=1.6, end=2.5, text=", world", words=words2)

    merged = merge_segments(seg1, seg2)

    assert merged["start"] == 1.0
    assert merged["end"] == 2.5
    assert merged["words"] == words1 + words2
    # Expect build_text_with_proper_spacing to handle the comma correctly
    assert merged["text"] == "Hello, world"

# --------------------------------- #
# --- Tests for process_segment --- #
# --------------------------------- #

def test_process_segment_normal(default_config):
    """Test process_segment with a segment that should require no correction/splitting."""
    segment_in: Segment = {
        "start": 10.0, "end": 13.0, "text": "This is fine.",
        "words": [
            {"word": "This", "start": 10.1, "end": 10.4},
            {"word": "is", "start": 10.5, "end": 10.7},
            {"word": "fine", "start": 10.9, "end": 11.3},
            {"word": ".", "start": 11.3, "end": 11.4},
        ]
    }
    processed_segments = process_segment(segment_in, default_config)

    # Expect 1 segment back, largely unchanged (though 'score': None added)
    assert len(processed_segments) == 1
    out_segment = processed_segments[0]
    assert out_segment["start"] == pytest.approx(10.1) # Start might adjust to first word
    assert out_segment["end"] == pytest.approx(11.4)   # End might adjust to last word
    assert out_segment["text"] == "This is fine."
    # Check if words got score added
    assert all("score" in w for w in out_segment["words"])

def test_process_segment_moderate_anomaly_needs_split(default_config):
    """Test process_segment with a moderate anomaly (long duration) requiring split."""
    # Create a segment longer than max_segment_duration (15s) but otherwise okayish
    words_in: List[WordToken] = [
        {"word": "Sentence", "start": 0.0, "end": 0.5},
        {"word": "one.", "start": 0.6, "end": 1.0},    # Sentence end, potential split point
        {"word": "And", "start": 1.5, "end": 2.0},     # Small gap
        {"word": "sentence", "start": 8.0, "end": 8.5}, # Moderate gap
        {"word": "two.", "start": 8.6, "end": 9.0},     # Sentence end, potential split point
        {"word": "Finally", "start": 15.5, "end": 16.0}, # Moderate gap
        {"word": "three.", "start": 16.1, "end": 16.5}, # Total duration 16.5s
    ]
    segment_in = create_segment_for_splitting(0.0, 16.5, words_in) # Use helper
    processed_segments = process_segment(segment_in, default_config)

    # Expect segment to be split due to duration > 15s
    assert len(processed_segments) > 1
    # Check continuity (end of seg N approx start of seg N+1)
    for i in range(len(processed_segments) - 1):
        assert processed_segments[i]["end"] <= processed_segments[i+1]["start"] + default_config.min_gap_duration * 2 # Allow small tolerance

    # Check total word count remains the same
    original_word_count = len(words_in)
    processed_word_count = sum(len(s.get("words",[])) for s in processed_segments)
    assert processed_word_count == original_word_count

def test_process_segment_severe_anomaly_distributes(default_config):
    """Test process_segment with severe anomaly triggering even distribution."""
    words_in: List[WordToken] = [
        {"word": "Very", "start": 1.0, "end": 1.5},
        {"word": "long----------word", "start": 1.6, "end": 10.0}, # Duration 8.4s
        {"word": "end", "start": 10.1, "end": 10.5},
    ]
    segment_in: Segment = {"start": 1.0, "end": 10.5, "text": "...", "words": words_in}
    processed_segments = process_segment(segment_in, default_config)

    assert len(processed_segments) == 1
    out_segment = processed_segments[0]
    out_words = out_segment["words"]

    assert out_segment["start"] == pytest.approx(1.0)
    assert out_segment["end"] == pytest.approx(10.5)
    last_end = out_segment["start"]
    found_long_word = False
    original_long_duration = 10.0 - 1.6 # 8.4s

    for i, word in enumerate(out_words):
         assert word["start"] >= last_end - 1e-6
         duration = word["end"] - word["start"]
         assert duration >= default_config.min_word_duration - 1e-9

         # Check that the duration is now significantly less than the original extreme duration.
         # It might still be > max_word_duration (2s) or even suspicious_max_word_duration (5s)
         # if the segment is long and the word has many characters proportionally.
         # The key is that it's *plausible* relative to distribution.
         if word["word"] == "long----------word":
              assert duration < original_long_duration # Ensure it's shorter than the original 8.4s
              # Optionally, add a looser upper bound check if desired, e.g.:
              # assert duration < 7.0 # Check it's less than the ~6.8s calculated duration + tolerance
              found_long_word = True

         last_end = word["end"]

    assert found_long_word # Make sure we actually checked the long word
    # Check original timestamps are gone
    assert out_words[1]["start"] != 1.6
    assert out_words[1]["end"] != 10.0

def test_process_segment_returns_empty_on_invalid(default_config):
    """Test that processing an invalid segment (e.g., no valid words after fix) returns empty list."""
    segment_in: Segment = {
        "start": 1.0, "end": 5.0, "text": "...",
        "words": [
            {"word": None, "start": 1.0, "end": 2.0},
            {"word": " ", "start": 2.0, "end": 3.0},
        ]
    }
    processed_segments = process_segment(segment_in, default_config)
    assert processed_segments == []

def test_calculate_stats_only_punctuation(default_config):
    """Test stats with only punctuation words (should likely be invalid or have odd stats)."""
    segment: Segment = {
        "start": 5.0, "end": 6.0, "text": ". , !",
        "words": [
            {"word": ".", "start": 5.1, "end": 5.2},
            {"word": ",", "start": 5.3, "end": 5.4},
            {"word": "!", "start": 5.5, "end": 5.6}
        ]
    }
    stats = calculate_segment_statistics(segment, default_config)
    assert stats["valid"] is True # It has >= 2 timed words
    assert stats["total_words"] == 3
    assert stats["word_count_with_timing"] == 3
    assert stats["avg_word_duration"] == pytest.approx(0.1)
    assert stats["max_word_duration"] == pytest.approx(0.1)
    # Gaps = [0.1, 0.1]
    assert stats["avg_gap"] == pytest.approx(0.1)
    assert stats["max_gap"] == pytest.approx(0.1)
    # Density will be low: (0.1*3) / (6.0-5.0) = 0.3 / 1.0 = 0.3
    assert stats["density"] == pytest.approx(0.3)
    # NOTE: Depending on thresholds, this might trigger a 'low_density' moderate anomaly later.

def test_calculate_stats_tiny_gaps(default_config):
    """Test stats when all gaps are smaller than min_gap_duration."""
    segment: Segment = {
        "start": 1.0, "end": 2.0, "text": "word word word",
        "words": [
            {"word": "word1", "start": 1.1, "end": 1.4}, # dur=0.3
            {"word": "word2", "start": 1.405, "end": 1.7}, # dur=0.295, gap=0.005 (< min_gap=0.02)
            {"word": "word3", "start": 1.710, "end": 1.9}  # dur=0.190, gap=0.010 (< min_gap=0.02)
        ]
    }
    stats = calculate_segment_statistics(segment, default_config)
    assert stats["valid"] is True
    assert stats["word_count_with_timing"] == 3
    # Check gap stats - word_gaps list should be empty as all were < min_gap_duration
    assert stats["avg_gap"] == 0.0
    assert stats["max_gap"] == 0.0
    assert stats["total_gap_time"] == 0.0
    assert stats.get("gap_stddev", 0.0) == 0.0 # Check default or explicit 0

def test_calculate_stats_overlapping_words(default_config):
    """Test stats calculation with overlapping words."""
    # Current _extract_timing_data filters words where end <= start,
    # but doesn't explicitly handle overlaps where word2 starts before word1 ends.
    # It calculates gaps based on next_word.start - current_word.end, so overlaps result
    # in negative gaps which are currently ignored by `if gap >= config.min_gap_duration`.
    segment: Segment = {
        "start": 3.0, "end": 5.0, "text": "overlap test",
        "words": [
            {"word": "overlap", "start": 3.1, "end": 4.0}, # duration=0.9
            {"word": "lap", "start": 3.8, "end": 4.5},     # duration=0.7, starts before prev ends, gap = 3.8-4.0 = -0.2 (ignored)
            {"word": "test", "start": 4.6, "end": 4.9}      # duration=0.3, gap = 4.6-4.5 = 0.1 (valid)
        ]
    }
    stats = calculate_segment_statistics(segment, default_config)
    assert stats["valid"] is True
    assert stats["word_count_with_timing"] == 3
    # Check durations are calculated correctly even with overlap
    assert stats["avg_word_duration"] == pytest.approx(np.mean([0.9, 0.7, 0.3]))
    # Check gap stats - only the valid gap (0.1s) should be included
    assert stats["total_gap_time"] == pytest.approx(0.1)
    assert stats["max_gap"] == pytest.approx(0.1)
    assert stats["avg_gap"] == pytest.approx(0.1)
    assert stats.get("gap_stddev", -1.0) == 0.0 # Stddev is 0 for a single gap data point
    
def test_detect_anomalies_moderate_gap_stddev(default_config):
    """Test detection of moderate anomaly due to high gap standard deviation."""
    # Create stats with high gap variance but other values normal
    # Need gap_stddev > suspicious_gap_stddev (2.0)
    stats = create_test_stats({"gap_stddev": 2.5})
    is_anom, severity, reasons = detect_anomalies(stats, default_config)
    assert is_anom is True
    assert severity == SeverityLevel.MODERATE
    assert "high_gap_variance:2.50" in reasons
    
def test_process_segment_moderate_density_and_long(default_config):
    """Test segment with low density AND long duration - expect split."""
    # Low density often happens with long silences represented as gaps
    words_in: List[WordToken] = [
        {"word": "Start", "start": 0.0, "end": 1.0},
        # Very large gap -> low density for the segment duration
        {"word": "End.", "start": 16.0, "end": 17.0}, # Total duration 17s (>15s), max gap 15s (severe)
    ]
    # Recalculate: Let's make density low without a severe gap
    words_in = [
        {"word": "Start", "start": 0.0, "end": 0.5}, # 0.5s word time
        {"word": "Middle", "start": 8.0, "end": 8.5}, # 0.5s word time
        {"word": "End.", "start": 16.0, "end": 16.5}, # 0.5s word time
    ]
    # Total word time = 1.5s. Segment duration = 16.5s. Density = 1.5/16.5 = ~0.09 (<0.3 threshold)
    # Segment duration > 15s. Max gap = 7.5s (severe)
    # Okay, this setup triggers severe due to gap. Let's adjust again.

    words_in = [
        {"word": "Start", "start": 0.0, "end": 0.5}, # 0.5s
        {"word": "Word2", "start": 2.0, "end": 2.5}, # 0.5s, gap 1.5s
        {"word": "Word3", "start": 5.0, "end": 5.5}, # 0.5s, gap 2.5s (moderate)
        {"word": "Word4", "start": 8.0, "end": 8.5}, # 0.5s, gap 2.5s (moderate)
        {"word": "Word5", "start": 11.0, "end": 11.5},# 0.5s, gap 2.5s (moderate)
        {"word": "End.", "start": 16.0, "end": 16.5} # 0.5s, gap 4.5s (moderate). Total dur 16.5s. Total word time 3.0s
    ]
    # Density = 3.0 / 16.5 = ~0.18 (<0.3 threshold). Max gap = 4.5s (<5s hard gap). Duration > 15s.
    # Should trigger MODERATE due to density AND length. Should split.
    segment_in = create_segment_for_splitting(0.0, 16.5, words_in)
    processed_segments = process_segment(segment_in, default_config)

    # Expect split because original segment was too long AND had low density
    assert len(processed_segments) > 1
    # Check word count preserved
    original_word_count = len(words_in)
    processed_word_count = sum(len(s.get("words",[])) for s in processed_segments)
    assert processed_word_count == original_word_count
    # Check individual segment durations are now likely reasonable
    for seg in processed_segments:
        assert seg["end"] - seg["start"] <= default_config.max_segment_duration + 0.1 # Allow small tolerance

# --- final review F6: a segment with text but no word timings is kept as it is ---

@pytest.mark.parametrize("words", [[], None])
def test_process_segment_keeps_text_without_words(default_config, words):
    seg = {"start": 3.0, "end": 4.5, "text": " bonjour à tous "}
    if words is not None:
        seg["words"] = words
    assert process_segment(seg, default_config) == [
        {"start": 3.0, "end": 4.5, "text": "bonjour à tous", "words": []}]


def test_process_segment_wordless_and_textless_is_dropped(default_config):
    assert process_segment({"start": 1.0, "end": 2.0, "text": "  ", "words": []}, default_config) == []


def test_text_only_segment_is_never_merged(default_config):
    with_words = create_segment(start=1.0, end=2.0, text="Segment one")
    text_only = Segment(start=2.1, end=3.0, text="sans mots", words=[])
    assert should_merge_segments(with_words, text_only, default_config) is False
    assert should_merge_segments(text_only, with_words, default_config) is False

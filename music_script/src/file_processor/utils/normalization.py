import re
import unicodedata
from functools import reduce
from typing import List, Tuple

def normalize_track_name(track_name: str) -> str:
    """Normalize track name by removing common prefixes and suffixes."""
    track_name = remove_track_numbers(track_name)
    track_name = remove_disc_numbers(track_name)
    track_name = remove_common_suffixes(track_name)
    return track_name.lower()

def remove_track_numbers(track_name: str) -> str:
    """Remove track numbers from the beginning of the name."""
    track_name = re.sub(r'^\d+\s*[-–—]\s*', '', track_name)
    track_name = re.sub(r'^\d+\.\s*', '', track_name)
    return track_name

def remove_disc_numbers(track_name: str) -> str:
    """Remove disc/CD numbers from the name."""
    track_name = re.sub(r'\(disc\s*\d+\)', '', track_name, flags=re.IGNORECASE)
    track_name = re.sub(r'\(cd\s*\d+\)', '', track_name, flags=re.IGNORECASE)
    return track_name

def remove_common_suffixes(track_name: str) -> str:
    """Remove common suffixes like (live), (remastered), etc."""
    suffixes: List[Tuple[str, str]] = [
        (r'\s*\(live\)$', ''),
        (r'\s*\(remastered\)$', ''),
        (r'\s*\(remaster\)$', '')
    ]
    return reduce(lambda s, pattern_replacement: re.sub(
        pattern_replacement[0], pattern_replacement[1], s, flags=re.IGNORECASE
    ), suffixes, track_name)

def normalize_string(s: str) -> str:
    """Normalize string by removing special characters and common words."""
    s = convert_to_ascii(s)
    s = remove_special_chars(s)
    s = remove_common_words(s)
    return s

def convert_to_ascii(s: str) -> str:
    """Convert string to ASCII, removing accents and special characters."""
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')

def remove_special_chars(s: str) -> str:
    """Remove special characters from string."""
    return re.sub(r'[^a-z0-9\s]', '', s.lower())

def remove_common_words(s: str) -> str:
    """Remove common words that might differ between versions."""
    return re.sub(r'\b(disc|cd|disc\s*\d+|cd\s*\d+)\b', '', s) 
"""YuE2 prompt helper: tag suggestions, prompt tidying and the instrumental recipe.

Grounded in YuE2's own guidance (third_party/YuE docs/generation.md and the
yue2-music skill):

* ``style`` is a comma-separated description of **language, genre, vocal
  character, instruments and tempo**, e.g. "English, warm piano pop,
  expressive female voice, acoustic piano, rounded bass and light drums,
  lyrical memorable melody, unhurried phrasing, 88 BPM".
* ``lyrics`` hold the words with **section tags** such as ``[Verse]`` and
  ``[Chorus]``; the score's own section labels are intro, verse, pre-chorus,
  chorus, bridge, interlude, outro and instrumental.
* Instrumentals are not asked for with an "[instrumental]" lyric. Upstream's
  recipe starts the style with "Instrumental", ends it with "no vocals, no
  singing, no choir, no spoken words", plans the score with placeholder
  section tags, then moves the sung melody onto an instrument before the
  audio is generated (``to_instrumental_score``), so nothing is left to sing.

Everything here is plain, offline Python: no model or network calls.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Instrumental recipe (upstream: skills/yue2-music/instrumental/scripts)
# ---------------------------------------------------------------------------
NO_VOCAL_CONDITIONS = ("no vocals", "no singing", "no choir", "no spoken words")
# Planning lyrics upstream uses when YuE2 writes an instrumental score.
INSTRUMENTAL_PLAN_LYRICS = "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n"


def is_instrumental(lyrics: str) -> bool:
    """Empty lyrics or a lone "[instrumental]"-style tag mean no singing."""
    text = (lyrics or "").strip().lower()
    return not text or (text.startswith("[") and text.endswith("]") and "instrumental" in text)


def instrumental_style(style: str) -> str:
    """Upstream's instrumental style wording around the user's description."""
    # A stray "instrumental" tag mid-prompt moves to the front instead of repeating.
    tags = [tag for tag in split_tags(style) if tag.lower() != "instrumental"]
    style = ", ".join(tags) or "Expressive instrumental music"
    if not re.match(r"^instrumental\b", style, re.I):
        style = "Instrumental, " + style
    for condition in NO_VOCAL_CONDITIONS:
        if condition not in style.lower():
            style += ", " + condition
    return style + "."


def strip_instrumental_style(style: str) -> str:
    """Undo instrumental_style(): drop "Instrumental" and the no-vocal conditions."""
    drop = {"instrumental", *NO_VOCAL_CONDITIONS}
    return ", ".join(tag for tag in split_tags(style) if tag.lower() not in drop)


def section_tags(abc: str) -> str:
    """Upstream's lyric_tags(): the score's "% verse" labels as "[Verse]" tags."""
    labels = [line[2:] for line in (abc or "").splitlines() if line.startswith("% ")]
    return "\n\n".join("[" + label.title() + "]" for label in labels) + ("\n" if labels else "")


def trim_truncated_score(abc: str) -> str:
    """Cut a score that hit the planner's token cap back to its last whole group.

    YuE2 plans a full-length score whatever the requested duration, so long
    plans can stop at the 4096-token cap mid-line, which upstream's converter
    rejects. Native scores alternate "V: Vocal" / "V: Ins" music lines; this
    keeps everything up to the last complete pair.
    """
    lines = (abc or "").rstrip("\n").splitlines()
    while lines and not (
        len(lines) >= 2 and lines[-2].strip().startswith("V: Ins") and lines[-1].rstrip().endswith("|")
    ):
        lines.pop()
    return "\n".join(lines) + "\n" if lines else abc


def drop_intro(abc: str) -> str:
    """Remove a leading "% intro" section from a score, so the song starts in its body.

    Only when another section follows: some plans label the whole song
    "intro", and dropping that would leave nothing to play.
    """
    lines = (abc or "").splitlines()
    labels = [i for i, line in enumerate(lines) if line.startswith("% ")]
    if len(labels) < 2 or not lines[labels[0]][2:].strip().lower().startswith("intro"):
        return abc
    body = next((i for i in labels[1:] if not lines[i][2:].strip().lower().startswith("intro")), None)
    if body is None:
        return abc
    return "\n".join(lines[:labels[0]] + lines[body:]) + "\n"


def drop_outro(abc: str) -> str:
    """Remove a trailing "% outro" section, so the song doesn't wind down at its end.

    Only when another section comes before it (a score labelled "outro"
    throughout is left alone).
    """
    lines = (abc or "").rstrip("\n").splitlines()
    labels = [i for i, line in enumerate(lines) if line.startswith("% ")]
    if len(labels) < 2 or not lines[labels[-1]][2:].strip().lower().startswith("outro"):
        return abc
    start = labels[-1]
    # Consecutive outro labels at the end all go.
    for i in reversed(labels[:-1]):
        if not lines[i][2:].strip().lower().startswith("outro"):
            break
        start = i
    if start == labels[0]:
        return abc
    return "\n".join(lines[:start]) + "\n"


def outro_start_seconds(abc: str) -> float:
    """Where a trailing outro section starts, in score seconds (0 when there's none)."""
    shorter = drop_outro(abc)
    return score_seconds(shorter) if shorter != abc else 0.0


def _drop_lyric_block(lyrics: str, name: str) -> str:
    out, skipping = [], False
    for line in (lyrics or "").splitlines():
        tag = re.fullmatch(r"\s*\[([^\]]+)\]\s*", line)
        if tag:
            skipping = tag.group(1).strip().lower().startswith(name)
        if not skipping:
            out.append(line)
    return "\n".join(out).strip() + "\n" if out else ""


def drop_intro_lyrics(lyrics: str) -> str:
    """Remove an [Intro] block (the tag and its lines) from lyrics."""
    return _drop_lyric_block(lyrics, "intro")


def drop_outro_lyrics(lyrics: str) -> str:
    """Remove an [Outro] block (the tag and its lines) from lyrics."""
    return _drop_lyric_block(lyrics, "outro")


def _group_ends(abc: str) -> List[Tuple[int, float]]:
    """(line index, seconds covered so far) after each complete Vocal/Ins group.

    Uses the header tempo (Q:) and meter (M:); empty when either is missing.
    """
    tempo = re.search(r"^Q:\s*1/4\s*=\s*(\d+)", abc or "", re.M)
    meter = re.search(r"^M:\s*(\d+)/(\d+)", abc or "", re.M)
    if not tempo or not meter:
        return []
    bar_seconds = int(meter.group(1)) * 4 / int(meter.group(2)) * 60 / int(tempo.group(1))
    lines = abc.rstrip("\n").splitlines()
    ends, covered = [], 0.0
    for index, line in enumerate(lines):
        if index and lines[index - 1].strip().startswith("V: Ins") and line.rstrip().endswith("|"):
            covered += len(re.findall(r"\|\]|\|\||\|", line)) * bar_seconds
            ends.append((index, covered))
    return ends


def score_seconds(abc: str) -> float:
    """How long a score plays, in seconds (0 when it can't be measured)."""
    ends = _group_ends(abc)
    return ends[-1][1] if ends else 0.0


def trim_score_to_seconds(abc: str, seconds: float) -> str:
    """Keep only as many whole groups as the song will actually play.

    The planner writes a full-length score whatever duration is requested
    (361 s of score for a 210 s song was seen), so the audio stage follows a
    plan longer than the song. Groups are kept until the score covers
    ``seconds``.
    """
    for index, covered in _group_ends(abc):
        if covered >= seconds:
            lines = abc.rstrip("\n").splitlines()
            return "\n".join(lines[:index + 1]) + "\n"
    return abc


def to_instrumental_score(abc: str, *, keep_chords: bool = True) -> str:
    """Move the score's vocal melody onto the instrumental voice (upstream code).

    Raises ValueError when the score can't be converted exactly (upstream
    refuses rather than guessing); callers fall back to the original score.
    """
    scripts = PROJECT_DIR / "third_party" / "YuE" / "skills" / "yue2-music" / "instrumental" / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from instrumentalize import convert_score  # type: ignore
    finally:
        sys.path.remove(str(scripts))
    converted, _check = convert_score(abc, keep_chords=keep_chords)
    return converted


# ---------------------------------------------------------------------------
# Tag vocabulary
# ---------------------------------------------------------------------------
LANGUAGES = ["English", "Japanese", "Korean", "Mandarin", "Cantonese", "Spanish", "French",
             "German", "Portuguese", "Italian", "Russian"]

# Genre profiles: what tends to belong with a genre in a YuE2 style line.
# bpm is a typical (low, high) range; the midpoint is suggested.
GENRES: Dict[str, Dict] = {
    "pop": dict(aliases=["pop"], bpm=(96, 120), vocal=["bright female vocal", "smooth male vocal"],
                instruments=["synth bass", "punchy drums", "piano"], mood=["catchy hook", "uplifting"],
                production=["polished modern mix"]),
    "rock": dict(aliases=["rock"], bpm=(110, 140), vocal=["gritty male vocal", "powerful female vocal"],
                 instruments=["distorted electric guitar", "bass guitar", "live drums"],
                 mood=["energetic", "anthemic"], production=["wide stereo guitars"]),
    "metal": dict(aliases=["metal"], bpm=(120, 180), vocal=["harsh screamed vocal", "soaring clean vocal"],
                  instruments=["down-tuned guitars", "double kick drums", "heavy bass"],
                  mood=["aggressive", "dark"], production=["tight heavy mix"]),
    "punk": dict(aliases=["punk"], bpm=(150, 190), vocal=["shouted male vocal", "snotty vocal"],
                 instruments=["fast power chords", "driving bass", "loud drums"], mood=["rebellious", "raw"],
                 production=["lo-fi garage sound"]),
    "hip hop": dict(aliases=["hip hop", "hip-hop", "hiphop", "rap", "boom bap"], bpm=(85, 98),
                    vocal=["confident male rap", "laid-back female rap"],
                    instruments=["boom bap drums", "sampled piano", "deep bass"], mood=["head-nodding groove"],
                    production=["vinyl samples"]),
    "trap": dict(aliases=["trap"], bpm=(130, 150), vocal=["autotuned male vocal", "melodic rap"],
                 instruments=["808 bass", "rolling hi-hats", "dark synth pads"], mood=["menacing", "hypnotic"],
                 production=["heavy sub bass"]),
    "lo-fi": dict(aliases=["lo-fi", "lofi", "lo fi"], bpm=(70, 88), vocal=["soft hummed vocal"],
                  instruments=["dusty piano", "mellow electric piano", "lazy swung drums"],
                  mood=["relaxed", "nostalgic"], production=["vinyl crackle", "tape warmth"]),
    "jazz": dict(aliases=["jazz", "swing", "bebop"], bpm=(90, 140), vocal=["smoky female jazz vocal",
                 "crooning male vocal"], instruments=["upright bass", "brushed drums", "piano", "tenor saxophone"],
                 mood=["sophisticated", "late-night"], production=["live room ambience"]),
    "bossa nova": dict(aliases=["bossa nova", "bossa"], bpm=(120, 140), vocal=["breathy Portuguese vocal"],
                       instruments=["nylon-string guitar", "soft brushed percussion", "flute"],
                       mood=["breezy", "romantic"], production=["warm vintage recording"]),
    "lounge": dict(aliases=["lounge", "easy listening", "muzak", "elevator music"], bpm=(90, 115),
                   vocal=["velvety vocal"], instruments=["vibraphone", "electric piano", "soft strings"],
                   mood=["smooth", "carefree"], production=["vintage hi-fi"]),
    "blues": dict(aliases=["blues"], bpm=(70, 100), vocal=["soulful raspy vocal"],
                  instruments=["bluesy electric guitar", "harmonica", "walking bass"], mood=["melancholic", "gritty"],
                  production=["live band feel"]),
    "funk": dict(aliases=["funk", "funky"], bpm=(100, 118), vocal=["groovy male vocal", "sassy female vocal"],
                 instruments=["slap bass", "wah guitar", "tight drums", "horn section"], mood=["groovy", "upbeat"],
                 production=["dry punchy mix"]),
    "soul": dict(aliases=["soul", "motown", "r&b", "rnb", "neo soul"], bpm=(70, 100),
                 vocal=["silky soulful vocal", "rich female vocal"],
                 instruments=["Rhodes electric piano", "warm bass", "soft drums"], mood=["sensual", "heartfelt"],
                 production=["lush vocal harmonies"]),
    "gospel": dict(aliases=["gospel"], bpm=(70, 110), vocal=["powerful gospel lead", "choir"],
                   instruments=["Hammond organ", "piano", "hand claps"], mood=["uplifting", "spiritual"],
                   production=["big church reverb"]),
    "country": dict(aliases=["country", "americana"], bpm=(85, 115), vocal=["twangy male vocal",
                    "warm female vocal"], instruments=["acoustic guitar", "pedal steel guitar", "fiddle"],
                    mood=["heartfelt", "storytelling"], production=["Nashville polish"]),
    "folk": dict(aliases=["folk", "acoustic", "singer-songwriter"], bpm=(80, 110),
                 vocal=["intimate gentle vocal"], instruments=["fingerpicked acoustic guitar", "banjo", "soft strings"],
                 mood=["earnest", "warm"], production=["close-miked natural sound"]),
    "reggae": dict(aliases=["reggae", "dub", "ska"], bpm=(70, 95), vocal=["laid-back reggae vocal"],
                   instruments=["offbeat skank guitar", "deep dub bass", "one-drop drums"], mood=["sunny", "relaxed"],
                   production=["dub echo"]),
    "latin": dict(aliases=["latin", "salsa", "reggaeton", "cumbia"], bpm=(90, 105), vocal=["passionate Spanish vocal"],
                  instruments=["congas", "timbales", "brass stabs"], mood=["festive", "danceable"],
                  production=["bright tropical mix"]),
    "house": dict(aliases=["house", "edm", "dance", "deep house"], bpm=(120, 128), vocal=["diva vocal chops"],
                  instruments=["four-on-the-floor kick", "offbeat hi-hats", "plucky synths"],
                  mood=["euphoric", "danceable"], production=["club-ready mix"]),
    "techno": dict(aliases=["techno"], bpm=(125, 140), vocal=["sparse spoken vocal"],
                   instruments=["pounding kick", "acid 303 bassline", "metallic percussion"],
                   mood=["hypnotic", "dark"], production=["warehouse reverb"]),
    "drum and bass": dict(aliases=["drum and bass", "dnb", "jungle", "liquid"], bpm=(170, 176),
                          vocal=["soulful female vocal"], instruments=["breakbeats", "reese bass", "atmospheric pads"],
                          mood=["driving", "energetic"], production=["crisp fast drums"]),
    "dubstep": dict(aliases=["dubstep", "riddim"], bpm=(140, 150), vocal=["chopped vocal samples"],
                    instruments=["wobble bass", "half-time drums", "growling synths"], mood=["heavy", "aggressive"],
                    production=["massive drops"]),
    "synthwave": dict(aliases=["synthwave", "retrowave", "outrun", "80s"], bpm=(95, 118),
                      vocal=["dreamy reverb vocal"], instruments=["analog synth arpeggios", "gated reverb drums",
                      "warm synth bass"], mood=["nostalgic", "neon-lit"], production=["retro 80s production"]),
    "chiptune": dict(aliases=["chiptune", "8-bit", "8 bit", "16-bit", "video game"], bpm=(120, 150),
                     vocal=["robotic vocal"], instruments=["square wave lead", "triangle bass", "noise-channel drums"],
                     mood=["playful", "adventurous"], production=["retro console sound"]),
    "ambient": dict(aliases=["ambient", "drone", "atmospheric"], bpm=(60, 80), vocal=["ethereal wordless vocal"],
                    instruments=["evolving synth pads", "soft piano", "field recordings"], mood=["calm", "spacious"],
                    production=["long reverb tails"]),
    "cinematic": dict(aliases=["cinematic", "orchestral", "epic", "film score", "soundtrack"], bpm=(70, 110),
                      vocal=["choir"], instruments=["full orchestra", "taiko drums", "soaring strings", "french horns"],
                      mood=["epic", "dramatic"], production=["wide concert hall"]),
    "classical": dict(aliases=["classical", "baroque", "string quartet", "piano sonata"], bpm=(60, 120),
                      vocal=["operatic soprano"], instruments=["string quartet", "grand piano", "harpsichord"],
                      mood=["elegant", "expressive"], production=["natural concert hall acoustics"]),
    "horror": dict(aliases=["horror", "spooky", "creepy", "haunting", "eerie", "halloween"], bpm=(60, 90),
                   vocal=["whispered vocal", "eerie wordless choir"],
                   instruments=["detuned music box", "dissonant strings", "deep drones"],
                   mood=["unsettling", "ominous"], production=["cavernous reverb"]),
    "k-pop": dict(aliases=["k-pop", "kpop"], bpm=(100, 128), vocal=["bright Korean group vocals"],
                  instruments=["punchy synths", "trap hi-hats", "big drops"], mood=["catchy", "energetic"],
                  production=["glossy modern mix"]),
    "city pop": dict(aliases=["city pop", "j-pop", "jpop", "anime"], bpm=(100, 120), vocal=["breezy Japanese vocal"],
                     instruments=["funky bass", "bright electric piano", "chorus guitar"],
                     mood=["sunny", "nostalgic"], production=["80s Japanese polish"]),
    "disco": dict(aliases=["disco", "nu-disco"], bpm=(115, 125), vocal=["falsetto vocal"],
                  instruments=["octave bassline", "string stabs", "four-on-the-floor drums"],
                  mood=["glamorous", "danceable"], production=["shimmering mix"]),
}

# Genre-independent pools for deeper suggestion pages.
GENERAL_POOLS = {
    "instruments": ["warm analog pads", "plucked synth lead", "fretless bass", "string ensemble", "muted trumpet",
                    "glockenspiel", "tape-saturated drums", "sub bass", "Rhodes chords", "clean electric guitar",
                    "cello", "hand percussion", "choir pads", "theremin"],
    "mood": ["bittersweet", "triumphant", "mysterious", "playful", "brooding", "hopeful", "tense", "dreamlike",
             "cinematic build", "late-night"],
    "production": ["wide stereo image", "sidechain pumping", "lo-fi tape hiss", "crisp modern mix", "lush reverb",
                   "punchy compression", "vintage analog warmth", "spacious mix"],
}

GENRE_STARTERS = ["pop", "lo-fi hip hop", "synthwave", "jazz", "rock", "cinematic orchestral",
                  "drum and bass", "ambient"]

VOCAL_WORDS = ("vocal", "voice", "singer", "rap", "choir", "harmonies", "falsetto", "croon", "soprano", "whisper")
INSTRUMENT_WORDS = ("guitar", "bass", "drum", "piano", "synth", "string", "violin", "cello", "sax", "trumpet",
                    "horn", "flute", "organ", "rhodes", "808", "kick", "hi-hat", "percussion", "orchestra",
                    "harp", "banjo", "fiddle", "ukulele", "vibraphone", "marimba", "bell", "pad", "arpeggio",
                    "music box", "conga", "brass", "harmonica", "clap", "breakbeat", "harpsichord")
PRODUCTION_WORDS = ("mix", "reverb", "tape", "vinyl", "lo-fi", "distortion", "recording", "production", "warmth",
                    "crackle", "polish", "master", "stereo", "hi-fi", "echo", "compressed", "vhs", "sound")
MOOD_WORDS = ("happy", "sad", "dark", "melanchol", "uplifting", "energetic", "calm", "relax", "dreamy", "epic",
              "romantic", "nostalgic", "haunting", "eerie", "aggressive", "playful", "chill", "mellow", "groovy",
              "ominous", "euphoric", "unsettling", "soulless", "spooky", "creepy", "moody", "bright", "warm",
              "gentle", "tender", "lyrical", "catchy", "anthemic", "hypnotic")

BPM_RE = re.compile(r"\b(\d{2,3})\s*bpm\b", re.I)
SECTION_TAGS = ["[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Outro]"]


def split_tags(style: str) -> List[str]:
    """Comma/newline-separated tags, trimmed, without empties or trailing dots."""
    return [tag.strip().strip(".").strip() for tag in re.split(r"[,\n]", style or "") if tag.strip(" .")]


def _has(text: str, words) -> bool:
    return any(word in text for word in words)


def detect_genres(style: str) -> List[str]:
    text = f" {(style or '').lower()} "
    found = []
    for name, profile in GENRES.items():
        if any(re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text) for alias in profile["aliases"]):
            found.append(name)
    return found


def classify(tag: str) -> str:
    """Category of one style tag, for ordering: language, instrumental, genre ..."""
    low = tag.lower()
    if low in {lang.lower() for lang in LANGUAGES}:
        return "language"
    if low == "instrumental":
        return "instrumental"
    if low in NO_VOCAL_CONDITIONS:
        return "conditions"
    if BPM_RE.search(low) or "tempo" in low:
        return "tempo"
    if _has(low, VOCAL_WORDS):
        return "vocal"
    if detect_genres(low):
        return "genre"
    if _has(low, INSTRUMENT_WORDS):
        return "instruments"
    if _has(low, PRODUCTION_WORDS):
        return "production"
    if _has(low, MOOD_WORDS):
        return "mood"
    return "other"


# Order follows YuE2's own example prompts: language, genre, voice, instruments,
# character of the music, tempo last.
ORDER = ["instrumental", "language", "genre", "vocal", "instruments", "mood", "other", "production",
         "tempo", "conditions"]


# ---------------------------------------------------------------------------
# Suggestions + tidy
# ---------------------------------------------------------------------------
def _missing(candidates: List[str], style_low: str) -> List[str]:
    return [tag for tag in candidates if tag.lower() not in style_low]


def suggest_style_tags(style: str, lyrics: str, limit: int = 9, page: int = 0) -> List[str]:
    """Tags worth adding next, most useful first; ``page`` shows further ones.

    Page 0 fills the gaps YuE2's guidance cares about (genre, voice,
    instruments, tempo, language) from the profile of each genre present.
    Later pages draw on the rest of those profiles and related genres, and
    wrap around when they run out.
    """
    pool = _ranked_style_tags(style, lyrics, limit)
    extras = _extra_style_tags(style, lyrics, exclude={tag.lower() for tag in pool[:limit]})
    pool = pool[:limit] + extras
    if not pool:
        return []
    pages = max(1, -(-len(pool) // limit))
    start = (page % pages) * limit
    return pool[start:start + limit]


def _extra_style_tags(style: str, lyrics: str, exclude: set) -> List[str]:
    """Deeper suggestions for the refresh button: the rest of each profile."""
    style_low = (style or "").lower()
    genres = detect_genres(style) or ["pop", "lo-fi", "synthwave"]
    out: List[str] = []
    for name in genres:
        profile = GENRES[name]
        keys = ["instruments", "mood", "production"]
        if not is_instrumental(lyrics):
            keys.insert(0, "vocal")
        for key in keys:
            out += _missing(profile[key], style_low)
    for key in ("instruments", "mood", "production"):
        out += _missing(GENERAL_POOLS[key], style_low)
    # Neighbouring genres share a profile's instruments; offer a couple as blends.
    out += _missing([name for name in GENRES if name not in genres], style_low)[:6]
    seen, unique = set(exclude), []
    for tag in out:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique.append(tag)
    return unique


def _ranked_style_tags(style: str, lyrics: str, limit: int = 9) -> List[str]:
    """The most useful next tags (page 0 of suggest_style_tags)."""
    style_low = (style or "").lower()
    instrumental = is_instrumental(lyrics)
    genres = detect_genres(style)
    out: List[str] = []

    if instrumental:
        if not re.search(r"\binstrumental\b", style_low):
            out.append("Instrumental")
        if "no vocals" not in style_low:
            out.append("no vocals")
    elif not any(lang.lower() in style_low for lang in LANGUAGES) and re.search(r"[a-z]", lyrics or "", re.I):
        out.append("English")

    if not genres:
        out += _missing(GENRE_STARTERS, style_low)[:4]
        return out[:limit]

    profiles = [GENRES[name] for name in genres]
    if not instrumental and not _has(style_low, VOCAL_WORDS):
        out += _missing([profiles[0]["vocal"][0]], style_low)
    if not BPM_RE.search(style_low):
        # Respect tempo words already in the prompt ("slow tempo", "upbeat").
        if re.search(r"\b(slow|downtempo|ballad|lazy|sluggish)\b", style_low):
            bpm = min(profile["bpm"][0] for profile in profiles)
        elif re.search(r"\b(fast|upbeat|uptempo|frantic|driving)\b", style_low):
            bpm = max(profile["bpm"][1] for profile in profiles)
        else:
            low, high = profiles[0]["bpm"]
            bpm = (low + high) // 2 // 2 * 2
        out.append(f"{bpm} BPM")
    for key, per_genre in (("instruments", 3), ("mood", 1), ("production", 1)):
        for profile in profiles:
            out += _missing(profile[key], style_low)[:per_genre]
    seen, unique = set(), []
    for tag in out:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique.append(tag)
    return unique[:limit]


def suggest_section_tags(lyrics: str) -> List[str]:
    """Section tags the lyrics don't use yet (none for instrumentals)."""
    if is_instrumental(lyrics):
        return []
    present = (lyrics or "").lower()
    return [tag for tag in SECTION_TAGS if tag.lower() not in present]


def add_tag(style: str, tag: str) -> str:
    style = (style or "").strip().rstrip(",")
    return f"{style}, {tag}" if style else tag


def add_section(lyrics: str, tag: str) -> str:
    lyrics = (lyrics or "").rstrip()
    return f"{lyrics}\n\n{tag}\n" if lyrics else f"{tag}\n"


def tidy_style(style: str, lyrics: str) -> str:
    """Normalize a style prompt into YuE2's documented shape.

    Deduplicates, fixes BPM spelling ("120bpm" -> "120 BPM"), orders tags as
    in YuE2's example prompts, and applies the instrumental wording when the
    lyrics ask for an instrumental. Words are never invented or removed.
    """
    tags, seen = [], set()
    for tag in split_tags(style):
        tag = BPM_RE.sub(lambda m: f"{m.group(1)} BPM", tag)
        if tag.lower() == "instrumental":
            tag = "Instrumental"
        if tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append(tag)
    tags.sort(key=lambda tag: ORDER.index(classify(tag)))
    joined = ", ".join(tags)
    return instrumental_style(joined) if is_instrumental(lyrics) else joined


def tidy_lyrics(lyrics: str) -> str:
    """Title-case section tags and put one blank line between sections."""
    if is_instrumental(lyrics):
        return (lyrics or "").strip()
    text = re.sub(r"\[([^\]\n]+)\]", lambda m: "[" + m.group(1).strip().title() + "]", lyrics or "")
    lines = [line.rstrip() for line in text.strip().splitlines()]
    out: List[str] = []
    for line in lines:
        if line.startswith("[") and out and out[-1] != "":
            out.append("")
        if line == "" and out and out[-1] == "":
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n" if out else ""


def improve_prompt(style: str, lyrics: str) -> Tuple[str, str]:
    return tidy_style(style, lyrics), tidy_lyrics(lyrics)

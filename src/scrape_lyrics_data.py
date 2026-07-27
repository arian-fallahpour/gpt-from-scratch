#########################################################
# Scrape lyrics for Drake's solo songs from AZLyrics
#
# The song list comes from MusicBrainz (AZLyrics has no
# artist-credit metadata), filtered down to recordings
# credited to Drake alone, i.e. no featured artists.
#
# Output format matches data/instruction-data-basic.json,
# with empty "instruction" and "input" fields.
#
# Note: AZLyrics blocks fast automated access. Keep the
# delay high and the worker count low, and treat the
# copyrighted lyrics as local training data only.
#########################################################

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup, Comment

MUSICBRAINZ_API = "https://musicbrainz.org/ws/2"
AZLYRICS_URL = "https://www.azlyrics.com/lyrics/{artist}/{song}.html"
LRCLIB_API = "https://lrclib.net/api"
USER_AGENT = (
    "gpt-from-scratch-lyrics-scraper/1.0 "
    "(https://github.com/; personal research use)"
)
FEATURE_PATTERN = re.compile(
    r"\(\s*(feat|ft|featuring|with)\b|\bfeat\.|\bft\.|\bfeaturing\b",
    re.IGNORECASE,
)
# '[00:12.34]' / '[01:02]' offsets, possibly several stacked at the start of a line
LRC_TIMESTAMP = re.compile(r"^(?:\s*\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\])+")
# A line that is nothing but a bracketed tag: '[Verse 1: Drake]', '[Chorus]', '[Produced by 40]'.
# Square/curly brackets only, since a standalone '(...)' line is usually a backing vocal.
ANNOTATION_ONLY = re.compile(r"^[\[\{][^\]\}]*[\]\}][\s.,!?]*$")
NON_LYRIC_LINES = {"instrumental", "instrumental break", "lyrics", "musik", "music"}


#########################################################
# Song list (MusicBrainz)
#########################################################

def musicbrainz_get(endpoint, params, delay=1.1, max_retries=4):
    params = {**params, "fmt": "json"}

    for attempt in range(max_retries):
        time.sleep(delay)  # MusicBrainz allows ~1 request per second
        response = requests.get(
            f"{MUSICBRAINZ_API}/{endpoint}",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if response.status_code in (429, 503):
            time.sleep(delay * (3 ** (attempt + 1)))
            continue
        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"MusicBrainz kept rate-limiting /{endpoint}; rerun in a minute.")


def find_artist_id(artist_name):
    data = musicbrainz_get("artist", {"query": artist_name, "limit": 5})
    for artist in data.get("artists", []):
        if artist["name"].lower() == artist_name.lower():
            return artist["id"]
    raise SystemExit(f"No MusicBrainz artist found for {artist_name!r}")


def browse_all(artist_id, endpoint, plural, extra_params=None):
    """Page through a MusicBrainz browse request until every result is collected."""
    items, offset = [], 0

    while True:
        data = musicbrainz_get(
            endpoint,
            {**(extra_params or {}), "artist": artist_id, "limit": 100, "offset": offset},
        )
        page = data.get(plural, [])
        if not page:
            break

        items.extend(page)
        offset += len(page)

        total = data.get(f"{endpoint}-count")
        if total is not None and offset >= total:
            break

    return items


def fetch_songs(artist_id, artist_name, singles_only=False):
    """Return titles of recordings credited to the artist alone, most-recent-release first.

    Browsing recordings rather than release groups matters: the release-group list only
    covers songs released as singles, which for Drake is ~60 titles after deduping, so
    --limit could never reach a few hundred. Recordings also carry their own artist
    credits, so features are caught here instead of one extra API search per title.
    """
    recordings = browse_all(artist_id, "recording", "recordings", {"inc": "artist-credits"})
    print(f"  {len(recordings)} recordings on MusicBrainz", file=sys.stderr)

    solo = []
    for recording in recordings:
        if recording.get("video"):
            continue
        credits = recording.get("artist-credit", [])
        # More than one credited artist means a collaboration or a feature
        if len(credits) != 1:
            continue
        if credits[0]["artist"]["name"].lower() != artist_name.lower():
            continue
        if FEATURE_PATTERN.search(recording["title"]):
            continue
        solo.append((recording["title"], recording.get("first-release-date") or ""))

    titles = dedupe_by_date(solo)

    if singles_only:
        released = single_keys(artist_id)
        titles = [(t, d) for t, d in titles if dedupe_key(t) in released]

    return [title for title, _ in titles]


def single_keys(artist_id):
    """Dedupe keys of the artist's release groups of type Single, for --singles-only."""
    groups = browse_all(artist_id, "release-group", "release-groups", {"type": "single"})
    return {
        dedupe_key(group["title"])
        for group in groups
        if group.get("primary-type") == "Single" and not group.get("secondary-types")
    }


def dedupe_key(title):
    """Collapse re-releases and fan edits, e.g. 'Nonstop' and 'Nonstop (Ticklish reboot)'."""
    return slugify(re.sub(r"\s*\([^)]*\)", "", title)) or slugify(title)


def dedupe(titles):
    """Collapse re-releases and fan edits in a plain list of titles."""
    seen, unique = set(), []
    for title in titles:
        key = dedupe_key(title)
        if key and key not in seen:
            seen.add(key)
            unique.append(title)
    return unique


def dedupe_by_date(releases):
    """Collapse re-releases/fan edits (title, date) pairs, keeping each song's earliest
    known release date, then sort most-recent-first so --limit can mean 'N newest'."""
    earliest = {}
    order = []
    for title, date in releases:
        key = dedupe_key(title)
        if not key:
            continue
        if key not in earliest:
            order.append(key)
            earliest[key] = [title, date]
        elif date and (not earliest[key][1] or date < earliest[key][1]):
            earliest[key][1] = date

    deduped = [tuple(earliest[key]) for key in order]
    # Missing dates sort last, everything else newest-first
    return sorted(deduped, key=lambda item: item[1] or "0000-00-00", reverse=True)


#########################################################
# Lyrics (AZLyrics)
#########################################################

def slugify(text):
    """AZLyrics URLs are the lowercased title with everything non-alphanumeric removed."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def url_candidates(artist_name, title):
    """AZLyrics is inconsistent about parentheticals and '&', so try a few spellings."""
    variants = [
        title,
        re.sub(r"\s*\([^)]*\)", "", title),          # drop "(Remix)"-style suffixes
        title.replace("&", "and"),
        re.sub(r"\s*/.*$", "", title),               # "A / B" split singles
    ]
    slugs, seen = [], set()
    for variant in variants:
        slug = slugify(variant)
        if slug and slug not in seen:
            seen.add(slug)
            slugs.append(slug)

    artist_slug = slugify(artist_name)
    return [AZLYRICS_URL.format(artist=artist_slug, song=slug) for slug in slugs]


def fetch_page(url, session, delay, max_retries=3):
    """Fetch one AZLyrics page. Returns None on 404, raises if AZLyrics gates the request."""
    for attempt in range(max_retries):
        time.sleep(delay * random.uniform(0.8, 1.4))
        response = session.get(url, timeout=30)

        if response.status_code == 404:
            return None
        if response.status_code in (403, 429, 503):
            backoff = delay * (4 ** (attempt + 1))
            print(f"  blocked ({response.status_code}), backing off {backoff:.0f}s", file=sys.stderr)
            time.sleep(backoff)
            continue
        response.raise_for_status()

        if is_access_challenge(response.text):
            raise BlockedError(url)
        return response.text

    raise BlockedError(url)


BLOCKED_HELP = (
    "AZLyrics gates automated requests behind a captcha, which this script does not try "
    "to defeat. Lyrics fetched before the block are cached and written out. Rerun with "
    "--source lrclib to use an open lyrics API, or lower --workers and raise --delay."
)


class BlockedError(RuntimeError):
    def __init__(self, url):
        super().__init__(
            f"AZLyrics served its access-request challenge instead of {url}.\n{BLOCKED_HELP}"
        )


def is_access_challenge(html):
    """AZLyrics answers bots with a 200 captcha interstitial rather than an error status."""
    head = html[:2000].lower()
    return "request for access" in head or "/captcha/" in head


def parse_lyrics(html):
    """The lyrics live in an unlabelled <div> tagged by an AZLyrics usage comment."""
    soup = BeautifulSoup(html, "html.parser")

    for comment in soup.find_all(string=lambda node: isinstance(node, Comment)):
        if "Usage of azlyrics.com content" in comment:
            container = comment.find_parent("div")
            if container is not None:
                return clean_lyrics(container.get_text(separator="\n"))

    # Fallback: the longest class-less div inside the lyrics column
    column = soup.find("div", class_="col-xs-12 col-lg-8 text-center")
    if column is None:
        return None
    candidates = [d for d in column.find_all("div") if not d.get("class")]
    if not candidates:
        return None
    return clean_lyrics(max(candidates, key=lambda d: len(d.get_text())).get_text("\n"))


#########################################################
# Lyrics (LRCLIB fallback)
#########################################################

def lrclib_get(endpoint, title, artist_name, session, delay, max_retries=5):
    """One LRCLIB request, backing off when concurrent workers trip its rate limit."""
    for attempt in range(max_retries):
        time.sleep(delay * random.uniform(0.8, 1.2))
        response = session.get(
            f"{LRCLIB_API}/{endpoint}",
            params={"artist_name": artist_name, "track_name": title},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(delay * (2 ** (attempt + 1)) * random.uniform(0.8, 1.2))
            continue
        return response

    response.raise_for_status()
    return response


def fetch_lrclib(title, artist_name, session, delay):
    """LRCLIB is an open, key-less lyrics API, unlike AZLyrics it welcomes API clients."""
    response = lrclib_get("get", title, artist_name, session, delay)

    if response.status_code == 404:
        # Exact match failed, fall back to the search index
        response = lrclib_get("search", title, artist_name, session, delay)
        response.raise_for_status()
        results = [r for r in response.json() if not r.get("instrumental")]
        if not results:
            return None
        record = results[0]
    else:
        response.raise_for_status()
        record = response.json()

    if record.get("instrumental"):
        return None
    plain = record.get("plainLyrics")
    return clean_lyrics(plain) if plain else None


def clean_lyrics(text):
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = strip_non_lyrics(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_non_lyrics(text):
    """Drop everything that isn't sung: LRC timestamps and bracketed annotations.

    LRCLIB plain lyrics occasionally keep '[00:12.34]' offsets, and both sources carry
    Genius-style structure tags such as '[Verse 1: Drake]', '[Chorus]' or '[Produced by 40]'.
    """
    lines = []

    for line in text.split("\n"):
        line = LRC_TIMESTAMP.sub("", line)
        line = line.strip()

        if ANNOTATION_ONLY.match(line):
            continue
        if line.lower() in NON_LYRIC_LINES:
            continue

        lines.append(line)

    return "\n".join(lines)


#########################################################
# Driver
#########################################################

def load_cache(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path, cache):
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def make_session(source):
    session = requests.Session()
    if source == "azlyrics":
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.azlyrics.com/",
        })
    else:
        session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_lyrics(title, artist_name, session, delay, source):
    if source == "azlyrics":
        for url in url_candidates(artist_name, title):
            html = fetch_page(url, session, delay)
            if html is None:
                continue
            lyrics = parse_lyrics(html)
            if lyrics:
                return lyrics
        return None

    return fetch_lrclib(title, artist_name, session, delay)


def scrape(titles, artist_name, delay, cache_path, source, workers):
    """Fetch every title's lyrics, `workers` songs at a time.

    Each worker keeps its own Session (requests.Session is not thread-safe) and sleeps
    `delay` before each of its own requests, so the overall request rate is roughly
    workers/delay per second. Results are collected by index so the output file stays in
    the input order regardless of which thread finishes first.
    """
    cache = load_cache(cache_path)
    lock = threading.Lock()
    local = threading.local()
    blocked = threading.Event()

    results = [None] * len(titles)
    done = [0]

    def report(title, status):
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(titles)}] {title} ... {status}", flush=True)

    def work(index, title):
        with lock:
            cached = cache.get(title)

        if cached is not None:
            # Re-clean: the cache may predate the current stripping rules
            results[index] = clean_lyrics(cached)
            report(title, f"cached ({len(results[index])} chars)")
            return

        if blocked.is_set():
            return  # another worker hit the captcha; don't keep hammering

        if not hasattr(local, "session"):
            local.session = make_session(source)

        try:
            lyrics = fetch_lyrics(title, artist_name, local.session, delay, source)
        except BlockedError:
            blocked.set()
            return
        except requests.RequestException as error:
            # One bad response shouldn't abandon the other 199 songs
            report(title, f"failed ({type(error).__name__})")
            return

        if not lyrics:
            report(title, f"not found on {source}")
            return

        results[index] = lyrics
        with lock:
            cache[title] = lyrics
            save_cache(cache_path, cache)
        report(title, f"scraped ({len(lyrics)} chars)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(work, range(len(titles)), titles):
            pass

    entries, missing, seen_lyrics = [], [], set()
    for title, lyrics in zip(titles, results):
        # Two title variants can resolve to the same lyrics page, so keep the text unique
        if not lyrics:
            missing.append(title)
        elif lyrics not in seen_lyrics:
            seen_lyrics.add(lyrics)
            entries.append({"instruction": "", "input": "", "output": lyrics})

    return entries, missing, blocked.is_set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artist", default="Drake")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "instruction-data-lyrics.json"),
    )
    parser.add_argument(
        "--cache",
        default=os.path.join(os.path.dirname(__file__), "..", "data", ".lyrics-cache.json"),
        help="Raw lyrics cache so reruns don't re-fetch songs already collected",
    )
    parser.add_argument(
        "--titles-file",
        help="Newline-separated song titles to use instead of the MusicBrainz song list",
    )
    parser.add_argument(
        "--source",
        choices=["azlyrics", "lrclib"],
        default="azlyrics",
        help="Lyrics source. AZLyrics captcha-gates bots; lrclib is an open API that does not.",
    )
    parser.add_argument(
        "--delay", type=float,
        help="Seconds each worker waits before its own requests to the lyrics source",
    )
    parser.add_argument(
        "--workers", type=int,
        help="Songs to scrape concurrently. AZLyrics blocks bursts, so keep it low there.",
    )
    parser.add_argument(
        "--singles-only", action="store_true",
        help="Restrict to songs released as singles (far fewer titles, ~60 for Drake)",
    )
    parser.add_argument(
        "--limit", type=int,
        help=(
            "Only scrape the N most recent songs (by first-release-date). "
            "With --titles-file, takes the first N lines as given instead."
        ),
    )
    parser.add_argument("--list-only", action="store_true", help="Print the song list and exit")
    args = parser.parse_args()

    if args.delay is None:
        args.delay = 6.0 if args.source == "azlyrics" else 1.0
    if args.workers is None:
        args.workers = 3 if args.source == "azlyrics" else 8

    if args.titles_file:
        with open(args.titles_file, "r", encoding="utf-8") as f:
            titles = dedupe([line.strip() for line in f if line.strip()])
    else:
        artist_id = find_artist_id(args.artist)
        titles = fetch_songs(artist_id, args.artist, args.singles_only)

    if args.limit:
        titles = titles[: args.limit]

    print(f"{len(titles)} solo songs for {args.artist}")
    if args.list_only:
        for title in titles:
            print(f"  {title}")
        return

    entries, missing, blocked = scrape(
        titles, args.artist, args.delay, args.cache, args.source, args.workers
    )

    # Write whatever was collected even when AZLyrics cut the run short
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)

    print(f"\nWrote {len(entries)} entries to {args.output}")
    if missing:
        print(f"{len(missing)} titles had no lyrics on {args.source}:", file=sys.stderr)
        for title in missing:
            print(f"  {title}", file=sys.stderr)
    if blocked:
        raise SystemExit(f"\n{BLOCKED_HELP}")


if __name__ == "__main__":
    main()

"""
Download example face images for football players from Wikipedia.

Usage:
    python download_faces.py

Reads players.json for the list of players, then downloads each player's
Wikipedia infobox image (a curated, clean portrait).
"""

import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PLAYERS_FILE = SCRIPT_DIR / "players.json"
EXAMPLES_DIR = SCRIPT_DIR / "examples"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "FaceExampleDownloader/1.0 (educational facial recognition project)"


def get_wikipedia_image_url(article_title: str) -> str | None:
    """Get the main infobox image URL from a Wikipedia article."""
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": article_title,
        "prop": "pageimages",
        "piprop": "thumbnail",
        "pithumbsize": "500",
        "format": "json",
    })
    url = f"{WIKIPEDIA_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {})
        if thumb.get("source"):
            return thumb["source"]
    return None


def download_image(url: str, dest: Path) -> bool:
    """Download an image from a URL to a local path."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            if len(content) < 5000:
                return False
            dest.write_bytes(content)
            return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def main():
    if not PLAYERS_FILE.exists():
        print(f"Error: {PLAYERS_FILE} not found")
        sys.exit(1)

    players = json.loads(PLAYERS_FILE.read_text())
    print(f"Found {len(players)} players\n")

    for player in players:
        name = player["name"]
        wikipedia_title = player.get("wikipedia_title", name)
        EXAMPLES_DIR.mkdir(exist_ok=True)
        dest = EXAMPLES_DIR / f"{name}.jpg"

        if dest.exists():
            print(f"  Already exists: {dest.name}, skipping")
            continue

        print(f"Downloading: {name}")

        image_url = get_wikipedia_image_url(wikipedia_title)
        if not image_url:
            print(f"  No Wikipedia image found")
            continue

        if download_image(image_url, dest):
            print(f"  Saved: {dest.name}")
        else:
            print(f"  Download failed")

        time.sleep(3)

    print("\nDone!")


if __name__ == "__main__":
    main()

"""
Step 1: Policy Scraper
Scrapes GDPR articles from gdpr-info.eu and ePrivacy Directive from EUR-Lex.
Saves raw .txt files and combined corpus.json.
"""

import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")


class PolicyScraper:
    """Scrapes GDPR and ePrivacy policy documents."""

    def __init__(self):
        os.makedirs(RAW_DIR, exist_ok=True)
        self.corpus = []
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    # ------------------------------------------------------------------
    # GDPR
    # ------------------------------------------------------------------
    def scrape_gdpr(self):
        """Scrape GDPR articles 1-99 from gdpr-info.eu."""
        print("[SCRAPER] Starting GDPR scrape …")
        base_url = "https://gdpr-info.eu/art-{}-gdpr/"
        for n in range(1, 100):
            url = base_url.format(n)
            try:
                resp = requests.get(url, headers=self.headers, timeout=30)
                if resp.status_code != 200:
                    print(f"  Art {n}: HTTP {resp.status_code}, skipping")
                    time.sleep(1)
                    continue

                soup = BeautifulSoup(resp.text, "lxml")
                # Main article content
                entry = soup.find("div", class_="entry-content")
                if not entry:
                    print(f"  Art {n}: no entry-content div, skipping")
                    time.sleep(1)
                    continue

                text = self._clean_text(entry.get_text(separator="\n"))
                if not text.strip():
                    time.sleep(1)
                    continue

                # Save raw file
                fname = f"gdpr_art_{n}.txt"
                with open(os.path.join(RAW_DIR, fname), "w", encoding="utf-8") as f:
                    f.write(text)

                self.corpus.append({
                    "id": f"GDPR_Art{n}",
                    "source": "GDPR",
                    "article": n,
                    "text": text,
                })
                print(f"  Art {n}: OK ({len(text)} chars)")

            except Exception as e:
                print(f"  Art {n}: ERROR — {e}")

            time.sleep(1)  # polite delay

        print(f"[SCRAPER] GDPR done — {len([c for c in self.corpus if c['source']=='GDPR'])} articles")

    # ------------------------------------------------------------------
    # ePrivacy
    # ------------------------------------------------------------------
    def scrape_eprivacy(self):
        """Scrape ePrivacy Directive from EUR-Lex consolidated text."""
        print("[SCRAPER] Starting ePrivacy scrape …")

        full_text = self._fetch_eprivacy_text()
        if not full_text:
            print("[SCRAPER] ePrivacy: could not fetch text from any source")
            return

        self._parse_eprivacy_articles(full_text)

    def _fetch_eprivacy_text(self) -> str | None:
        """Try multiple EUR-Lex URLs with retries; fall back to local file."""
        urls = [
            ("https://eur-lex.europa.eu/legal-content/EN/TXT/"
             "?uri=CELEX:02002L0058-20091219"),
            ("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/"
             "?uri=CELEX:02002L0058-20091219"),
            ("https://eur-lex.europa.eu/LexUriServ/LexUriServ.do?"
             "uri=CONSLEG:2002L0058:20091219:EN:HTML"),
        ]

        for url in urls:
            for attempt in range(5):
                try:
                    print(f"  Trying: {url[:70]}… (attempt {attempt+1})")
                    resp = requests.get(url, headers=self.headers, timeout=60)
                    if resp.status_code == 202:
                        # EUR-Lex is generating the document — wait & retry
                        wait = 3 * (attempt + 1)
                        print(f"  HTTP 202 — document generating, "
                              f"retrying in {wait}s …")
                        time.sleep(wait)
                        continue
                    if resp.status_code != 200:
                        print(f"  HTTP {resp.status_code}, trying next URL")
                        break
                    if len(resp.text.strip()) < 500:
                        print(f"  Response too short ({len(resp.text)} chars), "
                              f"retrying …")
                        time.sleep(3)
                        continue

                    soup = BeautifulSoup(resp.text, "lxml")
                    text = soup.get_text(separator="\n")
                    if len(text.strip()) > 500:
                        print(f"  Fetched {len(text)} chars from EUR-Lex")
                        return text
                except Exception as e:
                    print(f"  Attempt {attempt+1} error: {e}")
                    time.sleep(3)

        # Fallback: local file (placed manually or via browser save)
        fallback = os.path.join(RAW_DIR, "eprivacy_full.txt")
        if os.path.exists(fallback):
            print(f"  Using local fallback: {fallback}")
            with open(fallback, "r", encoding="utf-8") as f:
                return f.read()

        return None

    def _parse_eprivacy_articles(self, full_text: str):
        """Split ePrivacy full text into per-article entries."""
        # Split on "Article N" headings
        pattern = re.compile(
            r"(?=\bArticle\s+(\d+)\b)", re.IGNORECASE
        )
        parts = pattern.split(full_text)

        # parts comes as [preamble, num, text, num, text, …]
        i = 1
        count = 0
        while i < len(parts) - 1:
            art_num_str = parts[i].strip()
            art_text = parts[i + 1].strip()
            i += 2

            if not art_num_str.isdigit():
                continue
            art_num = int(art_num_str)

            # Re-attach heading
            cleaned = self._clean_text(f"Article {art_num}\n{art_text}")
            if not cleaned.strip():
                continue

            fname = f"eprivacy_art_{art_num}.txt"
            with open(os.path.join(RAW_DIR, fname), "w", encoding="utf-8") as f:
                f.write(cleaned)

            self.corpus.append({
                "id": f"EPRIVACY_Art{art_num}",
                "source": "ePrivacy",
                "article": art_num,
                "text": cleaned,
            })
            count += 1

        print(f"[SCRAPER] ePrivacy done — {count} articles")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip navigation, footnotes, excess whitespace."""
        # Remove common web artifacts
        for noise in [
            "Skip to content", "Table of Contents", "Menu",
            "Suitable Recitals", "Close", "Cookie", "GDPR",
        ]:
            text = text.replace(noise, "")

        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def save_corpus(self):
        """Save combined corpus to data/corpus.json."""
        path = os.path.join(DATA_DIR, "corpus.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, indent=2, ensure_ascii=False)
        print(f"[SCRAPER] Corpus saved → {path}  ({len(self.corpus)} docs)")

    def run(self):
        """Full scrape pipeline."""
        self.scrape_gdpr()
        self.scrape_eprivacy()
        self.save_corpus()


if __name__ == "__main__":
    PolicyScraper().run()

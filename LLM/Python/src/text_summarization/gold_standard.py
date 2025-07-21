import json
import time
import random
import pickle
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

from curl_cffi import requests as cf_requests
from bs4 import BeautifulSoup

from exploration_utilities import setup_logging, get_logger, get_project_root
from text_summarization.summarization_utilities import get_min_max_mean_std

OUT_DIR = get_project_root() / "Output" / "text_summarization_benchmark"
OUT_DIR.mkdir(exist_ok=True, parents=True)

RESOURCE_DIR = get_project_root() / "Resources"
RESOURCE_DIR.mkdir(exist_ok=True, parents=True)

CACHE_FILE = OUT_DIR / "gold_standard.pkl"

setup_logging(log_file=OUT_DIR / "gold_standard.log")
logger = get_logger(__name__)


@dataclass
class Publication:
    url: str
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    highlights: list[str] | None = None
    no_highlights: bool = False


@dataclass
class Journal:
    identifier: str
    limit_issues: int | None = None
    force_recheck: bool = False


class ElsevierGoldStandardRetriever:
    def __init__(self, sleep_min: int = 2, sleep_max: int = 10, cf_clearance: str = None):
        self.base_url = "https://www.sciencedirect.com"
        self.session = cf_requests.Session()

        if cf_clearance:
            self.session.cookies.set("cf-clearance", cf_clearance, domain=".sciencedirect.com", secure=True)
        else:
            logger.warning("cf-clearance cookie not set. Might result in earlier rate-limiting. To set the cookie, "
                           "load an Elsevier publication in the browser and go to Debugger > Storage > "
                           "Cookies > .sciencedirect.com > cf-clearance and copy/paste the value.")

        self._use_curl_cffi = cf_clearance is not None
        self._sleep_min = sleep_min
        self._sleep_max = sleep_max

        self.stop = False
        self.journals: defaultdict[str, list[Publication]] = defaultdict(list)

        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached publications from file"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'rb') as f:
                    self.journals = pickle.load(f)
                logger.info(f"Loaded cache with {sum(len(pubs) for pubs in self.journals.values())} publications "
                            f"from {CACHE_FILE}")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")

    def _save_cache(self) -> None:
        """Save publications to cache file"""
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self.journals, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _get_soup(self, url: str) -> BeautifulSoup | None:
        sleep_time = random.uniform(self._sleep_min, self._sleep_max)
        time.sleep(sleep_time)

        if self._use_curl_cffi:
            resp = self.session.get(url=url, timeout=10, impersonate="firefox135")
        else:
            resp = self.session.get(url=url, timeout=10)

        if resp.status_code in [403, 429, 503, 502, 451, 406]:
            logger.error(f"Status Code: {resp.status_code}. Might be rate-limited. Try again later.")
            self.stop = True
            return None

        resp.raise_for_status()
        return BeautifulSoup(resp.content, 'html.parser')

    def _get_latest_science_direct_issues(self, journal_identifier: str) -> list[str] | None:
        soup = self._get_soup(f"{self.base_url}/journal/{journal_identifier}/issues")

        if not soup:
            return None

        return [_["href"] for _ in soup.find_all("a", class_="js-issue-item-link")]

    def add_publication_urls_from_latest_issues(self, journal: Journal) -> None:
        if self.stop:
            return

        if journal.identifier in self.journals and not journal.force_recheck:
            logger.info(
                f"Journal {journal.identifier} already cached with "
                f"{len(self.journals[journal.identifier])} publications")
            return

        try:
            old_count = len(self.journals[journal.identifier])

            for idx, issue_url in enumerate(self._get_latest_science_direct_issues(
                    journal_identifier=journal.identifier)):

                added_per_issue = 0

                if journal.limit_issues and idx >= journal.limit_issues:
                    logger.info(f"Reached limit of {journal.limit_issues} issues for {journal.identifier}. "
                                f"Skipping remaining issues.")
                    break

                soup = self._get_soup(f"{self.base_url}{issue_url}")

                if not soup:
                    return

                hrefs = [
                    f"{self.base_url}{_['href']}"
                    for _ in soup.find_all("a", class_="article-content-title")
                ]

                for href in hrefs:
                    if not self._href_exists(href=href, journal_identifier=journal.identifier):
                        self.journals[journal.identifier].append(
                            Publication(url=href)
                        )
                        added_per_issue += 1

                current_total = len(self.journals[journal.identifier])
                total_added_this_run = current_total - old_count

                logger.info(f"Added {added_per_issue} publications for {journal.identifier}"
                            f" (total: {current_total}, added this run: {total_added_this_run})")
                self._save_cache()
        except Exception as e:
            logger.error(e)

    def _href_exists(self, href: str, journal_identifier: str) -> bool:
        for pub in self.journals[journal_identifier]:
            if pub.url == href:
                return True
        return False

    def fetch_highlights(self, limit_to_n_publications_per_journal: int | None = None) -> None:
        for journal_idx, (journal_identifier, publications) in enumerate(self.journals.items()):
            _pubs_with_highlights = 0

            for pub_idx, pub in enumerate(publications):

                if self.stop:
                    return

                if _pubs_with_highlights >= limit_to_n_publications_per_journal:
                    break

                _log = (f"J{journal_idx + 1}/{len(self.journals.keys())}"
                        f"|P{pub_idx}/{len(publications)}"
                        f"({_pubs_with_highlights}/{limit_to_n_publications_per_journal})"
                        f" | {pub.url}")

                if pub.title and pub.abstract and pub.highlights and pub.doi:
                    logger.info(f"{_log} | already cached, skipping")
                    _pubs_with_highlights += 1
                    continue

                if pub.no_highlights:
                    logger.info(f"{_log} | no highlights, skipping")
                    continue

                try:
                    soup = self._get_soup(pub.url)

                    if not soup:
                        continue

                    _doi = soup.find("a", class_="doi")
                    _title = soup.find("span", class_="title-text")
                    _abstract = soup.find("div", class_="abstract author")

                    if not _doi or not _title or not _abstract:
                        logger.warning(f"{_log} | Missing DOI, title or abstract")
                        pub.no_highlights = True
                        self._save_cache()
                        continue

                    pub.doi = _doi.text
                    pub.title = _title.text
                    pub.abstract = _abstract.find_next("div").text

                    highlights = soup.find("div", class_="abstract author-highlights")
                    if not highlights:
                        logger.warning(f"{_log} | No highlights found")
                        pub.no_highlights = True
                        self._save_cache()
                        continue

                    pub.highlights = [_.text for _ in highlights.find_all("span", class_="list-content")]

                    self._save_cache()
                    logger.info(f"{_log} | OK")
                    _pubs_with_highlights += 1
                except Exception as e:
                    logger.error(e)
                    continue

            self._save_cache()

    def export(self, out_file: str | Path) -> None:
        publications_data = []

        for journal_identifier, publications in self.journals.items():
            for pub in publications:
                if not pub.title or not pub.abstract or not pub.doi:
                    continue

                if not pub.highlights:
                    continue

                publications_data.append({
                    "title": pub.title,
                    "abstract": pub.abstract,
                    "id": pub.doi,
                    "summaries": [" ".join(pub.highlights)]
                })

        with open(out_file, mode="w", encoding="utf-8") as f:
            json.dump(publications_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote {len(publications_data)} publications with highlights to {out_file}")
        self.log_stats(publications_data=publications_data)

    @staticmethod
    def log_stats(publications_data: list[dict[str, str | list[str]]]) -> None:
        words = defaultdict(list)

        for p in publications_data:
            words["title"].append(len(re.findall(r'\w+', p["title"])))
            words["abstract"].append(len(re.findall(r'\w+', p["abstract"])))
            words["summary"].append(len(re.findall(r'\w+', p["summaries"][0])))

        logger.info("---")
        logger.info(f"Title word count: {get_min_max_mean_std(words['title'])}")
        logger.info(f"Abstract word count: {get_min_max_mean_std(words['abstract'])}")
        logger.info(f"Summary word count: {get_min_max_mean_std(words['summary'])}")
        logger.info("---")

    def clear_cache(self) -> None:
        del self.journals
        self.journals = defaultdict(list)

        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        logger.info(f"Cache cleared {CACHE_FILE}")


if __name__ == "__main__":
    retriever = ElsevierGoldStandardRetriever(sleep_min=2, sleep_max=10)

    journals = [
        Journal("drug-discovery-today", 10, False),
        Journal("journal-of-molecular-biology", 10, False),
        Journal("febs-letters", 10, False),
        Journal("journal-of-biotechnology", 10, False),
        Journal("gene", 10, False),
        Journal("genomics", 10, False),
        Journal("journal-of-proteomics", 10, False),
        Journal("journal-of-biomedical-informatics", 10, False),
        Journal("the-international-journal-of-biochemistry-and-cell-biology", 10, False),
        Journal("advanced-engineering-informatics", 10, False),
        Journal("drug-resistance-updates", 10, False),
        # Journal("drug-metabolism-and-pharmacokinetics", 10, False),
    ]

    for j in journals:
        retriever.add_publication_urls_from_latest_issues(journal=j)

    retriever.fetch_highlights(limit_to_n_publications_per_journal=10)
    retriever.export(out_file=RESOURCE_DIR / "text_summarization_goldstandard_data_elsevier.json")

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

        self.use_curl_cffi = cf_clearance is not None
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self.publications: defaultdict[str, list[Publication]] = defaultdict(list)
        self._load_cache()

        self.stop = False

    def _load_cache(self) -> None:
        """Load cached publications from file"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'rb') as f:
                    self.publications = pickle.load(f)
                logger.info(f"Loaded cache with {sum(len(pubs) for pubs in self.publications.values())} publications "
                            f"from {CACHE_FILE}")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self.publications = {}

    def _save_cache(self) -> None:
        """Save publications to cache file"""
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self.publications, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _get_soup(self, url: str) -> BeautifulSoup | None:
        sleep_time = random.uniform(self.sleep_min, self.sleep_max)
        # logger.info(f"Sleeping for {sleep_time:.2f} seconds before fetching {url}")
        time.sleep(sleep_time)
        if self.use_curl_cffi:
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

    def add_publication_urls_from_latest_issues(self, journal_identifier: str, limit_issues: int = 5) -> None:
        if self.stop:
            return

        if journal_identifier in self.publications:
            logger.info(
                f"Journal {journal_identifier} already cached with {len(self.publications[journal_identifier])} publications")
            return

        try:
            for idx, issue_url in enumerate(self._get_latest_science_direct_issues(journal_identifier=journal_identifier)):
                if limit_issues and idx >= limit_issues:
                    logger.info(f"Reached limit of {limit_issues} issues for {journal_identifier}. "
                                f"Skipping remaining issues.")
                    break

                soup = self._get_soup(f"{self.base_url}{issue_url}")

                if not soup:
                    return

                self.publications[journal_identifier].extend([
                    Publication(url=f"{self.base_url}{_['href']}")
                    for _ in soup.find_all("a", class_="article-content-title")
                ])

                logger.info(f"Added {len(self.publications[journal_identifier])} publications for {journal_identifier}")
                self._save_cache()
        except Exception as e:
            logger.error(e)

    def fetch_highlights(self, limit_to_n_publications_per_journal: int | None = None) -> None:
        for journal_idx, (journal_identifier, publications) in enumerate(self.publications.items()):
            _pubs_with_highlights = 0
            # _publications = publications[:limit_to_n_publications_per_journal]
            # for pub_idx, pub in enumerate(_publications):
            for pub_idx, pub in enumerate(publications):
                if self.stop:
                    return

                if _pubs_with_highlights >= limit_to_n_publications_per_journal:
                    break

                _log = (f"J{journal_idx + 1}/{len(self.publications.keys())}"
                        # f"|P{pub_idx + 1}/{len(_publications)}"
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

        for journal_identifier, publications in self.publications.items():
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
        self.publications = {}
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        logger.info(f"Cache cleared {CACHE_FILE}")


if __name__ == "__main__":
    retriever = ElsevierGoldStandardRetriever(sleep_min=2, sleep_max=10, cf_clearance="0Dbj4qZAMH3wE87n2KM8sGTgTogZnUZoOdXwKctQ3bE-1753089202-1.2.1.1-5e23c5n.lmgfrCkKk_juLvExmKkGQ7i1GpUjo.DUzGOFKHaLzqOmsj_NerMpthdbpD.mz9VGwa_CV8EyUlJeqRGGic.Za1xJNEhUoNGn_2ivSIueNuLz1oL2PIZJuPDiNk5SpEbjd1M2GsV7kx70tAwdu_9vikWk8OX0ulDl9xkaQqAI0H.202TCxBFztjaNoK9OZ5GH6fZM8sTus3YWUwZ2oWVvuFBg_yLSk_XMGS4")

    journal_identifiers = [
        "drug-discovery-today",
        "journal-of-molecular-biology",
        "febs-letters",
        "journal-of-biotechnology",
        "gene",
        "genomics",
        "journal-of-proteomics",
        "journal-of-biomedical-informatics",
        "the-international-journal-of-biochemistry-and-cell-biology",
        # "advanced-engineering-informatics",
        # "drug-resistance-updates",
        # "drug-metabolism-and-pharmacokinetics",
    ]

    for j_i in journal_identifiers:
        retriever.add_publication_urls_from_latest_issues(journal_identifier=j_i, limit_issues=10)

    retriever.fetch_highlights(limit_to_n_publications_per_journal=10)
    retriever.export(out_file=RESOURCE_DIR / "text_summarization_goldstandard_data_elsevier.json")

"""FairMatch Preprocessor Module.

Provides the Anonymizer class for PII redaction and section extraction,
mitigating demographic bias (names, gender, emails, phone numbers,
locations, elite institute pedigree) to support UN SDG 10 (Reduced Inequalities).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Technical terms, libraries, and frameworks that general NER models frequently misclassify as PERSON
TECHNICAL_SKILLS_WHITELIST: Set[str] = {
    "python", "pytorch", "tensorflow", "keras", "fastapi", "flask", "django",
    "docker", "kubernetes", "aws", "azure", "gcp", "git", "linux", "sql",
    "nosql", "mongodb", "postgresql", "mysql", "redis", "spacy", "scikit-learn",
    "sklearn", "pandas", "numpy", "scipy", "deep learning", "machine learning",
    "data science", "nlp", "computer vision", "algorithms", "data structures",
    "react", "angular", "vue", "javascript", "typescript", "c++", "c#", "java",
    "html", "css", "ci/cd", "microservices", "rest", "graphql", "devops",
    "kafka", "spark", "hadoop", "tableau", "power bi", "torch", "bert", "transformer",
}

# Resume field prefixes that NER models might greedily bundle with candidate names
LABEL_PREFIXES_REGEX = re.compile(
    r"^(?:email|phone|mobile|tel|location|address|linkedin|github|portfolio|website)[\s:]*",
    re.IGNORECASE,
)
# Resume field suffixes (with optional colons/spaces before AND after)
LABEL_SUFFIXES_REGEX = re.compile(
    r"[\s:]*(?:email|phone|mobile|tel|location|address|linkedin|github|portfolio|website)[\s:]*$",
    re.IGNORECASE,
)


class Anonymizer:
    """Anonymizer class for de-identifying resumes and extracting structured sections.

    Invariants:
      - Raw PII is never persisted to disk or emitted in logs.
      - Emails and phone numbers are redacted via regex with [EMAIL] and [PHONE].
      - PERSON entities are replaced with [CANDIDATE_NAME] using spaCy (en_core_web_sm).
      - GPE and LOC entities are replaced with [LOCATION] using spaCy (en_core_web_sm).
      - Cleans whitespace and parses core resume sections (Skills, Experience, Education).
    """

    # Section header patterns matching dedicated header lines (not descriptive bullet text)
    SECTION_PATTERNS = {
        "skills": re.compile(
            r"^(?:technical\s+skills|skills\s*(?:&|and)\s*tools|core\s+competencies|skills|technologies|proficiencies)(?:\s*:|\s*$)",
            re.IGNORECASE,
        ),
        "experience": re.compile(
            r"^(?:work\s+experience|professional\s+experience|employment\s+history|experience|internships|work\s+history)(?:\s*:|\s*$)",
            re.IGNORECASE,
        ),
        "education": re.compile(
            r"^(?:education|academic\s+background|educational\s+qualifications|academics|degrees)(?:\s*:|\s*$)",
            re.IGNORECASE,
        ),
    }

    # Elite institutions for pedigree de-biasing (UN SDG 10.2 & 10.3)
    ELITE_INSTITUTE_PATTERNS = [
        r"\bMIT\b|\bMassachusetts Institute of Technology\b",
        r"\bStanford(?:\s+University)?\b",
        r"\bHarvard(?:\s+University)?\b",
        r"\bPrinceton(?:\s+University)?\b",
        r"\bUC Berkeley\b|\bUniversity of California,?\s+Berkeley\b",
        r"\bCarnegie Mellon(?:\s+University)?\b|\bCMU\b",
        r"\bCaltech\b|\bCalifornia Institute of Technology\b",
        r"\bColumbia\s+University\b|\bUniversity of Columbia\b",
        r"\bYale(?:\s+University)?\b",
        r"\bOxford\s+University\b|\bUniversity of Oxford\b",
        r"\bCambridge\s+University\b|\bUniversity of Cambridge\b",
        r"\bIIT\b(?:\s+(?:Bombay|Delhi|Madras|Kanpur|Kharagpur|Roorkee|Guwahati|Hyderabad|Varanasi|BHU|Indore|Ropar|Bhubaneswar|Gandhinagar|Jodhpur|Patna|Mandi|Tirupati|Palakkad|Goa|Dharwad|Jammu|Bhilai))?",
        r"\bIndian Institute of Technology(?:\s+[A-Z][a-z]+)?\b",
        r"\bBITS(?:\s+Pilani)?\b",
        r"\bNIT\b(?:\s+(?:Trichy|Surathkal|Warangal|Rourkela|Calicut|Kurukshetra|Durgapur|Silchar|Hamirpur|Jamshedpur|Patna|Raipur|Srinagar|Agartala|Meghalaya|Nagpur|Allahabad))?",
        r"\bNational Institute of Technology(?:\s+[A-Z][a-z]+)?\b",
        r"\bIISc(?:\s+Bangalore)?\b|\bIndian Institute of Science\b",
    ]

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize regex rules and spaCy language pipeline."""
        self.spacy_model_name = spacy_model
        self._nlp = None
        self._nlp_attempted = False

        # Regex for Email Addresses -> [EMAIL]
        self._email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        )

        # Regex for Phone Numbers -> [PHONE]
        # Requires 7 to 14 digits with optional international/area codes.
        # Excludes standard 4-digit graduation/employment year ranges (e.g., 2020-2024).
        self._phone_pattern = re.compile(
            r"(?<!\d)(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]\d{4}\b(?!\d)|(?<!\d)(?:\+?\d{1,3}[\s.-]?)?[6-9]\d{9}\b"
        )

        # Regex for URLs and Social links
        self._url_pattern = re.compile(
            r"(?:https?://\S+|www\.\S+|linkedin\.com/(?:in/)?\S+|github\.com/\S+)",
            re.IGNORECASE,
        )

        # Regex for Gender Honorifics
        self._gender_honorifics = re.compile(
            r"\b(mr\.|mrs\.|ms\.|miss|dr\.)\b",
            re.IGNORECASE,
        )

        # Regex for Elite Institute Pedigree Bias
        combined_institutes = "|".join(self.ELITE_INSTITUTE_PATTERNS)
        self._institute_pattern = re.compile(combined_institutes, re.IGNORECASE)

    @property
    def nlp(self) -> Any:
        """Lazy loader for spaCy NER pipeline."""
        if self._nlp is None and not self._nlp_attempted:
            self._nlp_attempted = True
            try:
                import spacy

                self._nlp = spacy.load(self.spacy_model_name)
            except Exception:
                self._nlp = None
        return self._nlp

    def clean_whitespace(self, text: str) -> str:
        """Clean leading, trailing, and redundant whitespaces/newlines.

        - Replaces multiple consecutive horizontal spaces with a single space.
        - Trims whitespace on individual lines.
        - Normalizes runs of more than two newlines to double newlines.
        """
        if not text:
            return ""

        # Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace tabs and multiple inline spaces with a single space
        text = re.sub(r"[ \t]+", " ", text)

        # Trim spaces at start and end of each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def mask_regex_pii(self, text: str) -> str:
        """Detect and mask email addresses and phone numbers via regex.

        - Email -> [EMAIL]
        - Phone -> [PHONE]
        """
        # Mask emails
        text = self._email_pattern.sub("[EMAIL]", text)

        # Mask phone numbers (safe from year ranges)
        text = self._phone_pattern.sub("[PHONE]", text)

        # Mask URLs and portfolio links
        text = self._url_pattern.sub("[PORTFOLIO_LINK]", text)

        # Mask elite institute names to mitigate prestige bias
        text = self._institute_pattern.sub("[INSTITUTE]", text)

        # Mask gender honorifics
        text = self._gender_honorifics.sub("[TITLE]", text)

        return text

    def mask_entities_spacy(self, text: str) -> str:
        """Detect and mask named entities using spaCy (en_core_web_sm).

        - PERSON -> [CANDIDATE_NAME]
        - GPE / LOC -> [LOCATION]
        """
        if self.nlp is None:
            # Fallback heuristic if spaCy model is unavailable
            lines = text.split("\n")
            if lines and re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$", lines[0].strip()):
                lines[0] = "[CANDIDATE_NAME]"
                text = "\n".join(lines)
            return text

        doc = self.nlp(text)
        spans: List[Tuple[int, int, str]] = []

        for ent in doc.ents:
            ent_clean = ent.text.strip().lower()

            # Safeguard: Do not mask common technical skills/frameworks as candidate names
            if ent_clean in TECHNICAL_SKILLS_WHITELIST:
                continue

            # Entities shouldn't bridge across newlines into headers/labels
            if "\n" in ent.text:
                first_seg = ent.text.split("\n")[0].strip()
                if not first_seg:
                    continue
                start_char = ent.start_char
                end_char = start_char + len(first_seg)
            else:
                start_char = ent.start_char
                end_char = ent.end_char

            # Adjust if entity greedily included leading labels like 'Email:' or trailing labels
            span_slice = text[start_char:end_char]
            prefix_match = LABEL_PREFIXES_REGEX.match(span_slice)
            if prefix_match:
                start_char += prefix_match.end()

            # Measure suffix relative to the updated span start
            updated_slice = text[start_char:end_char]
            suffix_match = LABEL_SUFFIXES_REGEX.search(updated_slice)
            if suffix_match:
                end_char = start_char + suffix_match.start()

            if start_char >= end_char:
                continue

            if ent.label_ == "PERSON":
                spans.append((start_char, end_char, "[CANDIDATE_NAME]"))
            elif ent.label_ in ("GPE", "LOC"):
                spans.append((start_char, end_char, "[LOCATION]"))

        # Eliminate any overlapping spans and sort descending by start_char
        spans.sort(key=lambda x: (x[0], x[1]), reverse=True)
        non_overlapping_spans: List[Tuple[int, int, str]] = []
        last_start = float("inf")
        for start, end, label in spans:
            if end <= last_start:
                non_overlapping_spans.append((start, end, label))
                last_start = start

        chars = list(text)
        for start, end, placeholder in non_overlapping_spans:
            chars[start:end] = list(placeholder)

        return "".join(chars)

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract structured sections: Skills, Experience, Education.

        Returns a dictionary with keys:
          - 'skills': content under the Skills section
          - 'experience': content under the Experience section
          - 'education': content under the Education section
          - 'other': content not categorized under the three core sections
        """
        sections: Dict[str, str] = {
            "skills": "",
            "experience": "",
            "education": "",
            "other": "",
        }

        if not text:
            return sections

        lines = text.split("\n")
        current_section = "other"
        section_lines: Dict[str, List[str]] = {
            "skills": [],
            "experience": [],
            "education": [],
            "other": [],
        }

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if this line is a section header
            matched_section: Optional[str] = None
            for sec_name, pattern in self.SECTION_PATTERNS.items():
                if pattern.match(line_stripped):
                    matched_section = sec_name
                    break

            if matched_section:
                current_section = matched_section
                # Check if the header line has immediate content after a colon
                if ":" in line_stripped:
                    header_content = line_stripped.split(":", 1)[1].strip()
                    if header_content:
                        section_lines[current_section].append(header_content)
            else:
                section_lines[current_section].append(line_stripped)

        # Assemble cleaned section strings
        for sec in sections:
            sections[sec] = self.clean_whitespace("\n".join(section_lines[sec]))

        return sections

    def anonymize(self, text: str) -> str:
        """Complete de-identification pipeline for raw resume text.

        1. Clean whitespace.
        2. Mask emails and phone numbers via regex with [EMAIL] and [PHONE].
        3. Mask PERSON with [CANDIDATE_NAME] and GPE/LOC with [LOCATION] via spaCy NER.
        4. Clean resulting text whitespace.

        Returns:
            Sanitized, de-identified resume string.
        """
        if not text or not text.strip():
            return ""

        text = self.clean_whitespace(text)
        text = self.mask_regex_pii(text)
        text = self.mask_entities_spacy(text)
        text = self.clean_whitespace(text)
        return text

    def process(self, raw_resume: str) -> Dict[str, Any]:
        """Process raw resume text end-to-end.

        Performs anonymization and section extraction.
        Never persists raw resume personal details.
        """
        anonymized_text = self.anonymize(raw_resume)
        sections = self.extract_sections(anonymized_text)

        return {
            "anonymized_text": anonymized_text,
            "sections": sections,
            "has_content": bool(anonymized_text),
        }


# Compatibility alias
ResumePreprocessor = Anonymizer


# ---------------------------------------------------------------------------
# TextSanitizer – thin API wrapper for the FastAPI endpoint
# ---------------------------------------------------------------------------

class TextSanitizer:
    """High-level sanitizer that de-identifies text AND tracks which
    demographic proxies were redacted (for the SDG 10 audit trail).

    Wraps :class:`Anonymizer` and returns both the clean text and an
    itemised list of redacted proxy categories.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self._anon = Anonymizer(spacy_model=spacy_model)

    def clean_for_model(self, raw_text: str) -> str:
        """Process text identical to Colab training preprocessing:
        1. Strip emails, phone numbers, and URLs via regex.
        2. Filter out PERSON, GPE, and LOC named entities.
        3. Drop punctuation and whitespace tokens.
        4. Lowercase and lemmatize tokens (t.lemma_.lower()).
        """
        if not raw_text or not raw_text.strip():
            return ""

        text = str(raw_text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', ' ', text)

        nlp = self._anon.nlp
        if nlp is not None:
            doc = nlp(text)
            tokens = [
                t.lemma_.lower() for t in doc
                if t.ent_type_ not in ["PERSON", "GPE", "LOC"]
                and not t.is_punct and not t.is_space
            ]
            return " ".join(tokens)
        else:
            cleaned_words = [re.sub(r'[^\w\s]', '', w.lower()) for w in text.split()]
            return " ".join(w for w in cleaned_words if w)

    def sanitize(self, raw_text: str) -> Dict[str, Any]:
        """Return sanitized text plus a list of redacted proxy types.

        Returns
        -------
        dict with keys:
          - ``clean_text``       : str – de-identified text for UI display
          - ``model_text``       : str – lemmatized, punctuation-free text for model inference
          - ``redacted_proxies`` : list[str] – e.g. ["[EMAIL]", "[CANDIDATE_NAME]", ...]
        """
        if not raw_text or not raw_text.strip():
            return {"clean_text": "", "model_text": "", "redacted_proxies": []}

        clean = self._anon.anonymize(raw_text)
        model_text = self.clean_for_model(raw_text)

        # Catalogue which placeholder types are present
        proxy_tags = [
            "[EMAIL]", "[PHONE]", "[CANDIDATE_NAME]",
            "[LOCATION]", "[PORTFOLIO_LINK]", "[INSTITUTE]", "[TITLE]",
        ]
        found = [tag for tag in proxy_tags if tag in clean]

        return {
            "clean_text": clean,
            "model_text": model_text,
            "redacted_proxies": found,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("FairMatch Anonymizer - Self-Test Runner")
    print("=" * 60)

    sample_resume = """
    Johnathan Doe
    Email: john.doe.engineering@example.com
    Phone: +1 (555) 432-8765
    Location: San Francisco, California, USA
    LinkedIn: https://linkedin.com/in/johndoe-sample

    EDUCATION
    Bachelor of Science in Computer Science
    Stanford University, 2020 - 2024
    Relevant Coursework: Machine Learning, Data Structures, Algorithms.

    EXPERIENCE
    Software Engineer Intern at Acme Corporation
    - Built microservices using FastAPI and Docker.
    - Optimized PyTorch model inference latencies by 35% on CPU.
    - Collaborated with cross-functional product managers in Seattle.

    SKILLS
    Python, PyTorch, FastAPI, Docker, SQL, Git, Linux, spaCy, scikit-learn
    """

    anonymizer = Anonymizer()

    print("\n[1] Running Anonymization...")
    anonymized_result = anonymizer.anonymize(sample_resume)
    print("\n--- Anonymized Resume Output ---")
    print(anonymized_result)

    print("\n[2] Running Section Extraction...")
    extracted = anonymizer.extract_sections(anonymized_result)
    for sec_name, content in extracted.items():
        print(f"\n--- Section: [{sec_name.upper()}] ---")
        print(content if content else "(None)")

    # Assertions for verification
    print("\n[3] Running Automated Checks...")
    assert "[EMAIL]" in anonymized_result, "Email masking check failed!"
    assert "[PHONE]" in anonymized_result, "Phone masking check failed!"
    assert "[CANDIDATE_NAME]" in anonymized_result, "Candidate name masking check failed!"
    assert "[LOCATION]" in anonymized_result, "Location masking check failed!"
    assert "john.doe.engineering@example.com" not in anonymized_result, "PII leak: Email found!"
    assert "432-8765" not in anonymized_result, "PII leak: Phone found!"
    assert "Johnathan Doe" not in anonymized_result, "PII leak: Name found!"
    assert "Docker" in anonymized_result, "Technical skill Docker should not be masked as person!"
    assert "PyTorch" in anonymized_result, "Technical skill PyTorch should not be masked as person!"
    assert extracted["skills"], "Skills section should not be empty!"
    assert extracted["experience"], "Experience section should not be empty!"
    assert extracted["education"], "Education section should not be empty!"

    print("\n[SUCCESS] All self-test assertions passed successfully!")
    print("=" * 60)

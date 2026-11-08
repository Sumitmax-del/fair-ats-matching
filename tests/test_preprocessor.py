import unittest
from backend.core.preprocessor import Anonymizer


class TestAnonymizer(unittest.TestCase):
    def setUp(self):
        self.anonymizer = Anonymizer()

    def test_email_and_phone_masking(self):
        raw = "Contact me at alice.smith@example.org or +1 (123) 456-7890."
        sanitized = self.anonymizer.anonymize(raw)
        self.assertIn("[EMAIL]", sanitized)
        self.assertIn("[PHONE]", sanitized)
        self.assertNotIn("alice.smith@example.org", sanitized)
        self.assertNotIn("456-7890", sanitized)

    def test_entity_masking(self):
        raw = "Alexander Hamilton works in Chicago, Illinois."
        sanitized = self.anonymizer.anonymize(raw)
        self.assertIn("[CANDIDATE_NAME]", sanitized)
        self.assertIn("[LOCATION]", sanitized)
        self.assertNotIn("Alexander Hamilton", sanitized)

    def test_section_extraction(self):
        raw = """
        Candidate: Jane Doe

        SKILLS
        Python, FastAPI, PyTorch

        EXPERIENCE
        Worked at Tech Co building deep learning models

        EDUCATION
        BS in Computer Science
        """
        processed = self.anonymizer.process(raw)
        sections = processed["sections"]
        self.assertIn("Python", sections["skills"])
        self.assertIn("Tech Co", sections["experience"])
        self.assertIn("Computer Science", sections["education"])

    def test_tech_whitelist_preserved(self):
        raw = "Skills: PyTorch, Docker, Kubernetes, Deep Learning"
        sanitized = self.anonymizer.anonymize(raw)
        self.assertIn("PyTorch", sanitized)
        self.assertIn("Docker", sanitized)
        self.assertIn("Kubernetes", sanitized)


if __name__ == "__main__":
    unittest.main()

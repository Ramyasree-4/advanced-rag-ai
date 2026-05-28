from __future__ import annotations

import re


class TextPreprocessor:
    def clean(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        return text.strip()

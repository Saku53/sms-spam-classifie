"""Shared text preprocessing + handcrafted features.
Used by train_model.py (training) and app.py (serving) so both behave identically."""
import re
import numpy as np

URL_RE = re.compile(r'(https?://\S+|www\.\S+|\S+\.(?:com|net|org|info|biz)\S*)', re.I)
EMAIL_RE = re.compile(r'\S+@\S+\.\S+')
MONEY_RE = re.compile(r'[£$€]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:p|pence|pounds|gbp|usd)\b', re.I)
PHONE_RE = re.compile(r'\+?\d[\d\s\-]{7,}\d')
SHORTCODE_RE = re.compile(r'\b\d{4,6}\b')
NUM_RE = re.compile(r'\d+')


def normalize_text(text):
    """Light cleaning: keep spam signals (money, phone, urls, '!') as tokens instead of deleting them."""
    t = str(text)
    t = URL_RE.sub(' urltoken ', t)
    t = EMAIL_RE.sub(' emailtoken ', t)
    t = MONEY_RE.sub(' moneytoken ', t)
    t = PHONE_RE.sub(' phonetoken ', t)
    t = SHORTCODE_RE.sub(' shortcodetoken ', t)
    t = NUM_RE.sub(' numtoken ', t)
    t = t.replace('!', ' exclamtoken ')
    return t.lower()


def handcrafted_features(texts):
    rows = []
    for s in texts:
        s = str(s)
        n = max(len(s), 1)
        words = s.split()
        letters = [c for c in s if c.isalpha()]
        rows.append([
            len(s),
            len(words),
            sum(c.isdigit() for c in s) / n,
            (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
            s.count('!'),
            int(bool(URL_RE.search(s))),
            int(bool(PHONE_RE.search(s))),
            int(bool(MONEY_RE.search(s))),
        ])
    return np.array(rows, dtype=float)
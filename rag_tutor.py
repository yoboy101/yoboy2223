"""Retrieval-augmented reading tutor.

Retrieval: the markdown guides in knowledge_base/ are split into sections and indexed
with TF-IDF (scikit-learn). A query built from the screening result is matched by
cosine similarity. (FAISS or ChromaDB with sentence embeddings can replace TF-IDF
without changing the rest of the code.)

Generation: if a local Ollama server is running (https://ollama.com, e.g.
`ollama pull llama3.2`), the retrieved passages and the learner profile are sent to
the open-source LLM, which writes the activities. Without Ollama the tutor falls
back to a template that assembles activities from the retrieved passages, so the
demo always works offline.

Usage:
    python rag_tutor.py --reversal-rate 0.35 --grade 2 --language isiZulu
"""
import argparse
import re

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import KNOWLEDGE_DIR

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


class KnowledgeBase:
    def __init__(self, folder=KNOWLEDGE_DIR):
        self.chunks = []
        for f in sorted(folder.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip()
            for section in re.split(r"\n(?=## )", text):
                body = section.strip()
                if len(body) > 40:
                    self.chunks.append({"source": f.name, "title": title, "text": body})
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c["title"] + "\n" + c["text"] for c in self.chunks])

    def search(self, query, k=4):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in order if sims[i] > 0]


def build_query(profile):
    parts = ["intervention activities"]
    if profile["reversal_rate"] >= 0.15:
        parts.append("letter reversal b/d confusion orientation handwriting")
    if profile["corrected_rate"] >= 0.15:
        parts.append("self-correction handwriting multisensory tracing")
    parts.append("phonics short vowels blending decoding spelling")
    return " ".join(parts)


def describe(profile):
    return (f"Grade {profile['grade']} learner, home language {profile['language']}. "
            f"Handwriting screen: {profile['reversal_rate']:.0%} of letters reversed, "
            f"{profile['corrected_rate']:.0%} corrected, risk level {profile['risk']}.")


def generate_with_ollama(profile, passages, timeout=120):
    context = "\n\n".join(f"[{p['source']}]\n{p['text']}" for p in passages)
    prompt = (
        "You are a supportive reading tutor for South African Foundation Phase learners.\n"
        "Use ONLY the guidance in the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Learner: {describe(profile)}\n\n"
        "Write: (1) a short decodable reading passage (40-60 words), (2) five phonics exercises, "
        "(3) three comprehension questions, (4) five spelling words, and (5) two sentences of "
        f"encouraging feedback in {profile['language']}. Remind the teacher that this is screening, not a diagnosis."
    )
    r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def generate_with_template(profile, passages):
    bullets = []
    for p in passages:  # up to 3 tips from each retrieved section, in relevance order
        tips = [line.strip()[2:] for line in p["text"].splitlines() if line.strip().startswith("- ")]
        bullets += [(p["source"], t) for t in tips[:3]]
    reversal_focus = profile["reversal_rate"] >= 0.15
    words = ["bed", "dog", "bat", "dig", "bud"] if reversal_focus else ["cat", "pen", "sit", "hot", "cup"]
    passage = ("Dad has a big red bed. The dog digs by the bed. Bob the cat sits on the bed and naps. "
               "Dad gets the dog and the cat. Bed time is fun!")
    out = [f"Learner profile: {describe(profile)}", "",
           "1. Reading passage (decodable, short vowels, b/d words):", passage, "",
           "2. Phonics exercises:"]
    out += [f"   - Read and sort by first sound (/b/ or /d/): {', '.join(words)}" if reversal_focus
            else f"   - Blend the sounds: {', '.join(words)}"]
    out += ["   - Tap one finger per sound and say each word slowly (segmenting).",
            "   - Use sound boxes: push a counter for each sound in 'bed' and 'dog'.",
            "   - Trace b and d in sand while saying the path aloud (bat then ball; door handle then line).",
            "   - Change one sound to make a new word: bed -> bad -> bid.", "",
            "3. Comprehension questions:",
            "   - Who has a big red bed?", "   - What does the dog do?", "   - Where does Bob the cat nap?", "",
            f"4. Spelling practice: {', '.join(words)}", "",
            f"5. Feedback in {profile['language']}: (generated by the LLM when Ollama is running; "
            "a speaker of the language should check it before use).", "",
            "Guidance retrieved from the knowledge base:"]
    out += [f"   - {b} [{s}]" for s, b in bullets[:10]]
    out += ["", "Note: this is a screening aid, not a diagnosis. Refer through the SBST for assessment."]
    return "\n".join(out)


def tutor(profile, k=6):
    kb = KnowledgeBase()
    # Skill-focused passages first, then the best passage about the learner's language context.
    passages = kb.search(build_query(profile), k=k)
    seen = {(p["source"], p["text"]) for p in passages}
    passages += [p for p in kb.search(f"{profile['language']} home language South Africa CAPS referral", k=2)
                 if (p["source"], p["text"]) not in seen][:1]
    try:
        text = generate_with_ollama(profile, passages)
        engine = f"Ollama ({OLLAMA_MODEL})"
    except requests.RequestException:
        text = generate_with_template(profile, passages)
        engine = "template fallback (Ollama not running)"
    return {"engine": engine, "passages": passages, "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reversal-rate", type=float, default=0.35)
    ap.add_argument("--corrected-rate", type=float, default=0.10)
    ap.add_argument("--grade", type=int, default=2)
    ap.add_argument("--language", default="isiZulu")
    ap.add_argument("--risk", default="High")
    a = ap.parse_args()
    result = tutor({"reversal_rate": a.reversal_rate, "corrected_rate": a.corrected_rate,
                    "grade": a.grade, "language": a.language, "risk": a.risk})
    print(f"Engine: {result['engine']}")
    print("Retrieved:", [(p["source"], round(p["score"], 3)) for p in result["passages"]])
    print(result["text"])


if __name__ == "__main__":
    main()

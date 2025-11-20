"""
Flask API for Indic Hyphen Prediction Tool (Telugu + Hindi + Bengali)

- NER-based hyphen/segment prediction per language (optional; graceful fallback)
- Romanization (Indic → Latin) with NO translation fallback
- Hindi vibhakti merging (राम + ने → रामने)
- English gloss + language-specific synonyms on hover
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import re
import time
import unicodedata

import torch
from simpletransformers.ner import NERModel
from googletrans import Translator

# ============================== Flask ==============================

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ============================== Globals ============================

translator = Translator()
translation_cache = {}
synonym_cache = {}
romanization_cache = {}

LANG_TE = "te"
LANG_HI = "hi"
LANG_BN = "bn"
SUPPORTED_LANGS = [LANG_TE, LANG_HI, LANG_BN]

MODEL_PATHS = {
    LANG_TE: "./best_model",
    LANG_HI: "./best_model_hindi",
    LANG_BN: "./best_model_bengali",
}

# Model types for each language
MODEL_TYPES = {
    LANG_TE: "bert",
    LANG_HI: "bert",
    LANG_BN: "xlmroberta",  # Bengali model is XLM-RoBERTa
}

MODELS = {}  # loaded models by lang
LABELS = ["HYPHEN", "NO-HYPHEN"]

HINDI_VIBHAKTI = [
    "ने", "को", "से", "में", "पर", "का", "की", "के",
    "वाला", "वाली", "वाले"
]

TELUGU_SYNONYMS = {
    "రాముడు": ["రామ", "రామచంద్రుడు", "కోసలరాజు"],
    "సీతను": ["సీత", "జానకి", "వైదేహి"],
    "రక్షించాడు": ["కాపాడాడు", "సంరక్షించాడు"],
    "నాకు": ["నాకి", "నా వద్దకు"],
    "పుస్తకాలు": ["గ్రంథాలు", "పోథీలు"],
    "చదవడం": ["చదువుట", "అధ్యయనం"],
    "ఇష్టం": ["ప్రియము", "ఇచ్ఛ"],
    "అతను": ["వాడు", "ఆయన"],
    "ప్రతిరోజూ": ["రోజురోజుకూ"],
    "ఉదయం": ["ప్రాతఃకాలం", "పొద్దు"],
    "యోగా": ["యోగాభ్యాసం"],
    "చేస్తాడు": ["చేయును"],
}

HINDI_SYNONYMS = {
    "रामने": ["श्रीरामने", "राघवने"],
    "सीताको": ["जानकीको", "वैदेहीको"],
    "बचाया": ["रक्षा की"],
    "मुझे": ["मेरे लिए"],
    "किताबें": ["पुस्तकें"],
    "पढ़ना": ["अध्ययन करना"],
    "पसंद": ["रुचि"],
    "वह": ["वो"],
}

BENGALI_SYNONYMS = {
    "রাম": ["শ্রীরাম", "রাঘব"],
    "সীতাকে": ["জানকীকে", "সীতাদেবীকে"],
    "রক্ষা করেছে": ["উদ্ধার করেছে", "বাঁচিয়েছে"],
    "আমাকে": ["আমার জন্য"],
    "বই": ["গ্রন্থ", "পুস্তক"],
    "বই পড়তে": ["পাঠ করতে", "অধ্যয়ন করতে"],
    "ভালোবাসি": ["পছন্দ করি", "প্রিয় লাগে"],
    "সে": ["তিনি", "ও"],
    "প্রতিদিন": ["প্রতিটি দিন", "রোজ"],
    "সকালে": ["ভোরবেলা", "প্রভাতে"],
}

# ====================== Indic transliteration (primary) ======================

HAS_INDIC = False
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except Exception:
    HAS_INDIC = False


def _pick_pronunciation(gt_result):
    """Try to extract Latin pronunciation from googletrans result."""
    p = getattr(gt_result, "pronunciation", None)
    if p:
        return p
    try:
        extra = getattr(gt_result, "extra_data", {}) or {}
        p = extra.get("origin_pronunciation")
        if p:
            return p
    except Exception:
        pass
    return None


def romanize_text(text: str, lang: str, use_cache: bool = True) -> str:
    """
    Romanize Telugu/Hindi/Bengali without using translation text.
    Order: indic-transliteration (IAST) → google pronunciation → original.
    """
    text = (text or "").strip()
    if not text:
        return ""

    ck = (lang, "romanize", text)
    if use_cache and ck in romanization_cache:
        return romanization_cache[ck]

    roman = None
    try:
        # 1) indic-transliteration (preferred; high quality)
        if HAS_INDIC:
            if lang == LANG_HI:
                roman = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
            elif lang == LANG_TE:
                roman = transliterate(text, sanscript.TELUGU, sanscript.IAST)
            elif lang == LANG_BN:
                roman = transliterate(text, sanscript.BENGALI, sanscript.IAST)

        # 2) google pronunciation (never use .text)
        if not roman:
            time.sleep(0.05)
            r1 = translator.translate(text, src=lang, dest="en")
            roman = _pick_pronunciation(r1)
        if not roman:
            r2 = translator.translate(text, src=lang, dest=lang)
            roman = _pick_pronunciation(r2)

    except Exception as e:
        print(f"[WARN] Romanization pipeline error: {e}")

    roman = (roman or "").strip() or text
    romanization_cache[ck] = roman
    return roman


# ============================== Utilities ===============================

def merge_hindi_vibhakti(tokens):
    if not tokens:
        return tokens
    out = []
    i = 0
    while i < len(tokens):
        cur = tokens[i]
        if i + 1 < len(tokens) and tokens[i + 1] in HINDI_VIBHAKTI:
            out.append(cur + tokens[i + 1])
            i += 2
        else:
            out.append(cur)
            i += 1
    return out


def load_model_for_lang(lang: str) -> bool:
    """Try to load model; if not present, we just skip (graceful fallback)."""
    path = MODEL_PATHS.get(lang)
    if not path or not os.path.exists(path):
        print(f"[WARN] Model path not found for '{lang}': {path}")
        return False

    model_type = MODEL_TYPES.get(lang, "bert")

    try:
        model = NERModel(
            model_type,
            path,
            use_cuda=torch.cuda.is_available(),
            args={
                "silent": True,
                "use_multiprocessing": False,
            },
        )
        MODELS[lang] = model
        print(f"[OK] Loaded {lang} model from {path} as {model_type}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed loading {lang} model: {e}")
        return False


def preprocess_sentence(s: str) -> str:
    s = (s or "").replace("\ufeff", "")
    s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cf"})
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def predict_hyphens(sentence: str, lang: str):
    """
    If model is loaded → real prediction.
    If not → graceful: return [sentence] (still enables romanization/translation).
    """
    sentence = preprocess_sentence(sentence)
    if not sentence:
        return {"error": "Empty sentence after preprocessing"}

    if lang in MODELS:
        try:
            tokens = sentence.split()
            if lang == LANG_HI:
                tokens = merge_hindi_vibhakti(tokens)

            preds, _ = MODELS[lang].predict([tokens], split_on_space=False)
            labels = [list(d.values())[0] for d in preds[0]]

            segs = []
            cur = []
            for tok, lab in zip(tokens, labels):
                cur.append(tok)
                if lab == "HYPHEN":
                    segs.append(" ".join(cur))
                    cur = []
            if cur:
                segs.append(" ".join(cur))
            return segs or [sentence]
        except Exception as e:
            print(f"[ERROR] Prediction failed, falling back: {e}")

    # Fallback when model missing or failed
    return [sentence]


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)


def translate_to_en(text: str, lang: str, use_cache=True) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    key = (lang, "en", text)
    if use_cache and key in translation_cache:
        return translation_cache[key]
    try:
        time.sleep(0.05)
        out = translator.translate(text, src=lang, dest="en").text
        translation_cache[key] = out
        return out
    except Exception:
        return f"[{text}]"


def translate_en_to_lang(text: str, lang: str, use_cache=True) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    key = ("en", lang, text)
    if use_cache and key in translation_cache:
        return translation_cache[key]
    try:
        time.sleep(0.05)
        out = translator.translate(text, src="en", dest=lang).text
        translation_cache[key] = out
        return out
    except Exception:
        return text


def get_synonyms(lang: str, word: str, use_cache=True):
    word = _clean_text(word)
    if not word:
        return []
    ck = (lang, word)
    if use_cache and ck in synonym_cache:
        return synonym_cache[ck]

    if lang == LANG_TE:
        manual = TELUGU_SYNONYMS
    elif lang == LANG_HI:
        manual = HINDI_SYNONYMS
    elif lang == LANG_BN:
        manual = BENGALI_SYNONYMS
    else:
        manual = {}

    if word in manual:
        synonym_cache[ck] = manual[word]
        return manual[word]

    # lightweight back-translation variant
    try:
        en = translate_to_en(word, lang, True)
        var = translate_en_to_lang(en, lang, True)
        variants = [var] if var and var != word else []
        synonym_cache[ck] = variants[:3]
        return variants[:3]
    except Exception:
        synonym_cache[ck] = []
        return []


def _normalize_lang(lang: str) -> str:
    lang = (lang or "").lower()
    return lang if lang in SUPPORTED_LANGS else LANG_TE


# ================================ Routes =================================

@app.route("/")
def root():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "model_paths": MODEL_PATHS,
        "model_types": MODEL_TYPES,
        "cuda_available": bool(torch.cuda.is_available()),
        "has_indic_transliteration": HAS_INDIC,
        "translation_cache_size": len(translation_cache),
        "romanization_cache_size": len(romanization_cache),
        "synonym_cache_size": len(synonym_cache),
        "supported_langs": SUPPORTED_LANGS,
    })


@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    sentence = (data.get("sentence") or "").strip()
    lang = _normalize_lang(data.get("lang", LANG_TE))
    if not sentence:
        return jsonify({"error": "Empty sentence"}), 400

    segments = predict_hyphens(sentence, lang)
    if isinstance(segments, dict) and "error" in segments:
        return jsonify(segments), 500

    payload = []
    for seg in segments:
        payload.append({
            "lang": lang,
            "text": seg,
            "english": translate_to_en(seg, lang, True),
            "romanized": romanize_text(seg, lang, True),
        })

    return jsonify({
        "original": sentence,
        "lang": lang,
        "romanized_full": romanize_text(sentence, lang, True),
        "segments": payload,
    })


@app.route("/translate", methods=["POST"])
def api_translate():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    lang = _normalize_lang(data.get("lang", LANG_TE))
    if not text:
        return jsonify({"error": "No text provided"}), 400

    return jsonify({
        "original": text,
        "lang": lang,
        "translated": translate_to_en(text, lang, True),
        "romanized": romanize_text(text, lang, True),
        "synonyms": get_synonyms(lang, text, True),
    })


@app.route("/cache/clear")
def cache_clear():
    translation_cache.clear()
    synonym_cache.clear()
    romanization_cache.clear()
    return jsonify({"message": "Cleared all caches"})


# ================================ Main ==================================

if __name__ == "__main__":
    any_loaded = False
    for L in SUPPORTED_LANGS:
        ok = load_model_for_lang(L)
        any_loaded = any_loaded or ok

    if not any_loaded:
        print(" No models loaded. API will still work (fallback segmentation).")

    app.run(host="0.0.0.0", port=5000, debug=True)

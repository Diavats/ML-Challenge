"""Step 1: turn messy names/addresses into clean, comparable fields.

We never drop a row here. Every record keeps its id; we only rewrite text.

Output columns per record:
  name   - name words, lowercase, Latin letters, legal words removed
  legal  - the legal words we removed, canonical form (e.g. "ltd pvt")
  nosp   - name with spaces removed (catches "maurewilliamscolombier.com")
  phon   - sound-alike key of the name (catches Hindi transliteration: "phuds" ~ "foods")
  addr   - address words, abbreviations unified, state removed
  state  - state/region code if we found one (e.g. "ny", "mh")
  house  - first number in the address (house / plot number)
  street - first word after the house number (usually the street name)
  nums   - all numbers in the address, sorted, space separated

Run:  python -m src.normalize          (normalizes all 6 files -> data/*.parquet)
      python -m src.normalize --check  (quick self-test on hand-picked examples)
"""
import re
import sys
import unicodedata
from multiprocessing import Pool

import pandas as pd
from anyascii import anyascii

from src.load import DATA, read_source

# ---------------------------------------------------------------- word maps
# Legal-form words -> one canonical spelling. Typo'd / transliterated forms
# (praivet, limitet, elelpi ...) were taken from the most common words in the data.
LEGAL = {}
for canon, words in {
    "pvt": "private pvt prvt pvte praivet praibhet piraivet praivrr pra",
    "ltd": "limited ltd limitet limirrd li",
    "llp": "llp elelpi",
    "llc": "llc", "pllc": "pllc", "lp": "lp", "pc": "pc", "plc": "plc",
    "inc": "inc incorporated",
    "corp": "corp corporation",
    "co": "co company cie compagnie",
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "ei": "ei", "ets": "ets etablissements",
}.items():
    for w in words.split():
        LEGAL[w] = canon

# Address words -> short form. Long -> short, so "street"/"st" both become "st".
# ("saint" also becomes "st": "St Louis" and "Saint Louis" then agree.)
ABBR = {}
for short, words in {
    "st": "street saint", "rd": "road", "ave": "avenue av", "dr": "drive",
    "ln": "lane", "ct": "court", "blvd": "boulevard bd boul", "pl": "place",
    "cir": "circle", "hwy": "highway", "pkwy": "parkway", "ter": "terrace",
    "trl": "trail", "sq": "square", "mt": "mount", "n": "north", "s": "south",
    "e": "east", "w": "west", "fl": "floor", "apt": "apartment", "ste": "suite",
    "bldg": "building", "r": "rue", "all": "allee", "imp": "impasse",
    "rte": "route", "ch": "chemin", "dist": "district", "tq": "taluk tal",
    "delhi": "dilli",
}.items():
    for w in words.split():
        ABBR[w] = short

# Filler words in addresses that carry no signal (or are injected noise).
ADDR_DROP = set("null na no ndeg number near nr opp opposite city township "
                "townshiip twp".split())
# Words that can't be "the street name" (street types, directions, French little words).
STREET_SKIP = set(ABBR.values()) | set("de du des la le les the of".split())

# State / region names -> code, per country (codes collide across countries,
# e.g. "ga" is Georgia in US but Goa in India, so we keep them separate).
US_STATES = dict(al="alabama", ak="alaska", az="arizona", ar="arkansas",
    ca="california", co="colorado", ct="connecticut", de="delaware",
    dc="district of columbia", fl="florida", ga="georgia", hi="hawaii",
    id="idaho", il="illinois", in_="indiana", ia="iowa", ks="kansas",
    ky="kentucky", la="louisiana", me="maine", md="maryland",
    ma="massachusetts", mi="michigan", mn="minnesota", ms="mississippi",
    mo="missouri", mt="montana", ne="nebraska", nv="nevada",
    nh="new hampshire", nj="new jersey", nm="new mexico", ny="new york",
    nc="north carolina", nd="north dakota", oh="ohio", ok="oklahoma",
    or_="oregon", pa="pennsylvania", ri="rhode island", sc="south carolina",
    sd="south dakota", tn="tennessee", tx="texas", ut="utah", vt="vermont",
    va="virginia", wa="washington", wv="west virginia", wi="wisconsin",
    wy="wyoming", pr="puerto rico")
IN_STATES = dict(ap="andhra pradesh", ar="arunachal pradesh", as_="assam",
    br="bihar", cg="chhattisgarh|chattisgarh|ct", ga="goa", gj="gujarat",
    hr="haryana", hp="himachal pradesh", jh="jharkhand", ka="karnataka|krnatk",
    kl="kerala", mp="madhya pradesh|mdhy prdes", mh="maharashtra|mharastr",
    mn="manipur", ml="meghalaya", mz="mizoram", nl="nagaland", od="odisha|orissa|or",
    pb="punjab", rj="rajasthan", sk="sikkim", tn="tamil nadu", tg="telangana|ts",
    tr="tripura", up="uttar pradesh|uttr prdes|uttr pradesh|uttar prdes",
    uk="uttarakhand|ut", wb="west bengal", dl="delhi|new delhi|nct of delhi",
    jk="jammu and kashmir|jammu kashmir", la="ladakh", py="puducherry|pondicherry",
    ch="chandigarh", an="andaman and nicobar")
FR_REGIONS = dict(naq="nouvelle aquitaine", hdf="hauts de france", pdl="pays de la loire",
    nord="nord", gir="gironde", la="loire atlantique", pdc="pas de calais")


def _state_map(table, codes_in_data=True):
    """{("ny",): "ny", ("new","york"): "ny", ...}.
    State names go through ABBR too, because addresses do ("north carolina" -> "n carolina").
    codes_in_data=False for France: our region codes are made up, so the bare code
    must not match a real word (French "la" in "rue de la paix")."""
    out = {}
    for code, names in table.items():
        code = code.rstrip("_")  # in_/or_/as_ are python keywords, strip the "_"
        if codes_in_data:
            out[(code,)] = code
        for name in names.split("|"):
            out[tuple(ABBR.get(w, w) for w in name.split())] = code
    return out


STATES = {"us": _state_map(US_STATES), "india": _state_map(IN_STATES),
          "france": _state_map(FR_REGIONS, codes_in_data=False)}

DOMAIN = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:co\.in|com|net|org|in|fr|biz|info|us)\b")
NON_ALNUM = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------- helpers
def to_ascii(s):
    """Unicode -> lowercase plain Latin. Hindi script is transliterated, accents removed."""
    return anyascii(unicodedata.normalize("NFKC", s)).lower()


def words(s):
    """ascii text -> list of words. Dots are deleted first so "l.l.c." -> "llc", "s.a.s" -> "sas"."""
    return NON_ALNUM.sub(" ", s.replace(".", "")).split()


PHON_PAIRS = [("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"),
              ("dh", "d"), ("sh", "s"), ("ch", "c")]
PHON_TABLE = str.maketrans("cqwzm", "kkvjn", "aeiouyh")  # map similar sounds, delete vowels + h


def phonetic(word):
    """Crude sound-alike key: 'foods' and 'phuds' both -> 'fds'.
    ponytail: hand-rolled, tuned for Hindi->Latin transliteration; swap for a real
    phonetic lib (e.g. jellyfish metaphone) if it misses too much."""
    for a, b in PHON_PAIRS:
        word = word.replace(a, b)
    word = word.translate(PHON_TABLE)
    return re.sub(r"(.)\1+", r"\1", word)  # collapse repeats: "ll" -> "l"


def norm_name(raw):
    s = to_ascii(raw)
    s = DOMAIN.sub(r" \1 ", s)          # "pinnaclep.com" -> "pinnaclep"
    s = s.replace("&", " and ")
    toks = words(s)
    legal = sorted({LEGAL[t] for t in toks if t in LEGAL})
    name = [t for t in toks if t not in LEGAL]
    if not name:                        # name was only legal words ("Ltd") - keep them
        name = toks
    return (" ".join(name), " ".join(legal), "".join(name),
            " ".join(phonetic(t) for t in name))


def norm_addr(raw, country):
    smap = STATES.get(country.lower(), {})
    state, keep = "", []
    # Addresses are comma-separated pieces. The state is always a whole piece
    # (", MO" / ", New York"), so "Fl 0" (floor 0) is never mistaken for Florida.
    for piece in to_ascii(raw).replace("n/a", " ").split(","):
        toks = [ABBR.get(t, t) for t in words(piece)]
        if not state and tuple(toks) in smap:
            state = smap[tuple(toks)]
            continue
        keep += [t.lstrip("0") or "0" if t.isdigit() else t   # "062" -> "62"
                 for t in toks if t not in ADDR_DROP]

    # house number = first number; street = first real word after it
    house, street = "", ""
    for j, t in enumerate(keep):
        if t.isdigit():
            house = t
            street = next((w for w in keep[j + 1:]
                           if w.isalpha() and len(w) > 1 and w not in STREET_SKIP), "")
            break
    nums = " ".join(sorted({t for t in keep if t.isdigit()}))
    return " ".join(keep), state, house, street, nums


NAME_COLS = ["name", "legal", "nosp", "phon"]
ADDR_COLS = ["addr", "state", "house", "street", "nums"]


def _norm_rows(rows):
    """rows = list of (name, address, country) -> list of 9-tuples."""
    return [norm_name(n) + norm_addr(a, c) for n, a, c in rows]


def normalize_df(df, workers=None):
    """Adds the normalized columns to a source dataframe (keeps the raw ones too)."""
    rows = list(zip(df["business_name"], df["business_address"], df["country"]))
    chunks = [rows[i:i + 200_000] for i in range(0, len(rows), 200_000)]
    del rows
    with Pool(workers) as pool:  # one process per CPU core; chunk results become small frames
        parts = [pd.DataFrame(p, columns=NAME_COLS + ADDR_COLS)
                 for p in pool.imap(_norm_rows, chunks)]
    del chunks
    norm = pd.concat(parts, ignore_index=True).set_index(df.index)
    return pd.concat([df.rename(columns={"entity_id": "id"}), norm], axis=1)


def _check():
    """Hand-picked real examples from the training data. Fails loudly if a rule breaks."""
    n = norm_name("Maure Williams Colombier Inc")
    assert n[:2] == ("maure williams colombier", "inc"), n
    assert norm_name("maurewilliamscolombier.com")[2] == norm_name("Maure Williams Colombier")[2]
    assert norm_name("राम मार्केटिंग प्राइवेट लिमिटेड")[1] == "ltd pvt"
    assert norm_name("L.L.C. Moncada")[1] == "llc"
    assert phonetic("foods") == phonetic("phuds")
    a = norm_addr("62 Easton Street, MA, New Bedford", "US")
    b = norm_addr("MA, NEW BEDFRD CITY, 062 EASTON ST", "US")
    assert (a[1], a[2], a[3]) == (b[1], b[2], b[3]) == ("ma", "62", "easton"), (a, b)
    assert norm_addr("85 Wanye Avenue, Ticonderoga Townshiip, New York", "US")[1] == "ny"
    assert norm_addr("Andheri, Mumbai, mharastr", "India")[1] == "mh"
    assert norm_addr("105 ELM ST, MORGANTON, North Carolina", "US")[1] == "nc"
    assert norm_addr("5 rue de la paix, Nantes, Pays de la Loire", "France")[1] == "pdl"
    assert norm_addr("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France")[:4] == \
        ("63 r de dieppe lille", "hdf", "63", "dieppe")
    assert norm_addr("10065 Lilac Avenue, Fl 0, Saint Louis, MO", "US")[1] == "mo"
    assert norm_addr("N/A, Hyderabad, TG, Hyderabad, Villa 269", "India")[:2] == \
        ("hyderabad hyderabad villa 269", "tg")
    print("normalize self-check OK")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
        sys.exit()
    DATA.mkdir(exist_ok=True)
    for split in ("train", "test"):
        for n in (1, 2, 3):
            df = normalize_df(read_source(split, n))
            df.to_parquet(DATA / f"{split}_s{n}.parquet", index=False)
            print(split, n, len(df), "rows ->", DATA / f"{split}_s{n}.parquet", flush=True)

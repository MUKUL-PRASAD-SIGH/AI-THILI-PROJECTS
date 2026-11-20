"""Milestone 1: collect and preprocess a corpus of candle product descriptions.

Two sources feed the corpus:

1. ``data/raw/seed_descriptions.txt`` - hand-written listings in the style of
   a small candle shop's website. They arrive messy (HTML tags, prices,
   "SHOP NOW" shouting, smart quotes, duplicate listings) and are cleaned here.
2. A catalog generator. Twenty listings are far too few to train a GPT from
   scratch, so we expand the shop's catalog: products are built from a scent
   library (notes, families, moods, rooms) and described with many varied
   sentence patterns. The randomness is seeded so the corpus is reproducible.

Both sources are cleaned with the same function and written as one record
per product (see ``app.core.Product.to_record``) to ``data/corpus.txt``.
"""
from __future__ import annotations

import html
import random
import re
import unicodedata
from pathlib import Path

from app.core import END, LENGTHS, Product, length_bucket, parse_notes

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED_FILE = DATA_DIR / "raw" / "seed_descriptions.txt"
CORPUS_FILE = DATA_DIR / "corpus.txt"

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

_PROMO = re.compile(
    r"(\b(only|price:?)\s*)?\$\s?\d+(\.\d{2})?!*"          # prices
    r"|\bshop now\b[^.!?]*[.!?]*"                          # calls to action
    r"|\bbuy now\b[^.!?]*[.!?]*",
    re.IGNORECASE,
)
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"',
           "–": "-", "—": "-", "…": "..."}


def clean_text(text: str) -> str:
    """Normalise one description: strip HTML and promo text, fix quotes,
    tame repeated punctuation and collapse whitespace."""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    for bad, good in _QUOTES.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    text = _PROMO.sub("", text)
    text = re.sub(r"([!?])\1+", r"\1", text)          # "!!!" -> "!"
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"\s+-\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def parse_seed_file(path: Path = SEED_FILE) -> list[Product]:
    """Read the raw listings, clean them and drop exact duplicates."""
    raw = path.read_text(encoding="utf-8")
    raw = "\n".join(l for l in raw.splitlines() if not l.lstrip().startswith("#"))
    products, seen = [], set()
    for block in raw.split("\n---"):
        fields, body = {}, []
        for line in block.strip().splitlines():
            m = re.match(r"^\s*(name|notes|size)\s*:\s*(.*)$", line, re.IGNORECASE)
            if m and not body:
                fields[m.group(1).lower()] = m.group(2).strip()
            elif line.strip():
                body.append(line.strip())
        if "name" not in fields or not body:
            continue
        desc = clean_text(" ".join(body))
        p = Product(
            name=re.sub(r"\s+", " ", fields["name"]),
            notes=parse_notes(fields.get("notes", "")),
            size=re.sub(r"\s+", " ", fields.get("size", "")),
            description=desc,
        )
        p.length = length_bucket(desc)
        key = (p.name.lower(), desc.lower())
        if key in seen:
            continue
        seen.add(key)
        products.append(p)
    return products


# --------------------------------------------------------------------------
# Catalog generator
# --------------------------------------------------------------------------

# note -> (family, ways of describing it in a sentence)
NOTES: dict[str, tuple[str, list[str]]] = {
    "lavender": ("floral", ["soft lavender", "herbal lavender", "sun-warmed lavender"]),
    "rose": ("floral", ["velvet rose", "fresh-cut rose", "dewy rose petals"]),
    "jasmine": ("floral", ["night-blooming jasmine", "delicate jasmine"]),
    "peony": ("floral", ["blushing peony", "airy peony"]),
    "gardenia": ("floral", ["creamy gardenia", "lush gardenia"]),
    "neroli": ("floral", ["sparkling neroli", "bright neroli blossom"]),
    "lemon": ("citrus", ["bright lemon", "zesty lemon peel", "sunny lemon"]),
    "grapefruit": ("citrus", ["pink grapefruit", "juicy grapefruit"]),
    "bergamot": ("citrus", ["sparkling bergamot", "crisp bergamot"]),
    "orange peel": ("citrus", ["candied orange peel", "sweet orange peel"]),
    "lime": ("citrus", ["fresh lime", "a squeeze of lime"]),
    "yuzu": ("citrus", ["tangy yuzu", "bright yuzu"]),
    "cedarwood": ("woody", ["dry cedarwood", "grounding cedarwood", "smooth cedarwood"]),
    "sandalwood": ("woody", ["creamy sandalwood", "soft sandalwood"]),
    "oud": ("woody", ["deep oud", "resinous oud"]),
    "pine": ("woody", ["fresh-cut pine", "crisp pine needles"]),
    "smoked birch": ("woody", ["smoked birch", "smoky birch bark"]),
    "vetiver": ("woody", ["earthy vetiver", "smoky vetiver"]),
    "driftwood": ("woody", ["sun-bleached driftwood", "weathered driftwood"]),
    "vanilla": ("gourmand", ["warm vanilla", "soft vanilla", "creamy vanilla"]),
    "vanilla bean": ("gourmand", ["rich vanilla bean", "creamy vanilla bean"]),
    "honey": ("gourmand", ["golden honey", "wildflower honey"]),
    "brown sugar": ("gourmand", ["caramelised brown sugar", "a sprinkle of brown sugar"]),
    "espresso": ("gourmand", ["dark roasted espresso", "freshly pulled espresso"]),
    "oat milk": ("gourmand", ["creamy oat milk", "warm oat milk"]),
    "tonka bean": ("gourmand", ["sweet tonka bean", "toasty tonka bean"]),
    "almond": ("gourmand", ["toasted almond", "sweet almond"]),
    "maple": ("gourmand", ["maple syrup", "golden maple"]),
    "coconut": ("gourmand", ["creamy coconut", "sun-kissed coconut"]),
    "cinnamon": ("spicy", ["warm cinnamon", "cinnamon bark", "spiced cinnamon"]),
    "clove": ("spicy", ["clove", "a hint of clove"]),
    "nutmeg": ("spicy", ["a pinch of nutmeg", "freshly grated nutmeg"]),
    "cardamom": ("spicy", ["green cardamom", "warm cardamom"]),
    "black pepper": ("spicy", ["cracked black pepper", "a spark of black pepper"]),
    "ginger": ("spicy", ["fresh ginger", "zingy ginger root"]),
    "eucalyptus": ("fresh", ["cooling eucalyptus", "clean eucalyptus"]),
    "spearmint": ("fresh", ["cool spearmint", "fresh spearmint"]),
    "peppermint": ("fresh", ["crisp peppermint", "icy peppermint"]),
    "sea salt": ("fresh", ["crisp sea salt", "salty ocean air"]),
    "white musk": ("fresh", ["soft white musk", "clean white musk"]),
    "cotton": ("fresh", ["fresh cotton", "line-dried cotton"]),
    "rain": ("fresh", ["summer rain", "petrichor after rain"]),
    "green tea": ("herbal", ["fresh green tea leaves", "steeped green tea"]),
    "sage": ("herbal", ["dried white sage", "soft sage"]),
    "basil": ("herbal", ["a green snap of basil", "sweet basil"]),
    "rosemary": ("herbal", ["fresh rosemary", "garden rosemary"]),
    "fig leaf": ("herbal", ["green fig leaf", "leafy fig"]),
    "black fig": ("fruity", ["ripe black fig", "jammy black fig"]),
    "strawberry": ("fruity", ["juicy strawberries", "sweet strawberry"]),
    "rhubarb": ("fruity", ["tart rhubarb", "pink rhubarb"]),
    "pear": ("fruity", ["ripe pear", "crisp pear"]),
    "blackcurrant": ("fruity", ["dark blackcurrant", "tangy blackcurrant"]),
    "pumpkin": ("fruity", ["spiced pumpkin", "roasted pumpkin"]),
    "apple": ("fruity", ["crisp red apple", "baked apple"]),
    "amber": ("woody", ["glowing amber", "warm amber"]),
    "leather": ("woody", ["worn leather", "soft suede leather"]),
    "tobacco leaf": ("woody", ["sweet tobacco leaf", "dried tobacco leaf"]),
}

FAMILY = {
    "floral": {
        "adj": ["romantic", "elegant", "airy", "graceful", "dreamy"],
        "image": ["a garden in full bloom", "a bouquet on the kitchen table",
                  "a spring morning with the windows open", "a walk through a flower market"],
        "feel": ["Floral, elegant and clean.", "Soft and romantic without ever feeling heavy.",
                 "It is light, pretty and quietly uplifting.", "A gentle floral that feels like fresh flowers in every room."],
        "room": ["bedrooms", "dressing tables", "reading corners", "bathrooms"],
        "words": ["Bloom", "Garden", "Petal", "Blossom", "Meadow", "Posy"],
    },
    "citrus": {
        "adj": ["zesty", "bright", "uplifting", "sparkling", "cheerful"],
        "image": ["sunshine in a jar", "a bowl of fruit on a summer table",
                  "an open window on a bright morning", "a Mediterranean terrace"],
        "feel": ["Energising and uplifting.", "Bright, clean and full of sunshine.",
                 "It wakes up a room in minutes.", "A cheerful scent that lifts the mood of any space."],
        "room": ["kitchens", "home offices", "entryways", "sunny living rooms"],
        "words": ["Grove", "Sunrise", "Zest", "Riviera", "Sunshine", "Squeeze"],
    },
    "woody": {
        "adj": ["moody", "grounding", "rich", "earthy", "sophisticated"],
        "image": ["a cabin in the woods", "an old library with a crackling fire",
                  "a forest walk after the rain", "a quiet evening by the fireplace"],
        "feel": ["Warm, earthy and grounding.", "Rich without being sweet, and deeply calming.",
                 "A grown-up scent with real depth.", "It brings the calm of the forest indoors."],
        "room": ["living rooms", "studies", "dens", "large open spaces"],
        "words": ["Cabin", "Hearth", "Timber", "Forest", "Ember", "Library"],
    },
    "gourmand": {
        "adj": ["cozy", "comforting", "sweet", "indulgent", "creamy"],
        "image": ["a bakery at dawn", "your favorite corner cafe",
                  "a kitchen on a snowy afternoon", "a warm drink on a cold night"],
        "feel": ["Cozy, gentle and sweet in the best way.", "Comforting and warm without being sugary.",
                 "It smells good enough to eat.", "Pure comfort for slow evenings at home."],
        "room": ["kitchens", "living rooms", "cozy reading nooks", "dining rooms"],
        "words": ["Bakery", "Cafe", "Treat", "Cream", "Dessert", "Kitchen"],
    },
    "spicy": {
        "adj": ["warm", "spiced", "festive", "inviting", "glowing"],
        "image": ["a winter market", "mulled wine by the fire",
                  "a kitchen during the holidays", "a spice bazaar at dusk"],
        "feel": ["Warm, spiced and inviting.", "Festive and cozy from the very first light.",
                 "A glowing scent for long, cold evenings.", "It makes any room feel like a gathering."],
        "room": ["living rooms", "dining tables", "entryways", "holiday gatherings"],
        "words": ["Spice", "Hearth", "Market", "Ember", "Chai", "Festival"],
    },
    "fresh": {
        "adj": ["clean", "crisp", "airy", "refreshing", "calming"],
        "image": ["an open window by the coast", "a spa day at home",
                  "clean laundry drying in the breeze", "a cool morning by the sea"],
        "feel": ["Clean, airy and calming.", "Breathe a little deeper and let the day go.",
                 "Crisp and refreshing, like the air after a storm.", "A fresh, light scent that clears the air."],
        "room": ["bathrooms", "bedrooms", "laundry rooms", "kitchens"],
        "words": ["Breeze", "Coast", "Spa", "Tide", "Linen", "Morning"],
    },
    "herbal": {
        "adj": ["green", "calming", "earthy", "restorative", "peaceful"],
        "image": ["a kitchen herb garden", "a quiet greenhouse",
                  "a morning yoga practice", "a walk through wild hills"],
        "feel": ["Green, calm and quietly herbal.", "Ideal for a reset after a hectic week.",
                 "A peaceful scent for slowing down.", "It feels like stepping into a greenhouse."],
        "room": ["home offices", "yoga corners", "kitchens", "bathrooms"],
        "words": ["Garden", "Greenhouse", "Herb", "Sprig", "Leaf", "Field"],
    },
    "fruity": {
        "adj": ["juicy", "playful", "bright", "lush", "sweet"],
        "image": ["a farmers market in July", "an orchard at harvest time",
                  "a picnic in the sun", "a jar of homemade jam"],
        "feel": ["Playful, juicy and bright.", "Sweet and fruity without being too much.",
                 "A happy scent that brightens any day.", "Lush and fruity, like summer in a jar."],
        "room": ["kitchens", "living rooms", "kids' rooms", "patios"],
        "words": ["Orchard", "Harvest", "Picnic", "Jam", "Patch", "Market"],
    },
}

NAME_ADJ = ["Golden", "Midnight", "Velvet", "Wild", "Little", "Morning", "Quiet",
            "Sunday", "Honeyed", "Misty", "Autumn", "Winter", "Summer", "Spring", "Coastal"]

SIZES = {"4 oz": 25, "6 oz": 35, "8 oz": 50, "10 oz": 60, "12 oz": 70, "14 oz": 80, "16 oz": 90}
WAX = ["natural soy wax", "a coconut-soy blend", "pure beeswax", "coconut-apricot wax",
       "clean-burning soy wax", "a vegan coconut wax blend"]
WICK = ["a cotton wick", "a crackling wooden wick", "a lead-free cotton wick", "two cotton wicks"]
VESSEL = {"4 oz": ["travel tin", "mini glass jar"],
          "6 oz": ["frosted glass tumbler", "ceramic pot", "amber glass jar"],
          "8 oz": ["amber glass jar", "reusable ceramic tumbler", "frosted glass tumbler", "matte white jar"],
          "10 oz": ["smoky glass jar", "matte black jar with a wooden lid", "ceramic bowl"],
          "12 oz": ["amber glass jar", "hand-thrown ceramic bowl", "matte black jar"],
          "14 oz": ["large ceramic bowl", "two-wick glass tumbler"],
          "16 oz": ["large amber jar", "statement ceramic vessel"]}
OCCASION = ["while you read, stretch or work from home", "on the first cool evening of the year",
            "after a long day", "when friends come over for dinner", "during a slow Sunday morning",
            "as you run a warm bath", "whenever the house needs a reset", "on rainy afternoons"]
GIFT = ["birthdays, housewarmings and thank-yous", "weddings and anniversaries",
        "teachers, neighbours and new homeowners", "anyone who loves a cozy night in",
        "the holidays and everything in between"]


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _a(word: str) -> str:
    """Prefix the right indefinite article."""
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _make_name(rng: random.Random, notes: list[str], family: str) -> str:
    style = rng.random()
    first = notes[0].title()
    if style < 0.35 and len(notes) >= 2:
        return f"{first} & {notes[1].title()}"
    if style < 0.65:
        return f"{first} {rng.choice(FAMILY[family]['words'])}"
    return f"{rng.choice(NAME_ADJ)} {rng.choice(FAMILY[family]['words'])}"


def _opening(rng, name, phrases, adj, image):
    a = phrases
    patterns = [
        lambda: f"{_cap(a[0])} meets {_join(a[1:])} in this {adj} candle." if len(a) > 1 else None,
        lambda: f"{_cap(_join(a))} come together like {image}." if len(a) > 1 else None,
        lambda: f"Our {name} candle blends {a[0]} with {_join(a[1:])}." if len(a) > 1 else None,
        lambda: f"{_cap(_join(a))} make this candle feel {adj} from the very first light." if len(a) > 1 else None,
        lambda: f"Think {image}: {_join(a)}, poured into one {adj} candle.",
        lambda: f"{name} wraps the room in {_join(a)}.",
        lambda: f"{_cap(a[0])} takes the lead in {name}, softened by {_join(a[1:])}." if len(a) > 1 else None,
        lambda: f"{_cap(_a(adj))} blend of {_join(a)}.",
    ]
    while True:
        s = rng.choice(patterns)()
        if s:
            return s


def make_product(rng: random.Random) -> Product:
    """Invent one plausible candle and describe it."""
    n_notes = rng.choices([1, 2, 3], weights=[1, 5, 4])[0]
    first = rng.choice(list(NOTES))
    family = NOTES[first][0]
    # Companion notes come mostly from the same family, sometimes from another.
    notes = [first]
    while len(notes) < n_notes:
        same = [n for n in NOTES if NOTES[n][0] == family and n not in notes]
        other = [n for n in NOTES if n not in notes]
        notes.append(rng.choice(same if same and rng.random() < 0.6 else other))
    phrases = [rng.choice(NOTES[n][1]) for n in notes]
    fam = FAMILY[family]
    adj, image = rng.choice(fam["adj"]), rng.choice(fam["image"])
    size = rng.choice(list(SIZES))
    hours = SIZES[size]
    name = _make_name(rng, notes, family)
    vessel = rng.choice(VESSEL[size])

    opening = _opening(rng, name, phrases, adj, image)
    feel = rng.choice(fam["feel"])
    use = rng.choice([
        f"Light it {rng.choice(OCCASION)}.",
        "Perfect for {} and {}.".format(*rng.sample(fam["room"], 2)),
        f"It smells like {image}.",
        f"Made for {rng.choice(fam['room'])}, it fills the space in under an hour.",
    ])
    craft = rng.choice([
        f"Hand-poured in small batches with {rng.choice(WAX)} and {rng.choice(WICK)}.",
        f"Each candle is hand-poured with {rng.choice(WAX)} and cured for two weeks so the scent throws evenly.",
        f"We pour every candle by hand using {rng.choice(WAX)}.",
        "Vegan, cruelty-free and poured by hand.",
    ])
    burn = rng.choice([
        f"The {size} {vessel} burns for up to {hours} hours.",
        f"Poured into {_a(vessel)}, it gives about {hours} hours of slow, even burn.",
        f"Expect around {hours} hours of burn time from this {size} candle.",
        f"Our {size} {vessel} offers up to {hours} hours of fragrance.",
    ])
    gift = rng.choice([
        f"A lovely gift for {rng.choice(GIFT)}.",
        f"Wrap it up for {rng.choice(GIFT)}.",
        "The vessel is beautiful enough to keep long after the last burn.",
        "Trim the wick before every burn for the cleanest flame.",
    ])

    length = rng.choice(LENGTHS)
    if length == "short":
        body = [opening, rng.choice([feel, burn])]
    elif length == "medium":
        body = [opening, feel, rng.choice([use, craft]), burn][: rng.choice([3, 4])]
    else:
        body = [opening, feel, use, craft, burn, gift][: rng.choice([5, 6])]
    desc = " ".join(body)
    return Product(name=name, notes=notes, size=size, length=length_bucket(desc), description=desc)


def build_corpus(n_generated: int = 2400, seed: int = 7, seed_repeats: int = 3,
                 out: Path | None = CORPUS_FILE) -> list[Product]:
    """Clean the hand-written seeds, generate the catalog and write the corpus.

    Seeds are repeated a few times so the real shop voice is not drowned out.
    """
    rng = random.Random(seed)
    seeds = parse_seed_file()
    generated, seen = [], set()
    while len(generated) < n_generated:
        p = make_product(rng)
        p.description = clean_text(p.description)
        if p.description in seen:
            continue
        seen.add(p.description)
        generated.append(p)
    products = seeds * seed_repeats + generated
    rng.shuffle(products)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(p.to_record() for p in products) + "\n", encoding="utf-8")
    return products


def load_corpus(path: Path = CORPUS_FILE) -> str:
    if not path.exists():
        build_corpus(out=path)
    return path.read_text(encoding="utf-8")


def split_records(text: str) -> list[str]:
    return [r.strip() + END for r in text.split(END) if r.strip()]


if __name__ == "__main__":
    ps = build_corpus()
    print(f"wrote {len(ps)} records to {CORPUS_FILE}")

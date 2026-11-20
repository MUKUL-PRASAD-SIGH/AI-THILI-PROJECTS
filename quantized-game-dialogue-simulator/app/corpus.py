"""Build the game dialogue corpus: hand-written seed exchanges + a seeded template generator.

The setting is Emberfall, an original fantasy market town. Every exchange is two lines:

    PLAYER: <what the player says>
    <ROLE>: <the NPC's reply>

with a blank line between exchanges. Eight NPC roles each have their own topics,
vocabulary and voice (the blacksmith is gruff, the merchant haggles, the bard jokes),
so the model has to learn *who* is speaking, not just English.

The generator fills role-specific templates with slot values (items, places, monsters,
prices...). It is seeded, so ``build_corpus`` always produces the same file.
"""
from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

from app import core

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED_FILE = DATA_DIR / "raw" / "seed_dialogue.txt"
CORPUS_FILE = DATA_DIR / "corpus.txt"

MAX_LINE = 160

SHARED = {
    "place": ["the Hollow Marsh", "Cinder Peak", "the Whispering Woods", "the old mines", "the north road",
              "the river crossing", "the ruined abbey", "Ashfen Hill", "the sunken library", "the east gate"],
    "monster": ["wolves", "goblins", "a troll", "bandits", "giant spiders", "a marsh wraith", "boars",
                "a wyvern", "skeletons", "a swamp hag"],
    "coin": ["two copper", "five copper", "three silver", "eight silver", "ten silver", "twelve silver",
             "twenty silver", "one gold", "two gold"],
    "time": ["dawn", "dusk", "sundown", "midnight", "the next full moon", "tomorrow morning", "nightfall"],
    "weather": ["rain", "fog", "a cold wind", "snow", "a storm", "heavy clouds"],
    "name": ["Rhoswen", "old Tamsin", "Brannoc", "Wenna", "the miller", "Jory", "Mother Elowen", "Kitto"],
}

# Each role: slot values of its own, then intents as (player templates, reply templates).
ROLES: dict[str, dict] = {
    "BLACKSMITH": {
        "slots": {"item": ["sword", "axe", "shield", "helmet", "spear", "dagger", "mace", "chainmail"],
                  "metal": ["iron", "steel", "starsilver", "bronze", "black iron"]},
        "intents": [
            (["Can you repair my {item}?", "My {item} is broken. Can you fix it?", "Could you mend this {item}?"],
             ["Leave the {item} with me. Ready by {time}, {coin}.", "That {item} has seen worse. {coin} and it's good as new.",
              "Hmm, the {item} is cracked to the core. I'll need {metal} to fix it right."]),
            (["What does a {metal} {item} cost?", "How much for a {item}?", "Do you sell {metal} weapons?"],
             ["A {metal} {item} runs {coin}. Worth every coin when {monster} come calling.",
              "For a {item}? {coin}, and I won't haggle. The forge doesn't haggle.",
              "{metal} is scarce since the mines flooded. {coin}, take it or leave it."]),
            (["Can you make me something special?", "I want a weapon nobody else has.", "Can you forge a blade for me?"],
             ["Bring me {metal} from {place} and I'll forge something they'll sing about.",
              "Special costs special. Find me {metal} and I'll do the rest.",
              "Aye, if you bring the {metal}. The fire and the hammer are mine."]),
            (["Hello there.", "Good day, smith.", "Busy today?"],
             ["Busy every day. Speak quick, the iron's cooling.", "Mind the sparks. What do you need?",
              "Aye. If you're not buying, you're blocking my light."]),
        ],
    },
    "INNKEEPER": {
        "slots": {"drink": ["ember ale", "honey mead", "black stout", "spiced wine", "pear cider"],
                  "food": ["mutton stew", "fresh bread", "fish pie", "roast chicken", "barley soup"]},
        "intents": [
            (["Do you have a room?", "I need a bed for the night.", "How much for a room?"],
             ["A room is {coin} a night. Breakfast is {food}, if you're up before {time}.",
              "Last room's yours for {coin}. Mind the creaky stair.", "Full up, I'm afraid. Try the stable, it's dry at least."]),
            (["What's good to drink?", "Pour me a {drink}.", "What do you recommend?"],
             ["The {drink} is fresh today. {coin} a mug.", "Try the {drink}. It cures sorrow and causes singing.",
              "{drink}, and a bowl of {food} to soak it up. You look like you need both."]),
            (["Heard any rumours?", "What's the news?", "Anything strange happening?"],
             ["Folk say {monster} were seen near {place}. Nobody sensible goes there after {time}.",
              "{name} came back from {place} white as milk and won't say why.",
              "There's talk of treasure in {place}. There's always talk of treasure."]),
            (["Is there any work around here?", "Anyone need help?", "I'm looking for a job."],
             ["{name} is offering {coin} to anyone who clears the {monster} from {place}.",
              "Check the notice board. {name} has lost something in {place} again.",
              "Wash dishes for me and I'll knock the room down to {coin}."]),
        ],
    },
    "GUARD": {
        "slots": {"rule": ["no weapons drawn inside the walls", "no fires after dark", "no begging at the gate",
                           "all carts are searched"]},
        "intents": [
            (["Can I enter the town?", "Let me through.", "Is the gate open?"],
             ["State your business. The gate closes at {time}.", "Through you go. Remember, {rule}.",
              "Not until the captain says so. {monster} were sighted on {place}."]),
            (["Have you seen anything strange?", "Is the road safe?", "Any danger nearby?"],
             ["{monster} on {place}. Keep your torch lit and your blade close.",
              "Quiet so far. Quiet makes me nervous.", "We lost a patrol near {place} in the {weather}. Stay clear."]),
            (["Where can I find the captain?", "Who is in charge here?", "I need to report something."],
             ["Captain {name} is at the watchtower. Don't waste her time.",
              "Report to Captain {name}. And make it quick, she's in a mood.", "Tell me, and I'll decide if the captain needs to know."]),
            (["What are the rules here?", "Anything I should know?"],
             ["Simple. {rule}, and pay your taxes.", "{rule}. Break it and you sleep in the cells."]),
        ],
    },
    "MERCHANT": {
        "slots": {"good": ["silk", "spices", "a silver mirror", "a lantern", "maps", "a music box", "rope",
                           "a lucky charm", "dried figs", "a compass"]},
        "intents": [
            (["What are you selling?", "Show me your wares.", "Anything interesting today?"],
             ["{good}, fresh from the south! For you, only {coin}.",
              "Ah, a customer with taste! Look at this {good}. {coin}, a bargain.",
              "Today I have {good} and a story to go with it. The story is free, the {good} is {coin}."]),
            (["That's too expensive.", "Can you lower the price?", "I'll give you less."],
             ["You wound me! Fine, {coin}, and my children go hungry tonight.",
              "Too expensive? This {good} survived {monster}! {coin}, final offer.",
              "For a friend, {coin}. Do not tell the others."]),
            (["Do you buy treasure?", "Want to buy this?", "What will you give me for this?"],
             ["Let me see. Hmm, scratched, dented. {coin}, and I am being generous.",
              "Where did you find this, {place}? I'll give {coin} and ask no questions.",
              "I buy everything, my friend. Everything has a price. This one is {coin}."]),
            (["Where do you get your goods?", "Where are you from?"],
             ["A merchant who answers that soon has nothing to sell.", "From everywhere, and from {place} when the roads are kind."]),
        ],
    },
    "HEALER": {
        "slots": {"herb": ["moonpetal", "willow bark", "silverleaf", "embermoss", "frostroot"],
                  "ailment": ["a fever", "a deep cut", "spider poison", "a cough", "a broken arm"]},
        "intents": [
            (["I need healing.", "I'm hurt.", "Can you help me? I have {ailment}."],
             ["Sit. For {ailment}, a poultice of {herb} will do. Rest until {time}.",
              "Let me look. Yes, {herb} tea and no adventuring for a day.", "You're lucky. Another hour and that would have festered."]),
            (["Do you sell potions?", "What potions do you have?"],
             ["Red for wounds, blue for poison, and green only if you are very brave. {coin} each.",
              "A {herb} tonic, {coin}. Drink it slowly, it bites."]),
            (["Can I help you?", "Do you need anything?", "Is there sickness in town?"],
             ["I'm out of {herb}. It grows in {place}, if you can face the {monster}.",
              "The sickness spreads. Bring me {herb} from {place} and I can brew a cure.",
              "Just your kindness. And {herb}, if you see any."]),
        ],
    },
    "BARD": {
        "slots": {"song": ["the Ember King", "the Drowned Bell", "the Fox and the Moon", "the Last Ferry",
                           "the Goose Who Would Be Knight"]},
        "intents": [
            (["Play me a song.", "Sing something.", "Know any good songs?"],
             ["The ballad of {song}? A tale of glory, mostly. And dice.",
              "For {coin} I'll sing {song}. For {coin} more, I'll stop.", "Ah, {song}! Hum along if you know it, weep if you don't."]),
            (["Tell me a legend.", "Do you know any stories?", "What do the old tales say?"],
             ["They say {monster} once ruled {place}, until a cook with a ladle drove them out.",
              "Legend says a dragon sleeps under {place}. Cheerful, isn't it?",
              "The tale of {song}. It ends badly, but the middle is lovely."]),
            (["You're not very good.", "That song was terrible.", "Can you stop singing?"],
             ["And you are not a very good critic, yet here we both are.", "Art is never appreciated in its own time.",
              "I'll stop when the ale stops. So, never."]),
            (["Will you write a song about me?", "Am I in any songs?"],
             ["Do something worth singing about first. Defeat {monster}, perhaps.",
              "Clear {place} of {monster} and I'll rhyme your name with glory."]),
        ],
    },
    "ELDER": {
        "slots": {"relic": ["the shrine flame", "the moon bell", "the river stone", "the first lantern"]},
        "intents": [
            (["You wanted to see me?", "How can I help the town?", "Do you have a task for me?"],
             ["{relic} has gone dark. Restore it before {time}, or the spirits of {place} will wake.",
              "{monster} grow bold near {place}. Drive them back and Emberfall will remember you.",
              "Find {name}. They went to {place} three days ago and have not returned."]),
            (["Why should I help?", "What's in it for me?"],
             ["Because the road you walk ends here too, if {place} falls.", "{coin}, and the gratitude of a town. One is worth more."]),
            (["It's done.", "I've returned.", "I finished the task."],
             ["Then the town owes you. Take this, it belonged to the keeper of {relic}.",
              "Truly? Then rest. You've earned it, and {coin} besides.", "Good. But I fear {place} is not finished with us."]),
        ],
    },
    "FERRYMAN": {
        "slots": {"bank": ["the far bank", "the mill side", "the woods side", "the old dock"]},
        "intents": [
            (["Can you take me across?", "How much to cross?", "Is the ferry running?"],
             ["{coin} to {bank}. {coin} more if you talk the whole way.", "Aye, hop in. Mind the eels.",
              "Not in this {weather}. Come back at {time}."]),
            (["Is the river safe?", "What's in the water?"],
             ["Safe as anything. Keep your hands in the boat.", "Eels, mostly. And something bigger near {place}. Don't ask."]),
            (["What's on the other side?", "Where does this river go?"],
             ["{place}. Folk go in for herbs and come out with stories.", "{bank}, then {place} if you keep walking."]),
        ],
    },
}

_REPLACE = {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": "..."}


def clean_line(text: str) -> str:
    """ASCII-normalise, collapse whitespace, and keep only characters the tokenizer should see."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9 .,!?'\":;-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _fill(template: str, slots: dict[str, list[str]], rng: random.Random) -> str:
    return _cap(re.sub(r"\{(\w+)\}", lambda m: rng.choice(slots[m.group(1)]), template))


def generate_exchanges(n: int, seed: int = 0) -> list[core.Exchange]:
    """``n`` templated exchanges, spread evenly over the NPC roles."""
    rng = random.Random(seed)
    roles = sorted(ROLES)
    out = []
    for i in range(n):
        role = roles[i % len(roles)]
        spec = ROLES[role]
        slots = {**SHARED, **spec["slots"]}
        players, replies = rng.choice(spec["intents"])
        out.append(core.Exchange(role, _fill(rng.choice(players), slots, rng), _fill(rng.choice(replies), slots, rng)))
    return out


def load_seed(path: str | Path = SEED_FILE) -> list[core.Exchange]:
    text = "\n".join(l for l in Path(path).read_text(encoding="utf-8").splitlines() if not l.startswith("#"))
    return core.parse_corpus(text)


def preprocess(exchanges: list[core.Exchange]) -> list[core.Exchange]:
    """Clean every line, drop empty / over-long / malformed / duplicate exchanges."""
    seen: set[tuple[str, str, str]] = set()
    out = []
    for ex in exchanges:
        ex = core.Exchange(ex.role.upper().strip(), clean_line(ex.player), clean_line(ex.reply))
        if ex.role not in core.ROLES or not ex.player or not ex.reply:
            continue
        if len(ex.player) > MAX_LINE or len(ex.reply) > MAX_LINE:
            continue
        key = (ex.role, ex.player.lower(), ex.reply.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def build_corpus(n_generated: int = 6000, seed: int = 0, out: str | Path = CORPUS_FILE) -> dict:
    """Seed exchanges (kept first, in order) + generated ones, cleaned and de-duplicated."""
    seeds = preprocess(load_seed())
    generated = preprocess(generate_exchanges(n_generated, seed))
    seed_keys = {(e.role, e.player.lower(), e.reply.lower()) for e in seeds}
    generated = [e for e in generated if (e.role, e.player.lower(), e.reply.lower()) not in seed_keys]
    exchanges = seeds + generated
    text = core.format_corpus(exchanges)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8", newline="\n")
    return corpus_stats(exchanges, text) | {"seed_exchanges": len(seeds), "generated_exchanges": len(generated)}


def corpus_stats(exchanges: list[core.Exchange], text: str) -> dict:
    per_role: dict[str, int] = {}
    for e in exchanges:
        per_role[e.role] = per_role.get(e.role, 0) + 1
    return {"exchanges": len(exchanges), "characters": len(text), "vocab": len(set(text)),
            "per_role": dict(sorted(per_role.items()))}

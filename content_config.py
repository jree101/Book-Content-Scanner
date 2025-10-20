"""
Content Categories Configuration File - Enhanced Version
Contains all keyword lists with improved false positive filtering
"""

CONTENT_CATEGORIES = {
    "profanity": {
        "description": "Profanity, obscenity, and blasphemy",
        "enabled": True,
        "keywords": [
            # Severe profanity
            "fuck", "fucking", "fucked", "fucker", "motherfucker",
            "shit", "shitting", "shitty", "bullshit",
            "cunt", "cock", "cocksucker", "dick", "dickhead",
            "asshole", "ass hole", "arse", "arsehole",
            # Strong profanity/blasphemy
            "goddamn", "goddammit", "god damn", "jesus christ",
            "bitch", "bitches", "son of a bitch", "son-of-a-bitch",
            "bastard", "whore", "slut", "piss", "pissed",
            # Moderate profanity
            "damn", "dammit", "hell", "crap", "ass", "jackass",
            "prick", "twat", "wanker", "bollocks", "bugger", "douche",
            # Slurs (hate speech)
            "nigger", "nigga", "faggot", "fag", "retard", "retarded",
            "kike", "chink", "spic", "tranny",
        ]
    },

    "sexual_content": {
        "description": "Sexual content, nudity, and related themes",
        "enabled": True,
        "keywords": [
            # Explicit sexual content
            "sex", "sexual", "intercourse", "orgasm", "ejaculate",
            "erection", "aroused", "masturbate", "blow job", "blowjob",
            "oral sex", "penetration", "thrusting",
            # Sexual violence
            "rape", "raped", "molest", "molestation", "sexual assault",
            "grope", "groped", "pedophile", "incest", "sex slave",
            # Sexual themes
            "seduce", "seduction", "affair", "adultery", "prostitute",
            "porn", "pornography", "erotic", "lust",
            # Anatomy
            "naked", "nude", "nudity", "nipple",
            "penis", "vagina", "genitals", "testicle",
            # Immodesty  
            "lingerie", "underwear", "panties",
            "shirtless", "topless",
        ]
    },

    "violence": {
        "description": "Violence, gore, and death",
        "enabled": True,
        "keywords": [
            # Lethal violence
            "murder", "murdered", "kill", "killed", "killer",
            "assassinate", "execute", "slaughter", "massacre", "genocide",
            # Weapons
            "gun", "shoot", "shooting", "shot", "knife", "stab", "stabbed",
            "sword", "bomb", "explode", "poison",
            # Violent acts
            "torture", "mutilate", "dismember", "behead", "decapitate",
            "strangle", "choke", "suffocate", "drown", "burn", "burning",
            "electrocute", "hang", "hanged", "lynch",
            # Gore and results
            "blood", "bloody", "bleeding", "gore", "corpse", "dead body",
            "wound", "injury", "death", "dead", "died", "dying",
            "severed", "skull",
            # Physical abuse
            "beat", "beaten", "punch", "kick", "whip", "lash",
            "abuse", "assault", "violence", "brutal",
        ]
    },

    "lgbt_content": {
        "description": "LGBT themes, relationships, and identities",
        "enabled": True,
        "keywords": [
            # Relationships
            "lesbian", "homosexual", "same-sex",
            "gay couple", "lesbian couple", "gay marriage",
            "same-sex marriage", "gay pride", "coming out",
            # Identities
            "lgbt", "lgbtq", "queer", "bisexual", "pansexual",
            "transgender", "trans", "nonbinary", "non-binary",
            "genderqueer", "genderfluid", "gender identity",
            "hormone therapy",
            "pronouns", "they/them", "deadname",
            # Discrimination
            "homophobia", "transphobia", "conversion therapy",
            "misgendering",
        ]
    },

    "drugs_alcohol": {
        "description": "Drug use, alcohol consumption, and tobacco",
        "enabled": True,
        "keywords": [
            # Illegal drugs
            "cocaine", "coke", "crack", "heroin", "smack",
            "marijuana", "weed", "cannabis",
            "meth", "methamphetamine", "crystal meth",
            "lsd", "acid", "ecstasy", "molly", "mdma",
            "opium", "opioid", "oxy", "fentanyl",
            "pills", "speed", "joint", "blunt", "bong",
            # Drug use
            "snort", "inject", "shooting up",
            "stoned", "overdose",
            "addiction", "addict", "junkie", "dealer",
            # Alcohol - specific drinks and intoxication
            "whiskey", "bourbon", "vodka", "gin", "rum", "tequila",
            "liquor", "cocktail", "martini",
            "intoxicated", "hammered",
            "tipsy", "binge drinking", "hangover",
            "bar", "pub", "tavern",
            # Tobacco
            "cigarette", "cigar",
            "vape", "vaping", "e-cigarette",
        ]
    },

    "disturbing_themes": {
        "description": "Suicide, self-harm, eating disorders, and mental health",
        "enabled": True,
        "keywords": [
            # Suicide and self-harm
            "suicide", "suicidal", "kill myself", "end my life",
            "self-harm", "hang myself",
            # Eating disorders
            "anorexia", "anorexic", "bulimia", "bulimic",
            "purge", "purging", "starve", "starvation",
            # Mental health
            "depression", "anxiety", "panic attack", "ptsd",
            "schizophrenia", "psychosis", "bipolar", "trauma",
        ]
    },

    "occult": {
        "description": "Occult, witchcraft, and demonic content",
        "enabled": True,
        "keywords": [
            "witchcraft", "warlock", "sorcery", "sorcerer",
            "demon", "demons", "devil", "satanic", "satan",
            "possession", "exorcism", "black magic", "curse", "hex",
            "necromancy", "summon", "seance", "ouija",
        ]
    }
}

# ENHANCED Figurative language patterns that should NOT be flagged
FIGURATIVE_EXCLUSIONS = [
    # Death/violence figurative expressions
    r"pale as death",
    r"scared to death",
    r"bored to death",
    r"worried to death",
    r"frozen to death",
    r"starved to death",
    r"death of (the|a) (author|artist|era|movement)",
    r"death\s+(tax|rate|certificate|penalty|row)",
    r"kill(ed)?\s+time",
    r"kill(ed)?\s+two birds",
    r"dying to (know|see|meet|hear|tell)",
    r"to die for",

    # High/drunk with alternative meanings
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand)\s+(feet|meters|miles|inches)\s+high",
    r"high\s+(above|over|on|in the|up|ceiling|wall|roof|mountain|tower|building|cliff)",
    r"(looming|rising|soaring|towering|standing|reaching|flying|climbed|climbs)\s+high",
    r"high\s+(king|queen|prince|princess|priest|court|council|office|rank|status|regard|esteem)",
    r"high\s+(treason|crime|tea|noon|seas|tide|water|ground|speed|quality|standard)",
    r"most high",
    r"on high",
    r"\d+\s+high",

    # Drunk/drinking normal contexts
    r"(had|have|after we|they)\s+drunk\b",  # Past tense of drink
    r"drinking\s+(water|tea|coffee|milk|juice|from|it all in)",
    r"drinking\s+(vessel|cup|glass|container)",
    r"drunken\s+(surrealist|artist|style|dance)",

    # Miscellaneous
    r"anti[–-]climax",
    r"wine[–-]colored",
    r"wine\s+cellar",
    r"drug\s+store",
    r"machine[–-]gun\s+(battalion|unit|division|company)",
    r"pot\s+of\s+(coffee|tea|gold)",
    r"pot\s+luck",
    r"hash\s+(browns|arrived|table|function|tag)",
    r"gay\s+(parties|time|colors|mood)",
    r"transition\s+(period|phase|time|from|to|into|between)",
    r"breast\s+of\s+(chicken|turkey|the (new )?world|hill|mountain)",
    r"breast\s+(cancer|feeding|milk|pump|exam)",
    r"it'?s\s+a\s+bitch.*dog",
    r"female\s+dog",
    r"cock\s+and\s+bull",
    r"weather[–-]beaten",
    r"beaten\s+(path|track)",
    r"pussy\s+(cat|willow)",
    r"pussy\s+foot",

    # Bounce/high (physics/movement)
    r"bounce\s+high",
    r"leap(ed)?\s+high",
    r"jump(ed)?\s+high",
    r"flew\s+high",
    r"toss(ed)?\s+high",

    # Speed as velocity
    r"at\s+(top|full|great|high|low)\s+speed",
    r"speed\s+(up|down|of|limit)",
    r"\d+\s+mph",

    # Smoke as visible vapor
    r"smoke\s+(from|coming|rising|billowing|signal|detector|alarm)",
    r"smoking\s+(gun|jacket|room|area|section)",

    # Wasted as squandered
    r"wasted\s+(time|effort|energy|money|resources|opportunity)",
    r"no\s+time\s+wasted",

    # Affair as event
    r"(colossal|grand|state|family|public|private|business|political)\s+affair",
    r"affair\s+of\s+(state|honor|the heart)",

    # Desire/aroused in non-sexual contexts
    r"desire\s+(to|for)\s+(knowledge|freedom|peace|truth|understanding)",
    r"aroused\s+(curiosity|suspicion|interest|concern)",

    # Shot as past tense
    r"shot\s+(up|down|across|through|into|towards)",
    r"shot\s+(a|the)\s+(arrow|bullet|photo|picture|scene)",

    # Crack as split/joke
    r"crack\s+(in|of|on|open|the code|a smile|a joke)",
    r"crack\s+of\s+(dawn|thunder|whip)",
]

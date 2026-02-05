"""
Wikipedia Knowledge Module for WSD
===================================
Provides Wikipedia integration for enhanced word sense disambiguation.
Fetches article summaries and calculates knowledge overlap scores.
"""

import re
import os
import json
import hashlib
import requests
from typing import Optional, Dict, List, Tuple
from func_timeout import func_timeout, FunctionTimedOut

# Cache directory for Wikipedia results
CACHE_DIR = "./wiki_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_key(word: str) -> str:
    """Generate cache key for a word."""
    return hashlib.md5(word.lower().strip().encode()).hexdigest()

def _load_cache(word: str) -> Optional[Dict]:
    """Load cached Wikipedia result."""
    cache_file = os.path.join(CACHE_DIR, f"{_cache_key(word)}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None

def _save_cache(word: str, data: Dict):
    """Save Wikipedia result to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{_cache_key(word)}.json")
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    except:
        pass

# ============================================================================
# CONTEXT DETECTION - For disambiguating words based on sentence context
# ============================================================================

# Keywords that indicate programming/coding context
PROGRAMMING_KEYWORDS = [
    'code', 'coding', 'programming', 'program', 'software', 'developer', 'development',
    'function', 'method', 'variable', 'loop', 'syntax', 'compile', 'compiler',
    'script', 'scripting', 'debug', 'debugging', 'algorithm', 'data structure',
    'object', 'instance', 'constructor', 'inheritance', 'polymorphism', 'encapsulation',
    'api', 'library', 'framework', 'module', 'package', 'import', 'define', 'defines',
    'object-oriented', 'oop', 'ide', 'editor', 'terminal', 'command line',
    'backend', 'frontend', 'fullstack', 'web development', 'app development',
    'machine learning', 'ml', 'ai', 'artificial intelligence', 'neural network',
    'database', 'sql', 'query', 'server', 'client', 'http', 'rest', 'json', 'xml',
    'git', 'github', 'version control', 'repository', 'commit', 'branch', 'merge',
    'exception', 'error handling', 'try', 'catch', 'throw', 'return', 'print',
    'array', 'list', 'dictionary', 'tuple', 'set', 'string', 'integer', 'float', 'boolean',
    'if statement', 'for loop', 'while loop', 'switch', 'case', 'break',
    'selenium', 'pytest', 'unittest', 'django', 'flask', 'react', 'angular', 'vue',
    'tensorflow', 'pytorch', 'pandas', 'numpy', 'scipy', 'matplotlib'
]

# Keywords that indicate tech company context (tech-specific, not generic business)
TECH_COMPANY_KEYWORDS = [
    'launched', 'iphone', 'ipad', 'macbook', 'airpods', 'apple watch',
    'google', 'microsoft', 'amazon prime', 'facebook', 'meta', 'twitter',
    'samsung', 'tesla', 'spacex', 'nvidia', 'intel', 'amd',
    'android', 'ios', 'windows', 'macos', 'chromebook', 'pixel',
    'silicon valley', 'tech giant', 'big tech', 'trillion dollar',
    'tim cook', 'elon musk', 'mark zuckerberg', 'sundar pichai', 'satya nadella'
]

# Keywords that indicate biology/nature context
BIOLOGY_KEYWORDS = [
    'animal', 'species', 'habitat', 'wildlife', 'zoo', 'nature', 'ecosystem',
    'reptile', 'mammal', 'bird', 'insect', 'snake', 'predator', 'prey',
    'forest', 'jungle', 'wild', 'bite', 'venom', 'scales', 'tail', 'burrow'
]

# Keywords that indicate finance/banking context
FINANCE_KEYWORDS = [
    'money', 'deposit', 'withdraw', 'savings', 'account', 'loan', 'interest',
    'mortgage', 'credit', 'debit', 'transaction', 'balance', 'atm', 'bank account',
    'financial', 'investment', 'stocks', 'bonds', 'portfolio'
]

# Keywords that indicate food/eating context
FOOD_KEYWORDS = [
    'ate', 'eat', 'eating', 'food', 'fruit', 'vegetable', 'delicious', 'tasty',
    'cook', 'cooking', 'recipe', 'meal', 'breakfast', 'lunch', 'dinner', 'snack',
    'hungry', 'bite', 'chew', 'swallow', 'taste', 'flavor', 'sweet', 'sour',
    'ripe', 'fresh', 'organic', 'healthy', 'nutritious', 'diet', 'juice',
    'pie', 'salad', 'dessert', 'bake', 'baking', 'kitchen', 'plate', 'bowl',
    'orchard', 'farm', 'harvest', 'grow',
    'seed', 'skin', 'peel', 'slice', 'chop', 'blend', 'smoothie'
]

# Keywords that indicate entertainment/viewing context
ENTERTAINMENT_KEYWORDS = [
    'tv', 'television', 'movie', 'film', 'show', 'series', 'episode', 'channel',
    'netflix', 'youtube', 'stream', 'streaming', 'video', 'cinema', 'theater',
    'broadcast', 'programme', 'program', 'documentary', 'news', 'sports',
    'viewing', 'viewer', 'audience', 'screen', 'remote', 'couch', 'sofa',
    'night', 'evening', 'weekend', 'binge', 'marathon', 'premiere', 'season',
    'actor', 'actress', 'director', 'starring', 'cast', 'scene', 'plot',
    'comedy', 'drama', 'thriller', 'horror', 'action', 'romance', 'cartoon',
    'anime', 'sitcom', 'reality', 'game show', 'talk show', 'late night'
]

# Keywords that indicate timepiece/clock context
TIMEPIECE_KEYWORDS = [
    'wrist', 'wristwatch', 'clock', 'time', 'hour', 'minute', 'second',
    'digital', 'analog', 'strap', 'band', 'dial', 'face', 'hands',
    'wearing', 'wore', 'timer', 'stopwatch', 'alarm', 'bezel',
    'luxury', 'rolex', 'casio', 'seiko', 'omega', 'jewelry', 'accessory'
]

# Keywords that indicate observation/surveillance/guarding context
OBSERVATION_KEYWORDS = [
    'guard', 'security', 'monitor', 'monitoring', 'surveillance', 'patrol',
    'building', 'house', 'property', 'premises', 'door', 'entrance', 'gate',
    'protect', 'protection', 'keep an eye', 'lookout', 'alert', 'careful',
    'observe', 'observing', 'observation', 'supervise', 'supervision',
    'oversee', 'inspect', 'check', 'survey', 'scout', 'spy',
    'child', 'children', 'kids', 'baby', 'toddler', 'babysit', 'babysitting',
    'pet', 'dog', 'cat', 'prisoner', 'suspect', 'criminal',
    'night shift', 'duty', 'post', 'station', 'sentry', 'vigilant'
]

# Keywords that indicate fitness/exercise/sports context
FITNESS_KEYWORDS = [
    'morning', 'jog', 'jogging', 'exercise', 'workout', 'marathon', 'sprint',
    'gym', 'fitness', 'athletic', 'athlete', 'training', 'cardio', 'aerobic',
    'mile', 'kilometer', 'distance', 'race', 'racing', 'track', 'field',
    'running shoes', 'sneakers', 'stretching', 'warm up', 'cool down',
    'healthy', 'health', 'sweat', 'stamina', 'endurance', 'pace', 'speed',
    'treadmill', 'outdoor', 'park', 'trail', 'route', 'lap', 'finish line',
    'goes for', 'went for', 'take a', 'daily', 'routine', 'regularly'
]

# Keywords that indicate business/management/organization context
BUSINESS_KEYWORDS = [
    'successfully', 'manager', 'managing', 'management', 'ceo', 'director',
    'led', 'lead', 'leading', 'founder', 'founded', 'owner', 'ownership',
    'organization', 'organisation', 'corporation', 'enterprise', 'firm',
    'business', 'startup', 'employee', 'staff', 'team', 'department',
    'profit', 'revenue', 'growth', 'expand', 'expansion', 'strategy',
    'board', 'executive', 'operations', 'administered', 'oversaw',
    'headed', 'supervised', 'controlled', 'governed', 'steered'
]

# Keywords that indicate emotion/physical sensation context
EMOTION_KEYWORDS = [
    'tears', 'tear', 'crying', 'cry', 'sob', 'sobbing', 'weep', 'weeping',
    'cheek', 'cheeks', 'emotion', 'emotional',
    'sad', 'sadness', 'happy', 'happiness', 'joy', 'grief', 'sorrow',
    'pain', 'hurt', 'heartbreak', 'heartbroken', 'moved', 'touched',
    'down her', 'down his', 'down my', 'down the', 'began to', 'started to',
    'flow', 'flowing', 'drip', 'dripping', 'trickle'
]

# Keywords that indicate computer/digital context
COMPUTER_KEYWORDS = [
    'upload', 'download', 'document', 'folder', 'directory', 'save', 'open',
    'click', 'drag', 'drop', 'attach', 'attachment', 'email', 'send',
    'computer', 'laptop', 'desktop', 'storage', 'disk', 'drive', 'usb',
    'pdf', 'word', 'excel', 'image', 'photo', 'video', 'audio', 'mp3', 'mp4',
    'zip', 'compress', 'extract', 'rename', 'delete', 'copy', 'paste',
    'share', 'transfer', 'submit', 'format', 'extension'
]

# Keywords that indicate legal/court context
LEGAL_KEYWORDS = [
    'lawyer', 'attorney', 'court', 'judge', 'trial', 'case', 'lawsuit',
    'legal', 'law', 'filed', 'filing', 'petition', 'motion', 'hearing',
    'plaintiff', 'defendant', 'prosecution', 'defense', 'verdict', 'judgment',
    'appeal', 'testimony', 'witness', 'evidence', 'affidavit', 'subpoena',
    'litigation', 'settlement', 'damages', 'claim', 'complaint', 'injunction',
    'magistrate', 'barrister', 'solicitor', 'paralegal', 'notary', 'oath',
    'police', 'thief', 'arrest', 'arrested', 'crime', 'criminal', 'accused', 'suspect'
]

# Keywords that indicate tools/crafts/manufacturing context
TOOLS_KEYWORDS = [
    'wood', 'smooth', 'smoothen', 'smoothened', 'grind', 'grinding',
    'sand', 'sanding', 'polish', 'polishing', 'shape', 'shaping', 'sharpen',
    'workshop', 'workbench', 'tool', 'tools', 'hand tool', 'rasp', 'chisel',
    'carpenter', 'carpentry', 'metalwork', 'blacksmith', 'forge', 'craft',
    'edge', 'edges', 'rough', 'surface', 'material', 'iron', 'steel', 'brass',
    'nail', 'screw', 'bolt'
]

# Keywords that indicate season/time of year context
SEASON_KEYWORDS = [
    'flowers', 'flower', 'bloom', 'blooming', 'blossom', 'blossoming',
    'summer', 'autumn', 'fall', 'winter', 'seasonal', 'season',
    'weather', 'warm', 'cold', 'sunny', 'rainy', 'temperature',
    'months', 'march', 'april', 'may', 'june', 'september', 'october',
    'garden', 'gardening', 'planting', 'seeds', 'nature', 'trees',
    'birds', 'butterflies', 'allergies', 'pollen', 'year'
]

# Keywords that indicate water/hydrology context
WATER_KEYWORDS = [
    'water', 'flows', 'flow', 'flowing', 'river', 'stream', 'creek',
    'lake', 'pond', 'well', 'underground', 'aquifer', 'source',
    'drink', 'drinking', 'fresh', 'mineral', 'natural', 'bubbling',
    'fountain', 'hot springs', 'thermal', 'geothermal', 'geyser',
    'bottle', 'bottled', 'pure', 'clean', 'clear'
]

# Keywords that indicate mechanical/device context
MECHANICAL_KEYWORDS = [
    'toy', 'toys', 'coil', 'bounce', 'bouncing', 'elastic',
    'mechanism', 'mechanical', 'device', 'mattress', 'bed',
    'suspension', 'shock absorber', 'tension', 'compress',
    'compressed', 'stretch', 'stretched', 'force', 'pressure', 'push', 'pull',
    'jump', 'jumping', 'trampoline', 'pen', 'button', 'loaded'
]

# Keywords that indicate construction/machinery context
CONSTRUCTION_KEYWORDS = [
    'lifted', 'lifting', 'lift', 'heavy', 'load', 'loading', 'container', 'containers',
    'construction', 'site', 'building', 'tower', 'tall', 'height',
    'equipment', 'machinery', 'operator', 'hoist', 'hook', 'cable', 'wire',
    'cargo', 'shipyard', 'port', 'dock', 'warehouse', 'factory',
    'move', 'moving', 'transport', 'haul', 'weight', 'tons', 'industrial'
]

# Keywords that indicate bird/wildlife context
BIRD_KEYWORDS = [
    'flew', 'fly', 'flying', 'flight', 'wings', 'wing', 'feathers', 'feather',
    'nest', 'nesting', 'eggs', 'beak', 'migrate', 'migration', 'migratory',
    'lake', 'pond', 'wetland', 'marsh', 'swamp', 'habitat',
    'bird', 'birds', 'avian', 'flock', 'soar', 'soaring', 'glide', 'graceful',
    'wildlife', 'nature', 'sanctuary', 'endangered', 'species'
]

# Keywords that indicate electrical/battery context
ELECTRICAL_KEYWORDS = [
    'phone', 'battery', 'batteries', 'plug', 'plugged', 'charger', 'charging',
    'power', 'electric', 'electrical', 'outlet', 'socket', 'usb', 'cable',
    'laptop', 'device', 'wireless', 'adapter', 'volt', 'voltage', 'amp',
    'dead', 'low', 'full', 'percentage', 'rechargeable', 'lithium'
]

# Keywords that indicate payment/cost context
PAYMENT_KEYWORDS = [
    'service', 'fee', 'fees', 'cost', 'price', 'pay', 'payment', 'free',
    'no charge', 'extra', 'additional', 'bill', 'invoice', 'receipt',
    'discount', 'rate', 'flat rate', 'per hour', 'monthly', 'annual',
    'subscription', 'membership', 'premium', 'basic', 'refund'
]

# Keywords that indicate military/attack context
MILITARY_KEYWORDS = [
    'soldiers', 'soldier', 'army', 'troops', 'military', 'battle', 'war',
    'forward', 'attack', 'attacking', 'advance', 'advancing', 'rush', 'rushing',
    'enemy', 'combat', 'fight', 'fighting', 'battlefield', 'front line',
    'cavalry', 'infantry', 'retreat', 'assault', 'offensive', 'defense',
    'began to', 'started to', 'ordered to', 'commanded'
]

# Keywords that indicate writing/message context
WRITING_KEYWORDS = [
    'wrote', 'write', 'writing', 'written', 'letter', 'message', 'memo',
    'paper', 'pen', 'pencil', 'jot', 'jotted', 'scribble', 'scribbled',
    'sticky', 'post-it', 'reminder', 'journal', 'diary', 'notebook',
    'left a', 'leave a', 'send a', 'passed a', 'handed a', 'read a'
]

# Keywords that indicate music/sound context
MUSIC_KEYWORDS = [
    'musical', 'music', 'song', 'songs', 'melody', 'tune',
    'sing', 'singing', 'sang', 'instrument',
    'piano', 'guitar', 'violin', 'flute', 'orchestra', 'choir',
    'high note', 'low note', 'flat note', 'sharp note', 'scale', 'octave',
    'sound', 'sounds', 'tone', 'tones', 'frequency', 'high pitch', 'low pitch'
]

# Keywords that indicate education/school context
EDUCATION_KEYWORDS = [
    'math', 'mathematics', 'science', 'history', 'english', 'physics', 'chemistry',
    'biology', 'geography', 'economics', 'literature', 'school', 'college', 'university',
    'teacher', 'professor', 'student', 'students', 'classroom', 'lecture', 'lesson',
    'exam', 'test', 'homework', 'assignment', 'grade', 'grades', 'semester', 'course'
]

# Keywords that indicate currency/money context
CURRENCY_KEYWORDS = [
    '₹', 'rupee', 'rupees', 'dollar', 'dollars', '$', 'euro', 'euros', '€',
    'pound', 'pounds', '£', 'yen', '¥', 'cash', 'money', 'currency',
    'banknote', 'bill', 'bills', '100', '500', '1000', '2000', '50', '20',
    'gave me', 'handed me', 'paid', 'change', 'wallet', 'pocket', 'purse'
]

# Keywords that indicate industrial/factory context
INDUSTRIAL_KEYWORDS = [
    'factory', 'factories', 'power', 'manufacturing', 'production', 'assembly',
    'nuclear', 'thermal', 'electricity', 'generator', 'turbine', 'energy',
    'industrial', 'industry', 'processing', 'refinery', 'chemical', 'steel',
    'cement', 'textile', 'automobile', 'machinery', 'facility', 'facilities'
]

# Keywords that indicate botany/vegetation context
BOTANY_KEYWORDS = [
    'watered', 'water', 'watering', 'grow', 'growing', 'grew', 'growth',
    'flower', 'flowers', 'flowering', 'leaf', 'leaves', 'root', 'roots',
    'soil', 'pot', 'potted', 'garden', 'gardening', 'greenhouse', 'sunlight',
    'seed', 'seeds', 'stem', 'branch', 'branches', 'tree', 'trees', 'shrub',
    'green', 'vegetation', 'photosynthesis', 'fertilizer', 'indoor', 'outdoor'
]

# Keywords that indicate spy/undercover context
SPY_KEYWORDS = [
    'spy', 'spies', 'spying', 'undercover', 'secret', 'secrets', 'agent',
    'infiltrate', 'infiltrated', 'infiltration', 'mole', 'double agent',
    'insider', 'informant', 'informer', 'traitor', 'betrayal',
    'organization', 'gang', 'cartel', 'intelligence', 'cia', 'fbi',
    'mission', 'covert', 'operation', 'surveillance', 'planted'
]

# Keywords that indicate sports/athletics context
SPORTS_KEYWORDS = [
    'ball', 'balls', 'pitched', 'throw', 'throwing', 'threw', 'catch', 'catching',
    'baseball', 'cricket', 'bowling', 'bowled', 'batter', 'batsman', 'wicket',
    'game', 'games', 'match', 'matches', 'player', 'players', 'team', 'teams',
    'stadium', 'field', 'innings', 'score', 'runs', 'home run', 'strike', 'out',
    'sport', 'sports', 'athletic', 'athlete', 'coach', 'practice'
]

# Keywords that indicate sales/business presentation context
SALES_KEYWORDS = [
    'sales', 'impressive', 'presentation', 'client', 'clients', 'customer', 'customers',
    'business', 'deal', 'deals', 'proposal', 'marketing', 'advertising', 'product',
    'convince', 'persuade', 'meeting', 'investor', 'investors', 'startup', 'venture',
    'elevator pitch', 'shark tank', 'funding', 'investment', 'sell', 'selling'
]

# Keywords that indicate terrain/ground context
TERRAIN_KEYWORDS = [
    'tent', 'tents', 'flat', 'ground', 'camping', 'camp', 'campsite',
    'set up', 'setup', 'level', 'even', 'uneven', 'slope', 'sloped',
    'grass', 'grassy', 'outdoor', 'outdoors', 'terrain', 'surface',
    'football pitch', 'soccer pitch', 'cricket pitch', 'playing field'
]

# Keywords that indicate social class/hierarchy context
SOCIAL_KEYWORDS = [
    'upper class', 'lower class', 'middle class', 'working class', 'upper', 'lower',
    'wealthy', 'rich', 'poor', 'poverty', 'elite', 'aristocrat', 'aristocracy',
    'noble', 'nobility', 'royal', 'royalty', 'commoner', 'peasant', 'bourgeois',
    'status', 'hierarchy', 'society', 'belongs to', 'born into', 'privilege'
]

# Keywords that indicate insect/creature context
INSECT_KEYWORDS = [
    'crawling', 'crawl', 'crawled', 'wall', 'floor', 'ceiling', 'window',
    'ant', 'ants', 'spider', 'spiders', 'beetle', 'cockroach', 'fly', 'flies',
    'mosquito', 'butterfly', 'moth', 'insect', 'insects', 'pest', 'pests',
    'legs', 'wings', 'antenna', 'bite', 'bitten', 'sting', 'stung', 'squash'
]

# Keywords that indicate surveillance/spy device context
SURVEILLANCE_KEYWORDS = [
    'hidden', 'microphone', 'wiretap', 'listening', 'recording', 'secretly',
    'planted', 'device', 'spy', 'spying', 'surveillance', 'eavesdrop', 'tap',
    'room', 'office', 'phone', 'conversation', 'detected', 'sweep', 'found'
]

# Keywords that indicate fashion/modeling context
FASHION_KEYWORDS = [
    'fashion', 'runway', 'ramp', 'photoshoot', 'photo shoot', 'photographer',
    'pose', 'posing', 'beautiful', 'gorgeous', 'supermodel', 'catwalk',
    'magazine', 'vogue', 'designer', 'modeling', 'modelling', 'agency',
    'portfolio', 'commercial', 'advertisement', 'ad', 'campaign'
]

# Keywords that indicate product/vehicle context
PRODUCT_KEYWORDS = [
    'car', 'cars', 'vehicle', 'vehicles', 'automobile', 'bike', 'motorcycle',
    'new model', 'latest', 'version', 'year', 'brand', 'make', 'manufacturer',
    'features', 'specs', 'specifications', 'engine', 'horsepower', 'mileage',
    'release', 'launched', 'introduced', 'upgraded', 'improved', 'design'
]




# Mapping of ambiguous words to their context-specific Wikipedia search terms
CONTEXT_SEARCH_MAPPINGS = {
    'python': {
        'programming': ['Python (programming language)', 'Python programming'],
        'biology': ['Python (genus)', 'Pythonidae snake']
        # No default - use plain word if no context detected
    },
    'java': {
        'programming': ['Java (programming language)', 'Java software platform'],
        'geography': ['Java', 'Java island']
        # No default - use plain word if no context detected
    },
    'watch': {
        'entertainment': ['Television', 'Watching television', 'Viewer (television)'],
        'observation': ['Observation', 'Surveillance', 'Security guard'],
        'timepiece': ['Watch', 'Wristwatch', 'Timepiece']
        # No default - context determines meaning
    },
    'run': {
        'fitness': ['Running', 'Jogging', 'Exercise'],
        'programming': ['Execution (computing)', 'Run command', 'Computer program execution'],
        'business': ['Management', 'Business operations', 'Corporate governance'],
        'emotion': ['Crying', 'Tears', 'Weeping']
        # No default - context determines meaning
    },
    'ran': {
        'fitness': ['Running', 'Jogging', 'Exercise'],
        'programming': ['Execution (computing)', 'Run command', 'Computer program execution'],
        'business': ['Management', 'Business operations', 'Corporate governance'],
        'emotion': ['Crying', 'Tears', 'Weeping']
        # No default - context determines meaning
    },
    'company': {
        'business': ['Company', 'Business organization', 'Corporation'],
        'tech_company': ['Technology company', 'Tech company']
        # No default - context determines meaning
    },
    'file': {
        'computer': ['Computer file', 'Digital file', 'File (computing)'],
        'legal': ['Legal filing', 'Court filing', 'File (legal)'],
        'tools': ['File (tool)', 'Hand file', 'Metalworking file']
        # No default - context determines meaning
    },
    'mouse': {
        'computer': ['Computer mouse', 'Mouse (computing)', 'Input device'],
        'biology': ['Mouse', 'House mouse', 'Mus musculus']
        # No default - context determines meaning
    },
    'spring': {
        'season': ['Spring (season)', 'Springtime', 'Spring season'],
        'water': ['Spring (hydrology)', 'Natural spring', 'Water spring'],
        'mechanical': ['Spring (device)', 'Coil spring', 'Mechanical spring']
        # No default - context determines meaning
    },
    'crane': {
        'construction': ['Crane (machine)', 'Construction crane', 'Tower crane'],
        'bird': ['Crane (bird)', 'Gruidae', 'Crane bird']
        # No default - context determines meaning
    },
    'charge': {
        'legal': ['Criminal charge', 'Legal charge', 'Indictment'],
        'electrical': ['Battery charging', 'Electric charge', 'Charging battery'],
        'payment': ['Fee', 'Service charge', 'Price'],
        'military': ['Charge (warfare)', 'Military charge', 'Cavalry charge']
        # No default - context determines meaning
    },
    'note': {
        'writing': ['Note (typography)', 'Written note', 'Memorandum'],
        'music': ['Musical note', 'Note (music)', 'Pitch (music)'],
        'currency': ['Banknote', 'Currency note', 'Paper money']
        # No default - context determines meaning
    },
    'plant': {
        'industrial': ['Power plant', 'Industrial plant', 'Factory'],
        'botany': ['Plant', 'Flowering plant', 'Houseplant'],
        'spy': ['Sleeper agent', 'Undercover agent', 'Mole (espionage)']
        # No default - context determines meaning
    },
    'pitch': {
        'sports': ['Pitch (baseball)', 'Pitching (baseball)', 'Bowling (cricket)'],
        'sales': ['Sales pitch', 'Elevator pitch', 'Business pitch'],
        'terrain': ['Pitch (sports field)', 'Football pitch', 'Playing field'],
        'music': ['Pitch (music)', 'Audio frequency', 'Sound pitch']
        # No default - context determines meaning
    },
    'class': {
        'programming': ['Class (computer programming)', 'Object-oriented programming class'],
        'education': ['Class (education)', 'School class', 'Classroom'],
        'social': ['Social class', 'Class system', 'Social stratification']
        # No default - context determines meaning
    },
    'bug': {
        'programming': ['Software bug', 'Bug (software)', 'Programming error'],
        'insect': ['Insect', 'Bug (insect)', 'True bugs'],
        'surveillance': ['Covert listening device', 'Wiretap', 'Surveillance device']
        # No default - context determines meaning
    },
    'model': {
        'programming': ['Machine learning model', 'AI model', 'Statistical model'],
        'fashion': ['Model (person)', 'Fashion model', 'Supermodel'],
        'product': ['Model (product)', 'Product model', 'Vehicle model']
        # No default - context determines meaning
    },
    'object': {
        'programming': ['Object (computer science)', 'Object-oriented programming'],
        'default': ['Object (computer science)']
    },
    'function': {
        'programming': ['Function (computer programming)', 'Subroutine'],
        'math': ['Function (mathematics)'],
        'default': ['Function (computer programming)']
    },
    'method': {
        'programming': ['Method (computer programming)', 'Object-oriented method'],
        'default': ['Method (computer programming)']
    },
    'variable': {
        'programming': ['Variable (computer science)', 'Programming variable'],
        'math': ['Variable (mathematics)'],
        'default': ['Variable (computer science)']
    },
    'string': {
        'programming': ['String (computer science)', 'Character string'],
        'music': ['String instrument', 'Guitar string'],
        'default': ['String (computer science)']
    },
    'array': {
        'programming': ['Array (data structure)', 'Array data type'],
        'default': ['Array (data structure)']
    },
    'loop': {
        'programming': ['Loop (programming)', 'Control flow loop'],
        'default': ['Loop (programming)']
    },
    'inheritance': {
        'programming': ['Inheritance (object-oriented programming)', 'OOP inheritance'],
        'default': ['Inheritance (object-oriented programming)']
    },
    'interface': {
        'programming': ['Interface (computing)', 'Protocol (object-oriented programming)'],
        'default': ['Interface (computing)']
    },
    'module': {
        'programming': ['Module (programming)', 'Modular programming'],
        'default': ['Module (programming)']
    },
    'package': {
        'programming': ['Package (programming)', 'Software package'],
        'default': ['Package (programming)']
    },
    'exception': {
        'programming': ['Exception handling', 'Exception (computer programming)'],
        'default': ['Exception handling']
    },
    'constructor': {
        'programming': ['Constructor (object-oriented programming)', 'Class constructor'],
        'default': ['Constructor (object-oriented programming)']
    },
    'instance': {
        'programming': ['Instance (computer science)', 'Object instance'],
        'default': ['Instance (computer science)']
    },
    'pointer': {
        'programming': ['Pointer (computer programming)', 'Memory pointer'],
        'default': ['Pointer (computer programming)']
    },
    'stack': {
        'programming': ['Stack (abstract data type)', 'Call stack'],
        'default': ['Stack (abstract data type)']
    },
    'queue': {
        'programming': ['Queue (abstract data type)', 'FIFO queue'],
        'default': ['Queue (abstract data type)']
    },
    'tree': {
        'programming': ['Tree (data structure)', 'Binary tree'],
        'biology': ['Tree', 'Woody plant'],
        'default': ['Tree (data structure)']
    },
    'node': {
        'programming': ['Node (computer science)', 'Data structure node'],
        'default': ['Node (computer science)']
    },
    'graph': {
        'programming': ['Graph (abstract data type)', 'Graph theory'],
        'math': ['Graph (discrete mathematics)'],
        'default': ['Graph (abstract data type)']
    },
    'apple': {
        'tech_company': ['Apple Inc.', 'Apple (company)'],
        'food': ['Apple', 'Apple fruit'],
        'biology': ['Apple', 'Apple fruit']
        # No default - context determines meaning
    },
    'amazon': {
        'tech_company': ['Amazon (company)', 'Amazon.com'],
        'geography': ['Amazon River', 'Amazon rainforest']
        # No default - context determines meaning
    },
    'oracle': {
        'programming': ['Oracle Corporation', 'Oracle Database'],
        'default': ['Oracle Corporation']
    },
    'ruby': {
        'programming': ['Ruby (programming language)'],
        'default': ['Ruby (programming language)']
    },
    'rust': {
        'programming': ['Rust (programming language)'],
        'default': ['Rust (programming language)']
    },
    'swift': {
        'programming': ['Swift (programming language)'],
        'default': ['Swift (programming language)']
    },
    'go': {
        'programming': ['Go (programming language)'],
        'default': ['Go (programming language)']
    },
    'scala': {
        'programming': ['Scala (programming language)'],
        'default': ['Scala (programming language)']
    },
    'kotlin': {
        'programming': ['Kotlin (programming language)'],
        'default': ['Kotlin (programming language)']
    },
    'c': {
        'programming': ['C (programming language)'],
        'default': ['C (programming language)']
    },
    'r': {
        'programming': ['R (programming language)'],
        'default': ['R (programming language)']
    },
    'dart': {
        'programming': ['Dart (programming language)'],
        'default': ['Dart (programming language)']
    },
    'shell': {
        'programming': ['Shell (computing)', 'Unix shell', 'Command-line interface'],
        'biology': ['Shell (biology)', 'Seashell'],
        'default': ['Shell (computing)']
    },
    'bash': {
        'programming': ['Bash (Unix shell)', 'Bourne Again Shell'],
        'default': ['Bash (Unix shell)']
    },
    'script': {
        'programming': ['Scripting language', 'Script (computing)'],
        'default': ['Scripting language']
    },
    'library': {
        'programming': ['Library (computing)', 'Software library'],
        'default': ['Library (computing)']
    },
    'framework': {
        'programming': ['Software framework', 'Web framework'],
        'default': ['Software framework']
    },
    'compiler': {
        'programming': ['Compiler', 'Source code compiler'],
        'default': ['Compiler']
    },
    'interpreter': {
        'programming': ['Interpreter (computing)', 'Programming interpreter'],
        'default': ['Interpreter (computing)']
    },
    'runtime': {
        'programming': ['Runtime system', 'Runtime environment'],
        'default': ['Runtime system']
    },
    'thread': {
        'programming': ['Thread (computing)', 'Execution thread'],
        'default': ['Thread (computing)']
    },
    'process': {
        'programming': ['Process (computing)', 'Computer process'],
        'default': ['Process (computing)']
    },
    'socket': {
        'programming': ['Network socket', 'Socket (computing)'],
        'default': ['Network socket']
    },
    'port': {
        'programming': ['Port (computer networking)', 'Network port'],
        'default': ['Port (computer networking)']
    },
    'protocol': {
        'programming': ['Communications protocol', 'Network protocol'],
        'default': ['Communications protocol']
    },
    'api': {
        'programming': ['API', 'Application programming interface'],
        'default': ['API']
    },
    'sdk': {
        'programming': ['Software development kit', 'SDK'],
        'default': ['Software development kit']
    },
    'ide': {
        'programming': ['Integrated development environment', 'IDE'],
        'default': ['Integrated development environment']
    },
    'bug': {
        'programming': ['Software bug', 'Computer bug'],
        'biology': ['Insect', 'Bug (insect)'],
        'default': ['Software bug']
    },
    'patch': {
        'programming': ['Patch (computing)', 'Software patch'],
        'default': ['Patch (computing)']
    },
    'branch': {
        'programming': ['Branching (version control)', 'Git branch'],
        'biology': ['Branch (botany)', 'Tree branch'],
        'default': ['Branching (version control)']
    },
    'merge': {
        'programming': ['Merge (version control)', 'Git merge'],
        'default': ['Merge (version control)']
    },
    'commit': {
        'programming': ['Commit (version control)', 'Git commit'],
        'default': ['Commit (version control)']
    },
    'repository': {
        'programming': ['Repository (version control)', 'Software repository'],
        'default': ['Repository (version control)']
    },
    'container': {
        'programming': ['Container (computing)', 'Docker container', 'OS-level virtualization'],
        'default': ['Container (computing)']
    },
    'docker': {
        'programming': ['Docker (software)', 'Docker container platform'],
        'default': ['Docker (software)']
    },
    'kubernetes': {
        'programming': ['Kubernetes', 'Container orchestration'],
        'default': ['Kubernetes']
    },
    'cloud': {
        'programming': ['Cloud computing', 'Cloud infrastructure'],
        'default': ['Cloud computing']
    },
    'lambda': {
        'programming': ['Anonymous function', 'Lambda calculus', 'AWS Lambda'],
        'default': ['Anonymous function']
    },
    'expression': {
        'programming': ['Expression (computer science)', 'Programming expression'],
        'default': ['Expression (computer science)']
    },
    'statement': {
        'programming': ['Statement (computer science)', 'Programming statement'],
        'default': ['Statement (computer science)']
    },
    'operator': {
        'programming': ['Operator (computer programming)', 'Programming operator'],
        'default': ['Operator (computer programming)']
    },
    'type': {
        'programming': ['Data type', 'Type system'],
        'default': ['Data type']
    },
    'casting': {
        'programming': ['Type conversion', 'Type casting'],
        'default': ['Type conversion']
    },
    'abstract': {
        'programming': ['Abstract type', 'Abstraction (computer science)'],
        'default': ['Abstract type']
    },
    'static': {
        'programming': ['Static variable', 'Static method'],
        'default': ['Static variable']
    },
    'dynamic': {
        'programming': ['Dynamic typing', 'Dynamic programming language'],
        'default': ['Dynamic typing']
    },
    'private': {
        'programming': ['Access modifier', 'Private member'],
        'default': ['Access modifier']
    },
    'public': {
        'programming': ['Access modifier', 'Public member'],
        'default': ['Access modifier']
    },
    'protected': {
        'programming': ['Access modifier', 'Protected member'],
        'default': ['Access modifier']
    },
    'final': {
        'programming': ['Final (Java)', 'Constant (programming)'],
        'default': ['Final (Java)']
    },
    'const': {
        'programming': ['Constant (programming)', 'Const keyword'],
        'default': ['Constant (programming)']
    },
    'void': {
        'programming': ['Void type', 'Void (programming)'],
        'default': ['Void type']
    },
    'null': {
        'programming': ['Null pointer', 'Null (programming)'],
        'default': ['Null pointer']
    },
    'bank': {
        'finance': ['Bank', 'Financial institution'],
        'geography': ['River bank', 'Stream bank'],
        'default': ['Bank']
    }
}


def _detect_context_type(context_lower: str) -> str:
    """
    Detect the context type based on keywords in the sentence.
    Returns: 'programming', 'tech_company', 'biology', 'finance', 'food', 'entertainment', 'timepiece', or empty string
    """
    if not context_lower:
        return ''
    
    # Calculate scores for each context type
    programming_score = sum(1 for kw in PROGRAMMING_KEYWORDS if kw in context_lower)
    tech_company_score = sum(1 for kw in TECH_COMPANY_KEYWORDS if kw in context_lower)
    biology_score = sum(1 for kw in BIOLOGY_KEYWORDS if kw in context_lower)
    finance_score = sum(1 for kw in FINANCE_KEYWORDS if kw in context_lower)
    food_score = sum(1 for kw in FOOD_KEYWORDS if kw in context_lower)
    entertainment_score = sum(1 for kw in ENTERTAINMENT_KEYWORDS if kw in context_lower)
    timepiece_score = sum(1 for kw in TIMEPIECE_KEYWORDS if kw in context_lower)
    observation_score = sum(1 for kw in OBSERVATION_KEYWORDS if kw in context_lower)
    fitness_score = sum(1 for kw in FITNESS_KEYWORDS if kw in context_lower)
    business_score = sum(1 for kw in BUSINESS_KEYWORDS if kw in context_lower)
    emotion_score = sum(1 for kw in EMOTION_KEYWORDS if kw in context_lower)
    computer_score = sum(1 for kw in COMPUTER_KEYWORDS if kw in context_lower)
    legal_score = sum(1 for kw in LEGAL_KEYWORDS if kw in context_lower)
    tools_score = sum(1 for kw in TOOLS_KEYWORDS if kw in context_lower)
    season_score = sum(1 for kw in SEASON_KEYWORDS if kw in context_lower)
    water_score = sum(1 for kw in WATER_KEYWORDS if kw in context_lower)
    mechanical_score = sum(1 for kw in MECHANICAL_KEYWORDS if kw in context_lower)
    construction_score = sum(1 for kw in CONSTRUCTION_KEYWORDS if kw in context_lower)
    bird_score = sum(1 for kw in BIRD_KEYWORDS if kw in context_lower)
    electrical_score = sum(1 for kw in ELECTRICAL_KEYWORDS if kw in context_lower)
    payment_score = sum(1 for kw in PAYMENT_KEYWORDS if kw in context_lower)
    military_score = sum(1 for kw in MILITARY_KEYWORDS if kw in context_lower)
    writing_score = sum(1 for kw in WRITING_KEYWORDS if kw in context_lower)
    music_score = sum(1 for kw in MUSIC_KEYWORDS if kw in context_lower)
    currency_score = sum(1 for kw in CURRENCY_KEYWORDS if kw in context_lower)
    industrial_score = sum(1 for kw in INDUSTRIAL_KEYWORDS if kw in context_lower)
    botany_score = sum(1 for kw in BOTANY_KEYWORDS if kw in context_lower)
    spy_score = sum(1 for kw in SPY_KEYWORDS if kw in context_lower)
    sports_score = sum(1 for kw in SPORTS_KEYWORDS if kw in context_lower)
    sales_score = sum(1 for kw in SALES_KEYWORDS if kw in context_lower)
    terrain_score = sum(1 for kw in TERRAIN_KEYWORDS if kw in context_lower)
    social_score = sum(1 for kw in SOCIAL_KEYWORDS if kw in context_lower)
    education_score = sum(1 for kw in EDUCATION_KEYWORDS if kw in context_lower)
    insect_score = sum(1 for kw in INSECT_KEYWORDS if kw in context_lower)
    surveillance_score = sum(1 for kw in SURVEILLANCE_KEYWORDS if kw in context_lower)
    fashion_score = sum(1 for kw in FASHION_KEYWORDS if kw in context_lower)
    product_score = sum(1 for kw in PRODUCT_KEYWORDS if kw in context_lower)
    
    # Return the highest scoring context
    scores = [
        (programming_score, 'programming'),
        (tech_company_score, 'tech_company'),
        (biology_score, 'biology'),
        (finance_score, 'finance'),
        (food_score, 'food'),
        (entertainment_score, 'entertainment'),
        (timepiece_score, 'timepiece'),
        (observation_score, 'observation'),
        (fitness_score, 'fitness'),
        (business_score, 'business'),
        (emotion_score, 'emotion'),
        (computer_score, 'computer'),
        (legal_score, 'legal'),
        (tools_score, 'tools'),
        (season_score, 'season'),
        (water_score, 'water'),
        (mechanical_score, 'mechanical'),
        (construction_score, 'construction'),
        (bird_score, 'bird'),
        (electrical_score, 'electrical'),
        (payment_score, 'payment'),
        (military_score, 'military'),
        (writing_score, 'writing'),
        (music_score, 'music'),
        (currency_score, 'currency'),
        (industrial_score, 'industrial'),
        (botany_score, 'botany'),
        (spy_score, 'spy'),
        (sports_score, 'sports'),
        (sales_score, 'sales'),
        (terrain_score, 'terrain'),
        (social_score, 'social'),
        (education_score, 'education'),
        (insect_score, 'insect'),
        (surveillance_score, 'surveillance'),
        (fashion_score, 'fashion'),
        (product_score, 'product')
    ]
    
    best_score, best_context = max(scores, key=lambda x: x[0])
    
    if best_score > 0:
        return best_context
    return ''


def _build_context_aware_search_terms(word: str, context_lower: str) -> List[str]:
    """
    Build Wikipedia search terms based on word and context.
    Returns a list of search terms ordered by relevance.
    """
    word_lower = word.lower().strip()
    context_type = _detect_context_type(context_lower)
    
    # Check if we have mappings for this word
    if word_lower in CONTEXT_SEARCH_MAPPINGS:
        mappings = CONTEXT_SEARCH_MAPPINGS[word_lower]
        
        # Try to get context-specific search terms
        if context_type and context_type in mappings:
            return mappings[context_type]
        
        # Fall back to default
        if 'default' in mappings:
            return mappings['default']
    
    # No specific mapping - build generic search terms based on context
    if context_type == 'programming':
        return [
            f"{word} (programming)",
            f"{word} (computer science)",
            f"{word} (computing)",
            word
        ]
    elif context_type == 'tech_company':
        return [
            f"{word} Inc.",
            f"{word} (company)",
            word
        ]
    
    # Default - just use the word
    return [word]



def get_wikipedia_summary(word: str, max_sentences: int = 3, context: str = None) -> Optional[str]:
    """
    Fetch Wikipedia summary using direct API calls with strict timeout.
    Uses context to disambiguate between different meanings (e.g., Python programming vs snake).
    """
    context_lower = (context or "").lower()
    
    # Generate context-aware cache key
    cache_key_suffix = _detect_context_type(context_lower)
    cache_word = f"{word.lower()}_{cache_key_suffix}" if cache_key_suffix else word
    
    # Check cache first
    cached = _load_cache(cache_word)
    if cached is not None:
        return cached.get('summary')
    
    try:
        # Detect context type and build appropriate search terms
        search_terms = _build_context_aware_search_terms(word, context_lower)
        if not search_terms:
            search_terms = [word]
        
        summary = None
        
        # Requests session with strict timeout
        session = requests.Session()
        headers = {
            'User-Agent': 'WSDHybridModel/1.0 (viranchpatel@example.com)'
        }
        
        for term in search_terms:
            try:
                # 1. Search for the page
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    "action": "query",
                    "format": "json",
                    "list": "search",
                    "srsearch": term,
                    "srlimit": 1
                }
                
                # Strict 2 second timeout
                resp = session.get(search_url, params=search_params, headers=headers, timeout=2.0)
                data = resp.json()
                
                if not data.get("query", {}).get("search"):
                    continue
                    
                page_title = data["query"]["search"][0]["title"]
                
                # 2. Get the summary for the best match
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
                summary_resp = session.get(summary_url, headers=headers, timeout=2.0)
                
                if summary_resp.status_code == 200:
                    summary_data = summary_resp.json()
                    extract = summary_data.get("extract")
                    
                    if extract:
                        # Simple sentence splitting since we don't have nltk sentence splitter here 
                        # or to be fast. Just take first few chars/sentences approx.
                        sentences = extract.split('. ')
                        summary = '. '.join(sentences[:max_sentences]) + '.'
                        break
            except Exception:
                continue
                
        # Cache results with context-aware key
        _save_cache(cache_word, {'summary': summary})
        return summary
        
    except Exception as e:
        return None


def _find_best_disambiguation(options: List[str], word: str, context: str = None) -> str:
    """Find the best disambiguation option based on context."""
    context_lower = (context or "").lower()
    
    # Priority keywords for different contexts
    tech_keywords = ['iphone', 'phone', 'computer', 'software', 'launched', 'released', 'app', 'device']
    
    # Check if context suggests technology
    is_tech = any(kw in context_lower for kw in tech_keywords)
    
    for option in options:
        option_lower = option.lower()
        
        # For tech context, prefer company/technology options
        if is_tech:
            if 'company' in option_lower or 'inc' in option_lower or 'technology' in option_lower:
                return option
        
        # Default: prefer the simplest option with the word
        if word.lower() in option_lower and '(' not in option:
            return option
    
    # Fallback to first option
    return options[0] if options else word


def get_disambiguation_candidates(word: str) -> List[str]:
    """
    Get disambiguation page options for a word.
    Returns list of possible meanings/entities.
    """
    try:
        wikipedia.set_lang("en")
        wikipedia.page(word)
        return [word]  # No disambiguation needed
    except wikipedia.exceptions.DisambiguationError as e:
        return e.options[:10] if e.options else []
    except:
        return []


def get_wikipedia_context(word: str, synset_name: str) -> Optional[str]:
    """
    Get Wikipedia context relevant to a specific WordNet synset.
    Uses the synset lemma to find relevant Wikipedia content.
    """
    # Extract the base word from synset name (e.g., "bank.n.01" -> "bank")
    base_word = synset_name.split('.')[0] if '.' in synset_name else word
    
    # Try searching with the word
    return get_wikipedia_summary(base_word)


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def wikipedia_overlap_score(context_tokens: List[str], wiki_text: str) -> float:
    """
    Calculate lexical overlap between context and Wikipedia text.
    Returns a score based on shared tokens.
    """
    if not wiki_text:
        return 0.0
    
    wiki_tokens = set(simple_tokenize(wiki_text))
    if not wiki_tokens:
        return 0.0
    
    # Count overlapping tokens
    overlap = len(set(context_tokens) & wiki_tokens)
    
    # Normalize by context length to favor more comprehensive matches
    return overlap


def get_enriched_gloss(word: str, synset, max_wiki_chars: int = 200) -> str:
    """
    Create an enriched gloss combining WordNet and Wikipedia.
    
    Args:
        word: The target word
        synset: WordNet synset object
        max_wiki_chars: Maximum characters from Wikipedia to include
        
    Returns:
        Enriched gloss string
    """
    parts = []
    
    # WordNet definition
    parts.append(synset.definition())
    
    # WordNet examples
    examples = synset.examples()
    if examples:
        parts.append("Examples: " + "; ".join(examples[:2]))
    
    # Hypernym context
    hypernyms = synset.hypernyms()
    if hypernyms:
        parts.append("Category: " + hypernyms[0].definition())
    
    # Wikipedia context
    wiki_summary = get_wikipedia_summary(word)
    if wiki_summary:
        # Truncate to max chars
        wiki_text = wiki_summary[:max_wiki_chars]
        if len(wiki_summary) > max_wiki_chars:
            wiki_text = wiki_text.rsplit(' ', 1)[0] + "..."
        parts.append("Wikipedia: " + wiki_text)
    
    return " ".join(parts)


def batch_prefetch_wikipedia(words: List[str], show_progress: bool = True):
    """
    Pre-fetch Wikipedia summaries for a list of words.
    Useful for batch processing training data.
    """
    from tqdm import tqdm
    
    words_to_fetch = []
    for word in words:
        if _load_cache(word) is None:
            words_to_fetch.append(word)
    
    if show_progress:
        print(f"Pre-fetching {len(words_to_fetch)} Wikipedia articles...")
        iterator = tqdm(words_to_fetch, desc="Wikipedia")
    else:
        iterator = words_to_fetch
    
    for word in iterator:
        get_wikipedia_summary(word)


if __name__ == "__main__":
    # Test the module
    print("Testing Wikipedia Knowledge Module...")
    
    test_words = ["apple", "bank", "python", "java"]
    
    for word in test_words:
        print(f"\n{'='*50}")
        print(f"Word: {word}")
        
        summary = get_wikipedia_summary(word)
        if summary:
            print(f"Summary: {summary[:200]}...")
        else:
            print("No Wikipedia summary found")
        
        candidates = get_disambiguation_candidates(word)
        if len(candidates) > 1:
            print(f"Disambiguation options: {candidates[:5]}")

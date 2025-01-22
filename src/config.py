"""Configuration settings for the Ford Ka analysis project."""

from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Ensure directories exist
for dir_path in [DATA_DIR, OUTPUT_DIR, FIGURE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data files
DEMOGRAPHIC_FILE = DATA_DIR / "FordKaDemographicData.csv"
PSYCHOGRAPHIC_FILE = DATA_DIR / "FordKaPsychographicData.csv"

# Analysis parameters
RANDOM_SEED = 42
N_CLUSTERS = 3
GROUP_NAMES = {
    1: "Group 1 (Ka Chooser - Top 3)",
    2: "Group 2 (Ka Non-Chooser - Bottom 3)",
    3: "Group 3 (Middle - Middle 4)"
}

# Plotting parameters
FIGURE_SIZE = (12, 8)
STYLE = "seaborn"
DPI = 300

# Question descriptions
QUESTION_DESCRIPTIONS = {
    'Q1': 'Trendiness/Style Importance',
    'Q2': 'Design Appeal',
    'Q14': 'Practical Considerations',
    'Q20': 'Value for Money',
    'Q23': 'Brand Image',
    'Q41': 'Environmental Concerns',
    'Q44': 'Social Status',
    'Q45': 'Technology Features',
    'Q46': 'Performance Expectations',
    'Q55': 'Safety Features'
} 
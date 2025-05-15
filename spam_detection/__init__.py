import logging
import sys
import matplotlib.pyplot as plt
import nltk

# Setting up logging configuration to log to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
# Setting the style for matplotlib plots
plt.style.use("ggplot")

# check if the required NLTK data is available, if not, download it
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    logging.info("Punkt tokenizer and stopwords corpus are already downloaded.")
except LookupError:
    logging.info("Downloading the punkt tokenizer and stopwords corpus...")
    nltk.download("punkt")
    nltk.download("stopwords")
    logging.info("Download complete.")

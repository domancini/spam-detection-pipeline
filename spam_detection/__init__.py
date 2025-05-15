import logging
import sys
import matplotlib.pyplot as plt
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

import joblib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🔁 Loading data...")
try:
    spot = joblib.load('spot_cleaned.pkl')
    cos_sim = joblib.load('cosine_sim.pkl')
    logging.info("✅ Data loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load required files: %s", str(e))
    raise e


def recommend_songs(song_name, top_n=6):
    logging.info("🎵 Recommending songs for: '%s'", song_name)
    ind = spot[spot['song'].str.lower() == song_name.lower()].index
    if len(ind) == 0:
        logging.warning("⚠️ Song not found in dataset.")
        return None
    ind = ind[0]
    sim_scores = list(enumerate(cos_sim[ind]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]
    logging.info("✅ Top %d recommendations ready.", top_n)
    # Create DataFrame with clean serial numbers starting from 1
    result_spot = spot[['artist', 'song']].iloc[song_indices].reset_index(drop=True)
    result_spot.index = result_spot.index + 1  # Start from 1 instead of 0
    result_spot.index.name = "S.No."

    return result_spot
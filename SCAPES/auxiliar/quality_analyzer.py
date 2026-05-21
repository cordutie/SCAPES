import os
import sqlite3
import gc
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------
# 1. THE ANALYZER (Feature Extraction & DB Storage)
# ---------------------------------------------------------
class AudioQualityExpert:
    # UPDATED: Changed default SR to 44100 so we can actually measure high frequencies
    def __init__(self, db_path: str = "audio_stats_v2.db", sr: int = 44100):
        self.sr = sr
        self.db_path = db_path
        self._prepare_db()

    def _prepare_db(self):
        """Initializes SQLite database with new columns for SR and HF Energy."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    file TEXT PRIMARY KEY,
                    original_sr INTEGER,
                    duration REAL, snr_rms REAL, clipping_ratio REAL,
                    silence_ratio REAL, spectral_flatness REAL,
                    spectral_centroid REAL, spectral_rolloff REAL,
                    rms_mean REAL, complexity REAL, hf_energy_ratio REAL
                )
            """)

    def get_processed_files(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT file FROM stats")
            return {row[0] for row in cursor.fetchall()}

    def analyze_folder(self, folder_path: str):
        processed = self.get_processed_files()
        
        files = []
        for root, _, fs in os.walk(folder_path):
            for f in fs:
                if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                    files.append(os.path.join(root, f))
                    
        to_process = [f for f in files if f not in processed]
        
        if not to_process:
            print("All files in folder are already processed!")
            return

        for path in tqdm(to_process, desc="Analyzing Audio"):
            try:
                stats = self._extract_features(path)
                self._save_to_db(stats)
            except Exception as e:
                print(f"\nError processing {path}: {e}")
            
            if len(processed) % 50 == 0:
                gc.collect()

    def _extract_features(self, path: str):
        # FAST-FAIL: Check the native sample rate by reading the header only
        orig_sr = librosa.get_samplerate(path)
        
        if orig_sr < 44100:
            # Return a "dummy" dict. This logs it in the DB so we don't check it again,
            # but marks duration=0 so it instantly fails our Pandas filters later.
            return {
                "file": path, "original_sr": orig_sr, "duration": 0.0, "snr_rms": 0.0,
                "clipping_ratio": 0.0, "silence_ratio": 1.0, "spectral_flatness": 0.0,
                "spectral_centroid": 0.0, "spectral_rolloff": 0.0, "rms_mean": 0.0,
                "complexity": 0.0, "hf_energy_ratio": 0.0
            }

        # If it passes the SR check, load the full file
        y, sr = librosa.load(path, sr=self.sr)
        
        if len(y) == 0:
            raise ValueError("Empty audio file.")

        # RMS & SNR
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        median = np.median(y)
        mad = np.median(np.abs(y - median))
        snr = float(20 * np.log10((rms_mean + 1e-10) / (mad + 1e-10)))

        # NEW SILENCE METRIC: Frames that are 45dB quieter than the peak volume
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        silence_ratio = float(np.mean(rms_db < -45))

        # NEW HF ENERGY METRIC: Ratio of energy above 10kHz
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=self.sr)
        hf_mask = freqs > 10000 # Frequencies above 10kHz
        hf_energy_ratio = float(np.sum(S[hf_mask, :]) / (np.sum(S) + 1e-10))

        # Spectral and Complexity Features
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

        return {
            "file": path,
            "original_sr": orig_sr,
            "duration": len(y) / sr,
            "snr_rms": snr,
            "clipping_ratio": float(np.mean(np.abs(y) >= 0.98)),
            "silence_ratio": silence_ratio,
            "spectral_flatness": flatness,
            "spectral_centroid": centroid,
            "spectral_rolloff": rolloff,
            "rms_mean": rms_mean,
            "complexity": zcr,
            "hf_energy_ratio": hf_energy_ratio
        }

    def _save_to_db(self, stats: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO stats VALUES 
                (:file, :original_sr, :duration, :snr_rms, :clipping_ratio, :silence_ratio, 
                 :spectral_flatness, :spectral_centroid, :spectral_rolloff, 
                 :rms_mean, :complexity, :hf_energy_ratio)
            """, stats)

    def get_dataframe(self):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM stats", conn)


# ---------------------------------------------------------
# 2. THE VISUALIZER (Data Exploration)
# ---------------------------------------------------------
class MinimalAudioVisualizer:
    def __init__(self, db_path: str = "audio_stats_v2.db"):
        with sqlite3.connect(db_path) as conn:
            self.df = pd.read_sql_query("SELECT * FROM stats", conn)
            
        if self.df.empty:
            print("Database is empty! Run the Analyzer first.")

    def show_summary(self):
        if self.df.empty: return
        print("=== Audio Dataset Summary ===")
        print(f"Total Files Analyzed: {len(self.df)}")
        print(f"Files skipped due to low SR: {len(self.df[self.df['duration'] == 0])}")
        
        metrics = ['duration', 'snr_rms', 'silence_ratio', 'hf_energy_ratio']
        # Filter out the 0-duration dummy files for accurate summary stats
        valid_df = self.df[self.df['duration'] > 0]
        display(valid_df[metrics].describe().round(3)) 

    def plot_distributions(self):
        if self.df.empty: return
        valid_df = self.df[self.df['duration'] > 0]
        metrics = ['snr_rms', 'silence_ratio', 'hf_energy_ratio', 'spectral_flatness']
        valid_df[metrics].hist(figsize=(12, 8), bins=50, grid=False, edgecolor='black', alpha=0.7)
        plt.suptitle("Feature Distributions (Valid Files Only)", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_health_check(self):
        if self.df.empty: return
        valid_df = self.df[self.df['duration'] > 0]
        plt.figure(figsize=(10, 6))
        
        # Now plotting Silence Ratio vs HF Energy
        scatter = plt.scatter(
            valid_df['silence_ratio'], 
            valid_df['hf_energy_ratio'], 
            alpha=0.6, 
            c=valid_df['snr_rms'], 
            cmap='plasma'
        )
        
        plt.colorbar(scatter, label="SNR (Brighter Yellow is Better)")
        plt.xlabel("Silence Ratio -> (Higher = More dead air)")
        plt.ylabel("High-Freq Energy Ratio -> (Higher = More harsh/hissy)")
        plt.title("Audio Health: Find files with too much silence or hiss")
        
        plt.tight_layout()
        plt.show()
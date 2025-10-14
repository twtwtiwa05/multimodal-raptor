"""
GTFS data loader with BOM encoding fix
Based on GTFSLOADER2.py
"""

import pandas as pd
import os
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class GTFSLoader:
    """
    GTFS data loader with Korean BOM encoding support
    
    Handles Korean GTFS data with BOM encoding issues and
    provides clean CSV output with metadata.
    """
    
    def __init__(self, gtfs_dir: str, output_dir: str = None):
        """
        Initialize GTFS loader
        
        Args:
            gtfs_dir: Path to raw GTFS directory
            output_dir: Path to output processed data (optional)
        """
        self.gtfs_dir = gtfs_dir
        self.output_dir = output_dir or os.path.join(os.path.dirname(gtfs_dir), 'cleaned_gtfs_data')
        
    def load_and_clean(self) -> Dict[str, pd.DataFrame]:
        """
        Load and clean all GTFS files
        
        Returns:
            Dictionary of cleaned DataFrames
        """
        files_to_process = [
            'agency.txt', 'calendar.txt', 'routes.txt', 
            'stop_times.txt', 'stops.txt', 'trips.txt'
        ]
        
        cleaned_data = {}
        
        for filename in files_to_process:
            filepath = os.path.join(self.gtfs_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"Processing {filename}...")
                df = self._load_with_encoding_detection(filepath)
                cleaned_data[filename.replace('.txt', '')] = df
            else:
                logger.warning(f"File not found: {filename}")
        
        if self.output_dir:
            self._save_cleaned_data(cleaned_data)
            
        return cleaned_data
    
    def _load_with_encoding_detection(self, filepath: str) -> pd.DataFrame:
        """Load file with proper encoding detection"""
        encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                logger.debug(f"Successfully loaded {filepath} with {encoding}")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
                
        raise ValueError(f"Could not decode {filepath} with any encoding")
    
    def _save_cleaned_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Save cleaned data to CSV files"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        metadata = {
            'files_processed': list(data.keys()),
            'total_records': {k: len(v) for k, v in data.items()},
            'processing_date': pd.Timestamp.now().isoformat()
        }
        
        for name, df in data.items():
            output_path = os.path.join(self.output_dir, f"{name}.csv")
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Saved {name}: {len(df)} records")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"GTFS processing complete. Output: {self.output_dir}")


def main():
    """CLI entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m mmraptor.data.gtfs_loader <gtfs_dir> [output_dir]")
        return
    
    gtfs_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    loader = GTFSLoader(gtfs_dir, output_dir)
    data = loader.load_and_clean()
    
    print(f"✅ Processed {len(data)} GTFS files")
    for name, df in data.items():
        print(f"  - {name}: {len(df)} records")


if __name__ == "__main__":
    main()
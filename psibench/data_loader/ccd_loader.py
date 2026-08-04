"""Utilities for loading pre-extracted CCDs from local files or HuggingFace."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CCDLoader:
    """Load pre-extracted CCDs from local directory or HuggingFace dataset."""
    
    def __init__(self, source: str, cache_dir: Optional[str] = None):
        """Initialize CCD loader.
        
        Args:
            source: Either a local directory path or HuggingFace dataset ID
            cache_dir: Optional cache directory for HuggingFace datasets
        """
        self.source = source
        self.cache_dir = cache_dir
        self.is_local = Path(source).exists()
        self.ccds = {}
        
        if self.is_local:
            self._load_local()
        else:
            self._load_huggingface()
    
    def _load_local(self):
        """Load CCDs from local directory."""
        source_path = Path(self.source)
        logger.info(f"Loading CCDs from local directory: {source_path}")
        
        ccd_files = list(source_path.glob('ccd_*.json'))
        if not ccd_files:
            logger.warning(f"No CCD files found in {source_path}")
            return
        
        for ccd_file in ccd_files:
            try:
                with open(ccd_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session_id = data.get('session_id')
                    if session_id is None:
                        # Try to extract from filename
                        try:
                            session_id = int(ccd_file.stem.split('ccd_')[-1])
                            data['session_id'] = session_id
                        except Exception:
                            logger.warning(f"Cannot extract session_id from {ccd_file}")
                            continue
                    
                    self.ccds[session_id] = data
            except Exception as e:
                logger.error(f"Error loading {ccd_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.ccds)} CCDs from local directory")
    
    def _load_huggingface(self):
        """Load CCDs from HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required for HuggingFace loading. "
                "Install with: pip install datasets"
            )
        
        logger.info(f"Loading CCDs from HuggingFace: {self.source}")
        
        try:
            ds = load_dataset(
                self.source,
                split='train',
                cache_dir=self.cache_dir
            )
            
            for item in ds:
                session_id = item.get('session_id')
                if session_id is None:
                    continue
                
                # Parse JSON strings back to objects
                data = {
                    'session_id': session_id,
                    'messages': json.loads(item['messages']) if isinstance(item['messages'], str) else item['messages'],
                    'ccd': json.loads(item['ccd']) if isinstance(item['ccd'], str) else item['ccd'],
                    'source': item.get('source', 'unknown')
                }
                
                self.ccds[session_id] = data
            
            logger.info(f"Loaded {len(self.ccds)} CCDs from HuggingFace")
            
        except Exception as e:
            logger.error(f"Error loading from HuggingFace: {e}")
            raise
    
    def get_ccd(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get CCD for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with 'ccd', 'messages', 'source' keys, or None if not found
        """
        return self.ccds.get(session_id)
    
    def get_ccd_dict(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get only the CCD dictionary for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            CCD dictionary or None if not found
        """
        data = self.ccds.get(session_id)
        return data.get('ccd') if data else None
    
    def has_ccd(self, session_id: int) -> bool:
        """Check if CCD exists for session."""
        return session_id in self.ccds
    
    def __len__(self) -> int:
        """Return number of loaded CCDs."""
        return len(self.ccds)
    
    def __contains__(self, session_id: int) -> bool:
        """Check if session_id has a CCD."""
        return session_id in self.ccds


def load_ccds(source: str, cache_dir: Optional[str] = None) -> CCDLoader:
    """Convenience function to create a CCDLoader.
    
    Args:
        source: Either a local directory path or HuggingFace dataset ID
        cache_dir: Optional cache directory for HuggingFace datasets
        
    Returns:
        CCDLoader instance
        
    Example:
        # Load from local directory
        ccds = load_ccds('data/ccds/gpt-4o-mini/esc')
        ccd = ccds.get_ccd_dict(42)
        
        # Load from HuggingFace
        ccds = load_ccds('username/psibench-ccds')
        ccd = ccds.get_ccd_dict(42)
    """
    return CCDLoader(source, cache_dir)

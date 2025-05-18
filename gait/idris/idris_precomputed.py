from gait.idris import IdrisLiteEmb
import numpy as np
from typing import List, Tuple, Sequence, Dict, Any
import os
import json

class IdrisPrecomputedEmb(IdrisLiteEmb):
    """Embedding class that uses precomputed embeddings"""
    
    def __init__(
            self,
            embedding_folder: str,
            model_name: str = "text-embedding-ada-002",
            **kwargs
    ) -> None:
        """Initialize with precomputed embeddings
        
        Args:
            embedding_folder: Path to folder containing embeddings
            model_name: Model name (only used for new query embedding)
            **kwargs: Additional parameters for the embedding API
        """
        super().__init__(model_name=model_name, **kwargs)
        
        # Load embeddings
        self._context_embeddings_array = np.load(
            os.path.join(embedding_folder, "context_embeddings.npy"))
        self._question_embeddings_array = np.load(
            os.path.join(embedding_folder, "question_embeddings.npy"))
        
        # Load original data
        with open(os.path.join(embedding_folder, "context_data.json"), "r") as f:
            context_data = json.load(f)
            
        with open(os.path.join(embedding_folder, "question_sql_data.json"), "r") as f:
            question_sql_data = json.load(f)
            
        # Initialize internal dictionaries
        self._context = context_data
        self._question_sql = question_sql_data
        
        # Map data to embeddings
        self._context_embeddings = {}
        for i, ctx in enumerate(self._context):
            self._context_embeddings[ctx] = self._context_embeddings_array[i]
            
        self._question_sql_embeddings = {}
        for i, (q, sql) in enumerate(self._question_sql):
            self._question_sql_embeddings[(q, sql)] = self._question_embeddings_array[i]
        
        # Mark as initialized
        self._initialized = True
    
    def add_context(self, context: str) -> None:
        """Override to prevent modification of preloaded data"""
        # Optionally log warning that this is ignored
        pass
        
    def load_context(self, context: List[str]) -> None:
        """Override to prevent modification of preloaded data"""
        # Optionally log warning that this is ignored
        pass
        
    def load_question_sql(self, question_sql: List[Tuple[str, str]]) -> None:
        """Override to prevent modification of preloaded data"""
        # Optionally log warning that this is ignored
        pass
        
    # The similarity search methods can use the parent class implementation
    # as they'll work with our preloaded data structures
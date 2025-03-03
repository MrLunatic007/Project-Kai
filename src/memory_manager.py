import sqlite3
import os
from datetime import datetime

class MemoryManager:
    """Manages personalized memory for the assistant using SQLite"""
    
    def __init__(self, memory_path="data/memory.db", max_memories=2000, retrain_threshold=100, importance_threshold=0.5):
        self.memory_path = memory_path
        self.max_memories = max_memories
        self.retrain_threshold = retrain_threshold
        self.importance_threshold = importance_threshold
        self.training_needed = False
        
        # Initialize SQLite database
        self.conn = sqlite3.connect(self.memory_path)
        self._create_tables()
        self._check_initial_load()
    
    def _create_tables(self):
        """Create user_memory and web_data tables if they don’t exist"""
        with self.conn:
            # User-Kai memory table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            # Web-scraped data table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS web_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
    
    def _check_initial_load(self):
        """Check memory counts on startup"""
        with self.conn:
            stm_count = self.conn.execute("SELECT COUNT(*) FROM user_memory WHERE importance < ?", 
                                         (self.importance_threshold,)).fetchone()[0]
            ltm_count = self.conn.execute("SELECT COUNT(*) FROM user_memory WHERE importance >= ?", 
                                         (self.importance_threshold,)).fetchone()[0]
            web_count = self.conn.execute("SELECT COUNT(*) FROM web_data").fetchone()[0]
            print(f"Loaded {stm_count} short-term, {ltm_count} long-term user memories, {web_count} web data entries")
    
    def save_memories(self):
        """Commit changes to SQLite (auto-committed, but explicit save for clarity)"""
        self.conn.commit()
        print("Saved memories to SQLite database")
    
    def add_memory(self, user_input, response, importance=None):
        """Add a new user-Kai memory"""
        if importance is None:
            importance = self._calculate_importance(user_input, response)
        
        timestamp = datetime.now().isoformat()
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO user_memory (user_input, response, timestamp, importance, access_count)
                VALUES (?, ?, ?, ?, ?)
            """, (user_input, response, timestamp, importance, 0))
        
        self._check_memory_limits()
        if self._count_short_term() >= self.retrain_threshold:
            self.training_needed = True
    
    def add_web_data(self, query, content, source_url):
        """Add web-scraped data to the database"""
        timestamp = datetime.now().isoformat()
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO web_data (query, content, source_url, timestamp)
                VALUES (?, ?, ?, ?)
            """, (query, content[:500], source_url, timestamp))  # Cap content at 500 chars
    
    def _calculate_importance(self, user_input, response):
        importance = 0.5
        if len(user_input) > 100 or len(response) > 100:
            importance += 0.1
        learning_indicators = ["how", "why", "what is", "explain", "teach"]
        if any(indicator in user_input.lower() for indicator in learning_indicators):
            importance += 0.2
        personal_indicators = ["i am", "my name", "i like", "i want", "i need", "i prefer"]
        if any(indicator in user_input.lower() for indicator in personal_indicators):
            importance += 0.3
        return min(1.0, importance)
    
    def _count_short_term(self):
        """Count short-term memories (importance < threshold)"""
        with self.conn:
            return self.conn.execute("SELECT COUNT(*) FROM user_memory WHERE importance < ?",
                                    (self.importance_threshold,)).fetchone()[0]
    
    def _check_memory_limits(self):
        """Manage memory limits"""
        stm_count = self._count_short_term()
        ltm_count = self.conn.execute("SELECT COUNT(*) FROM user_memory WHERE importance >= ?",
                                     (self.importance_threshold,)).fetchone()[0]
        web_count = self.conn.execute("SELECT COUNT(*) FROM web_data").fetchone()[0]
        
        if stm_count > self.max_memories:
            self._consolidate_memories()
        if ltm_count > self.max_memories:
            self._prune_long_term_memory()
        if web_count > self.max_memories:
            self._prune_web_data()
    
    def _consolidate_memories(self):
        """Consolidate short-term user memories"""
        with self.conn:
            # Fetch and sort short-term by importance
            cursor = self.conn.execute("SELECT * FROM user_memory WHERE importance < ? ORDER BY importance DESC",
                                      (self.importance_threshold,))
            short_term = [{"id": row[0], "user_input": row[1], "response": row[2], "timestamp": row[3], 
                           "importance": row[4], "access_count": row[5]} for row in cursor.fetchall()]
            
            keep_count = int(self.max_memories * 0.3)
            important_memories = short_term[:keep_count]
            
            # Move high-importance to long-term
            for memory in important_memories:
                if memory["importance"] >= self.importance_threshold:
                    self.conn.execute("UPDATE user_memory SET importance = ? WHERE id = ?",
                                     (memory["importance"], memory["id"]))
            
            # Delete excess short-term
            excess_ids = [m["id"] for m in short_term[keep_count:]]
            if excess_ids:
                self.conn.execute(f"DELETE FROM user_memory WHERE id IN ({','.join('?' * len(excess_ids))})", excess_ids)
            print(f"Consolidated short-term: kept {len(important_memories)}")
    
    def _prune_long_term_memory(self):
        """Prune long-term user memories"""
        with self.conn:
            cursor = self.conn.execute("""
                SELECT id, importance, timestamp, access_count 
                FROM user_memory 
                WHERE importance >= ? 
                ORDER BY importance * 0.6 + 1.0 / ((julianday('now') - julianday(timestamp)) + 1) * 0.2 + 
                         LEAST(1.0, access_count / 10.0) * 0.2 DESC
            """, (self.importance_threshold,))
            long_term = [{"id": row[0]} for row in cursor.fetchall()]
            
            keep_count = int(self.max_memories * 0.8)
            excess_ids = [m["id"] for m in long_term[keep_count:]]
            if excess_ids:
                self.conn.execute(f"DELETE FROM user_memory WHERE id IN ({','.join('?' * len(excess_ids))})", excess_ids)
            print(f"Pruned long-term: kept {keep_count}")
    
    def _prune_web_data(self):
        """Prune web data by timestamp"""
        with self.conn:
            cursor = self.conn.execute("SELECT id FROM web_data ORDER BY timestamp DESC")
            web_entries = [row[0] for row in cursor.fetchall()]
            
            keep_count = self.max_memories
            excess_ids = web_entries[keep_count:]
            if excess_ids:
                self.conn.execute(f"DELETE FROM web_data WHERE id IN ({','.join('?' * len(excess_ids))})", excess_ids)
            print(f"Pruned web data: kept {keep_count}")
    
    def get_relevant_memories(self, user_input, max_results=5):
        """Retrieve relevant user-Kai memories"""
        with self.conn:
            cursor = self.conn.execute("SELECT user_input, response, importance, access_count FROM user_memory")
            all_memories = [{"user_input": row[0], "response": row[1], "importance": row[2], "access_count": row[3]} 
                            for row in cursor.fetchall()]
        
        relevant_memories = []
        for memory in all_memories:
            relevance = self._calculate_relevance(user_input, memory["user_input"])
            if relevance > 0.3:
                memory["relevance"] = relevance
                relevant_memories.append(memory)
                # Update access count
                self.conn.execute("UPDATE user_memory SET access_count = access_count + 1 WHERE user_input = ? AND response = ?",
                                 (memory["user_input"], memory["response"]))
        
        relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant_memories[:max_results]
    
    def get_web_data(self, query, max_results=5):
        """Retrieve relevant web-scraped data"""
        with self.conn:
            cursor = self.conn.execute("SELECT query, content, source_url FROM web_data")
            web_entries = [{"query": row[0], "content": row[1], "source_url": row[2]} for row in cursor.fetchall()]
        
        relevant_entries = []
        for entry in web_entries:
            relevance = self._calculate_relevance(query, entry["query"])
            if relevance > 0.3:
                entry["relevance"] = relevance
                relevant_entries.append(entry)
        
        relevant_entries.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant_entries[:max_results]
    
    def _calculate_relevance(self, query, stored_input):
        query_words = set(query.lower().split())
        stored_words = set(stored_input.lower().split())
        if not query_words or not stored_words:
            return 0
        overlap = query_words.intersection(stored_words)
        return len(overlap) / max(len(query_words), len(stored_words))
    
    def get_training_data(self):
        """Get user-Kai memory data for retraining"""
        with self.conn:
            cursor = self.conn.execute("SELECT user_input, response FROM user_memory")
            training_data = [{"input": row[0], "output": row[1]} for row in cursor.fetchall()]
        
        self.training_needed = False
        return training_data
    
    def check_retrain_needed(self):
        return self.training_needed
    
    def clear_short_term_memory(self):
        """Clear short-term user memories after retraining"""
        with self.conn:
            self.conn.execute("DELETE FROM user_memory WHERE importance < ?", (self.importance_threshold,))
        print("Short-term memory cleared")
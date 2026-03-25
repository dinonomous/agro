import sqlite3
import json
import os
from datetime import datetime

# Define database path relative to this script (in the root Agro_hub directory)
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database.db')

def init_db():
    """
    Initializes the SQLite database with the required schema.
    Creates a simple user_requests table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_json TEXT,
            output_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_request(input_data, output_data):
    """
    Saves a prediction request and its response to the database.
    Stores the data as JSON strings for simplicity.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute(
            'INSERT INTO user_requests (timestamp, input_json, output_json) VALUES (?, ?, ?)',
            (timestamp, json.dumps(input_data), json.dumps(output_data))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database save error: {e}")

def get_recent_requests(limit=10):
    """
    Retrieves the most recent requests from the database.
    Used by the /history API endpoint for the UI.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, timestamp, input_json, output_json FROM user_requests ORDER BY id DESC LIMIT ?', 
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "timestamp": row[1],
                "input_json": json.loads(row[2]),
                "output_json": json.loads(row[3])
            })
        return result
    except Exception as e:
        print(f"Database read error: {e}")
        return []

# Ensure the database is initialized when this module is imported
init_db()

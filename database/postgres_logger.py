"""
postgres_logger.py - Store results and metadata in PostgreSQL
"""

import psycopg2

def log_result_to_postgres(mosquito_id, species, confidence, timestamp, image_path):
    try:
        conn = psycopg2.connect(
            dbname="mosquito_db",
            user="username",
            password="password",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO mosquito_results (mosquito_id, species, confidence, timestamp, image_path) VALUES (%s, %s, %s, %s, %s)",
            (mosquito_id, species, confidence, timestamp, image_path)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("Database error:", e)

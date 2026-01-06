#!/usr/bin/env python3
"""
Simple database viewer - shows all data at once
"""
import sqlite3
import json
from tabulate import tabulate
import os

def show_database():
    """Show all database contents"""
    db_path = "backend/enterprise_genai.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        print("🗄️ Enterprise GenAI Database Contents")
        print("=" * 80)
        
        # Show table counts
        print("\n📊 Table Overview:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"📋 {table_name}: {count} records")
        
        # Show documents
        print("\n📄 Documents:")
        print("-" * 80)
        cursor.execute("""
            SELECT document_id, filename, file_size, page_count, 
                   processing_status, created_at
            FROM documents 
            ORDER BY created_at DESC
        """)
        documents = cursor.fetchall()
        
        if documents:
            headers = ["Document ID", "Filename", "Size", "Pages", "Status", "Created"]
            rows = []
            for doc in documents:
                rows.append([
                    doc[0][:12] + "...",
                    doc[1],
                    f"{doc[2]:,}",
                    doc[3],
                    doc[4],
                    doc[5][:19] if doc[5] else "N/A"
                ])
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print("No documents found")
        
        # Show chunks summary
        print("\n📝 Document Chunks:")
        print("-" * 80)
        cursor.execute("""
            SELECT document_id, COUNT(*) as chunk_count, 
                   AVG(char_count) as avg_chars, page_number
            FROM chunks 
            GROUP BY document_id
        """)
        chunk_summary = cursor.fetchall()
        
        if chunk_summary:
            headers = ["Document ID", "Chunks", "Avg Chars", "Page"]
            rows = []
            for chunk in chunk_summary:
                rows.append([
                    chunk[0][:12] + "...",
                    chunk[1],
                    f"{chunk[2]:.0f}" if chunk[2] else "0",
                    chunk[3]
                ])
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print("No chunks found")
        
        # Show recent queries
        print("\n🔍 Recent Queries:")
        print("-" * 80)
        cursor.execute("""
            SELECT query_id, user_query, processing_time, timestamp
            FROM query_logs 
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        queries = cursor.fetchall()
        
        if queries:
            for i, query in enumerate(queries, 1):
                print(f"{i}. Query: {query[1][:60]}{'...' if len(query[1]) > 60 else ''}")
                print(f"   ID: {query[0][:12]}...")
                print(f"   Time: {query[2]:.2f}s" if query[2] else "   Time: N/A")
                print(f"   When: {query[3]}")
                print()
        else:
            print("No queries found")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    show_database()
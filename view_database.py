#!/usr/bin/env python3
"""
Database viewer for Enterprise GenAI Platform
"""
import sqlite3
import json
from datetime import datetime
from tabulate import tabulate
import os

def connect_to_db():
    """Connect to the SQLite database"""
    db_path = "backend/enterprise_genai.db"
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None

def view_tables():
    """List all tables in the database"""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("📊 Database Tables:")
        print("=" * 50)
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"📋 {table_name}: {count} records")
        
        conn.close()
    except Exception as e:
        print(f"❌ Error viewing tables: {e}")

def view_documents():
    """View all documents in the database"""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT document_id, filename, file_size, page_count, 
                   processing_status, processing_date, created_at
            FROM documents 
            ORDER BY created_at DESC
        """)
        
        documents = cursor.fetchall()
        
        if not documents:
            print("📄 No documents found in database")
            conn.close()
            return
        
        print("📄 Documents:")
        print("=" * 100)
        
        headers = ["Document ID", "Filename", "Size (bytes)", "Pages", "Status", "Processed", "Created"]
        rows = []
        
        for doc in documents:
            rows.append([
                doc[0][:12] + "...",  # Truncate document ID
                doc[1],
                f"{doc[2]:,}",
                doc[3],
                doc[4],
                doc[5][:19] if doc[5] else "N/A",  # Format datetime
                doc[6][:19] if doc[6] else "N/A"
            ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing documents: {e}")

def view_chunks(document_id=None):
    """View document chunks"""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        if document_id:
            cursor.execute("""
                SELECT chunk_id, document_id, page_number, chunk_index, 
                       char_count, content
                FROM chunks 
                WHERE document_id = ?
                ORDER BY chunk_index
            """, (document_id,))
        else:
            cursor.execute("""
                SELECT chunk_id, document_id, page_number, chunk_index, 
                       char_count, content
                FROM chunks 
                ORDER BY document_id, chunk_index
                LIMIT 20
            """)
        
        chunks = cursor.fetchall()
        
        if not chunks:
            print("📝 No chunks found in database")
            conn.close()
            return
        
        print("📝 Document Chunks:")
        print("=" * 120)
        
        for chunk in chunks:
            print(f"🔹 Chunk ID: {chunk[0]}")
            print(f"   Document: {chunk[1][:12]}...")
            print(f"   Page: {chunk[2]}, Index: {chunk[3]}, Characters: {chunk[4]}")
            print(f"   Content: {chunk[5][:100]}{'...' if len(chunk[5]) > 100 else ''}")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing chunks: {e}")

def view_queries():
    """View query logs"""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query_id, user_query, response, processing_time, 
                   token_usage, timestamp
            FROM query_logs 
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        queries = cursor.fetchall()
        
        if not queries:
            print("🔍 No queries found in database")
            conn.close()
            return
        
        print("🔍 Recent Queries:")
        print("=" * 120)
        
        for query in queries:
            print(f"🔹 Query ID: {query[0]}")
            print(f"   Question: {query[1]}")
            print(f"   Response: {query[2][:100] if query[2] else 'No response'}{'...' if query[2] and len(query[2]) > 100 else ''}")
            print(f"   Processing Time: {query[3]:.2f}s" if query[3] else "   Processing Time: N/A")
            
            # Parse token usage if available
            if query[4]:
                try:
                    token_data = json.loads(query[4]) if isinstance(query[4], str) else query[4]
                    print(f"   Tokens: {token_data}")
                except:
                    print(f"   Tokens: {query[4]}")
            
            print(f"   Timestamp: {query[5]}")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing queries: {e}")

def view_schema():
    """View database schema"""
    conn = connect_to_db()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("🏗️ Database Schema:")
        print("=" * 80)
        
        for table in tables:
            table_name = table[0]
            print(f"\n📋 Table: {table_name}")
            print("-" * 40)
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            headers = ["Column", "Type", "Not Null", "Default", "Primary Key"]
            rows = []
            
            for col in columns:
                rows.append([
                    col[1],  # name
                    col[2],  # type
                    "YES" if col[3] else "NO",  # not null
                    col[4] if col[4] else "",  # default
                    "YES" if col[5] else "NO"   # primary key
                ])
            
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing schema: {e}")

def main():
    """Main function with menu"""
    print("🗄️ Enterprise GenAI Database Viewer")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. View tables overview")
        print("2. View documents")
        print("3. View document chunks")
        print("4. View query logs")
        print("5. View database schema")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            view_tables()
        elif choice == "2":
            view_documents()
        elif choice == "3":
            doc_id = input("Enter document ID (or press Enter for all): ").strip()
            view_chunks(doc_id if doc_id else None)
        elif choice == "4":
            view_queries()
        elif choice == "5":
            view_schema()
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
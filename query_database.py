#!/usr/bin/env python3
"""
Direct SQL query tool for Enterprise GenAI Database
"""
import sqlite3
import json
from tabulate import tabulate
import os

def execute_query(query):
    """Execute a SQL query and display results"""
    db_path = "backend/enterprise_genai.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(query)
        
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            
            if results:
                # Get column names
                columns = [description[0] for description in cursor.description]
                
                # Convert rows to list of lists for tabulate
                rows = []
                for row in results:
                    formatted_row = []
                    for item in row:
                        if isinstance(item, str) and len(item) > 50:
                            # Truncate long strings
                            formatted_row.append(item[:47] + "...")
                        elif item is None:
                            formatted_row.append("NULL")
                        else:
                            formatted_row.append(str(item))
                    rows.append(formatted_row)
                
                print(f"\n📊 Query Results ({len(results)} rows):")
                print("=" * 100)
                print(tabulate(rows, headers=columns, tablefmt="grid"))
            else:
                print("📊 No results found.")
        else:
            # For non-SELECT queries
            conn.commit()
            print(f"✅ Query executed successfully. Rows affected: {cursor.rowcount}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error executing query: {e}")

def main():
    """Interactive SQL query tool"""
    print("🔍 Enterprise GenAI Database Query Tool")
    print("=" * 50)
    print("Enter SQL queries to execute against the database.")
    print("Type 'help' for common queries, 'exit' to quit.")
    print()
    
    # Common queries
    common_queries = {
        "tables": "SELECT name FROM sqlite_master WHERE type='table';",
        "documents": "SELECT document_id, filename, file_size, page_count, processing_status FROM documents;",
        "chunks": "SELECT chunk_id, document_id, page_number, chunk_index, char_count FROM chunks LIMIT 10;",
        "queries": "SELECT query_id, user_query, processing_time, timestamp FROM query_logs ORDER BY timestamp DESC LIMIT 5;",
        "doc_count": "SELECT COUNT(*) as total_documents FROM documents;",
        "chunk_count": "SELECT COUNT(*) as total_chunks FROM chunks;",
        "query_count": "SELECT COUNT(*) as total_queries FROM query_logs;",
        "recent_docs": "SELECT filename, processing_date FROM documents ORDER BY processing_date DESC LIMIT 5;",
        "schema_documents": "PRAGMA table_info(documents);",
        "schema_chunks": "PRAGMA table_info(chunks);",
        "schema_queries": "PRAGMA table_info(query_logs);"
    }
    
    while True:
        query = input("\nSQL> ").strip()
        
        if query.lower() == 'exit':
            print("👋 Goodbye!")
            break
        elif query.lower() == 'help':
            print("\n📚 Common Queries:")
            print("-" * 30)
            for name, sql in common_queries.items():
                print(f"{name:15}: {sql}")
            continue
        elif query.lower() in common_queries:
            query = common_queries[query.lower()]
        
        if query:
            execute_query(query)

if __name__ == "__main__":
    main()
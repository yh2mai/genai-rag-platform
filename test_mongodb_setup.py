#!/usr/bin/env python3
"""
Test MongoDB setup and connection
"""
import sys

def test_mongodb_connection():
    """Test MongoDB connection"""
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure
        
        print("🧪 Testing MongoDB Connection")
        print("=" * 50)
        
        # Try to connect
        print("📡 Connecting to MongoDB...")
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # Get database
        db = client['enterprise_genai']
        
        # Test insert
        print("\n📝 Testing insert operation...")
        result = db.test_collection.insert_one({"test": "data", "timestamp": "2024-01-01"})
        print(f"✅ Insert successful! ID: {result.inserted_id}")
        
        # Test query
        print("\n🔍 Testing query operation...")
        doc = db.test_collection.find_one({"test": "data"})
        print(f"✅ Query successful! Document: {doc}")
        
        # Test update
        print("\n✏️ Testing update operation...")
        result = db.test_collection.update_one(
            {"test": "data"},
            {"$set": {"updated": True}}
        )
        print(f"✅ Update successful! Modified: {result.modified_count}")
        
        # Test delete
        print("\n🗑️ Testing delete operation...")
        result = db.test_collection.delete_one({"test": "data"})
        print(f"✅ Delete successful! Deleted: {result.deleted_count}")
        
        # List collections
        print("\n📋 Existing collections:")
        collections = db.list_collection_names()
        if collections:
            for coll in collections:
                count = db[coll].count_documents({})
                print(f"   - {coll}: {count} documents")
        else:
            print("   No collections found (database is empty)")
        
        client.close()
        
        print("\n" + "=" * 50)
        print("✅ All MongoDB tests passed!")
        print("\n🚀 You can now use MongoDB with the application:")
        print("   1. Edit backend/.env")
        print("   2. Set DATABASE_TYPE=mongodb")
        print("   3. Restart the backend server")
        
        return True
        
    except ConnectionFailure as e:
        print(f"\n❌ MongoDB connection failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure MongoDB is installed and running")
        print("   2. Check if MongoDB service is started:")
        print("      - Windows: sc query MongoDB")
        print("      - macOS: brew services list")
        print("      - Linux: sudo systemctl status mongod")
        print("   3. Try starting MongoDB:")
        print("      - Windows: net start MongoDB")
        print("      - macOS: brew services start mongodb-community")
        print("      - Linux: sudo systemctl start mongod")
        return False
        
    except ImportError:
        print("❌ pymongo not installed")
        print("\n💡 Install it with:")
        print("   pip install pymongo")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_application_config():
    """Test application MongoDB configuration"""
    try:
        print("\n🔧 Testing Application Configuration")
        print("=" * 50)
        
        # Add backend to path
        sys.path.insert(0, 'backend')
        
        from app.core.config import settings
        
        print(f"Database Type: {settings.DATABASE_TYPE}")
        print(f"MongoDB URL: {settings.MONGODB_URL}")
        print(f"MongoDB Database: {settings.MONGODB_DATABASE}")
        
        if settings.DATABASE_TYPE.lower() == "mongodb":
            print("\n✅ Application is configured to use MongoDB")
            
            # Test MongoDB models
            print("\n📦 Testing MongoDB models...")
            from app.models.mongodb import init_mongodb, get_database
            
            init_mongodb()
            print("✅ MongoDB models initialized successfully")
            
            db = get_database()
            print(f"✅ Connected to database: {db.name}")
            
            # List collections
            collections = db.list_collection_names()
            print(f"✅ Collections: {collections}")
            
        else:
            print(f"\n⚠️  Application is configured to use {settings.DATABASE_TYPE}")
            print("   To use MongoDB, edit backend/.env and set:")
            print("   DATABASE_TYPE=mongodb")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🗄️ MongoDB Setup Test")
    print("=" * 50)
    print()
    
    # Test MongoDB connection
    mongo_ok = test_mongodb_connection()
    
    if mongo_ok:
        # Test application configuration
        test_application_config()
    
    print()


if __name__ == "__main__":
    main()

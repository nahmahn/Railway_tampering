
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGO_URI")
print(f"Testing connection to: {uri.split('@')[1] if '@' in uri else 'local'}")

try:
    client = MongoClient(uri)
    client.admin.command('ping')
    print("✅ Ping successful!")
    
    db = client[os.getenv("DB_NAME", "test_db")]
    print(f"✅ Database selected: {db.name}")
    
    # Try writing a document
    result = db.test_collection.insert_one({"test": "data", "status": "verified"})
    print(f"✅ Insert successful: {result.inserted_id}")
    
    # Try reading it back
    doc = db.test_collection.find_one({"_id": result.inserted_id})
    print(f"✅ Read successful: {doc}")
    
    # Clean up
    db.test_collection.delete_one({"_id": result.inserted_id})
    print("✅ Cleanup successful")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")

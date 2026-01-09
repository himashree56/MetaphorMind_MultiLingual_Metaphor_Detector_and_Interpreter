# MongoDB Setup Guide for History Feature

## Overview

The history feature uses MongoDB to store all predictions, allowing you to:
- View past predictions
- Filter by language or type (metaphor/normal)
- See statistics about your usage
- Delete individual predictions or clear all history

---

## Installation Options

### Option 1: Local MongoDB (Recommended for Development)

#### Windows

1. **Download MongoDB Community Server**
   - Visit: https://www.mongodb.com/try/download/community
   - Download the Windows installer (.msi)
   - Version: 6.0 or higher

2. **Install MongoDB**
   ```bash
   # Run the installer with default settings
   # MongoDB will be installed to: C:\Program Files\MongoDB\Server\6.0\
   ```

3. **Start MongoDB Service**
   ```bash
   # MongoDB should start automatically as a Windows service
   # To check if it's running:
   net start MongoDB
   ```

4. **Verify Installation**
   ```bash
   # Open Command Prompt and run:
   mongosh
   
   # You should see:
   # Current Mongosh Log ID: ...
   # Connecting to: mongodb://127.0.0.1:27017/
   ```

#### Linux/Mac

```bash
# Ubuntu/Debian
sudo apt-get install mongodb

# Mac (using Homebrew)
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # Mac
```

---

### Option 2: MongoDB Atlas (Cloud - Free Tier)

1. **Create Account**
   - Visit: https://www.mongodb.com/cloud/atlas/register
   - Sign up for free

2. **Create Cluster**
   - Choose "Shared" (Free tier)
   - Select your region
   - Click "Create Cluster"

3. **Configure Access**
   - Go to "Database Access" → Add user
   - Create username and password
   - Go to "Network Access" → Add IP Address
   - Add `0.0.0.0/0` (allow from anywhere) for development

4. **Get Connection String**
   - Click "Connect" on your cluster
   - Choose "Connect your application"
   - Copy the connection string
   - Replace `<password>` with your actual password

---

## Configuration

### 1. Update Environment Variables

Edit your `.env` file (create from `.env.example` if needed):

```bash
# For Local MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=metaphor_detector

# For MongoDB Atlas (Cloud)
MONGODB_URL=mongodb+srv://<username>:<password>@<cluster-url>/<db-name>
MONGODB_DB_NAME=metaphor_detector
```

### 2. Install Python Dependencies

```bash
pip install motor pymongo
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

---

## Testing the Connection

### Quick Test Script

Create a file `test_mongodb.py`:

```python
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "metaphor_detector")

try:
    client = MongoClient(MONGODB_URL)
    client.admin.command('ping')
    print("✓ Successfully connected to MongoDB!")
    
    db = client[MONGODB_DB_NAME]
    print(f"✓ Using database: {MONGODB_DB_NAME}")
    
    # Test insert
    test_collection = db.test
    result = test_collection.insert_one({"test": "data"})
    print(f"✓ Test insert successful: {result.inserted_id}")
    
    # Clean up
    test_collection.delete_one({"_id": result.inserted_id})
    print("✓ Test cleanup successful")
    
    client.close()
    print("\n✅ MongoDB is ready to use!")
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {str(e)}")
    print("\nTroubleshooting:")
    print("1. Make sure MongoDB is running")
    print("2. Check your MONGODB_URL in .env file")
    print("3. Verify network connectivity (for Atlas)")
```

Run the test:
```bash
python test_mongodb.py
```

---

## Using the History Feature

### Starting the Application

1. **Start Backend** (with MongoDB running)
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

   You should see:
   ```
   INFO:     Connecting to MongoDB at mongodb://localhost:27017
   INFO:     ✓ Successfully connected to MongoDB database: metaphor_detector
   INFO:     ✓ Database indexes created
   ```

2. **Start Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

### Accessing History

1. Click the **"📜 History"** button in the top-right corner
2. View all your past predictions
3. Filter by language or type
4. Click on any prediction to see full details
5. Delete individual predictions or clear all history

---

## API Endpoints

### Get History
```http
GET /history?limit=50&skip=0&language=hindi&label=metaphor
```

### Get Specific Prediction
```http
GET /history/{prediction_id}
```

### Delete Prediction
```http
DELETE /history/{prediction_id}
```

### Clear All History
```http
DELETE /history
```

### Get Statistics
```http
GET /statistics
```

---

## Database Structure

### Collection: `predictions`

```json
{
  "_id": "ObjectId",
  "text": "वह आसमान छू रहा है",
  "language": "hindi",
  "label": "metaphor",
  "confidence": 0.9234,
  "translation": "He is touching the sky",
  "explanation": "This metaphor compares...",
  "timestamp": "2025-01-16T10:30:00.000Z"
}
```

### Indexes

- `timestamp` (descending) - For sorting by date
- `language` - For filtering by language
- `label` - For filtering by type

---

## Troubleshooting

### MongoDB Not Starting

**Windows:**
```bash
# Check if service is running
sc query MongoDB

# Start service
net start MongoDB

# If fails, check logs at:
# C:\Program Files\MongoDB\Server\6.0\log\mongod.log
```

**Linux:**
```bash
# Check status
sudo systemctl status mongod

# Start service
sudo systemctl start mongod

# Check logs
sudo journalctl -u mongod
```

### Connection Refused

1. **Check if MongoDB is running:**
   ```bash
   mongosh
   ```

2. **Verify port 27017 is not blocked:**
   ```bash
   netstat -an | grep 27017
   ```

3. **Check firewall settings**

### Atlas Connection Issues

1. **Whitelist your IP address** in Network Access
2. **Verify username/password** in connection string
3. **Check connection string format:**
   ```
   mongodb+srv://username:password@cluster.mongodb.net/
   ```

### Application Works Without History

If MongoDB connection fails, the application will continue to work but:
- History feature will be disabled
- Predictions won't be saved
- You'll see a warning in logs:
  ```
  WARNING: History feature will be disabled
  ```

This is by design - MongoDB is optional for basic functionality.

---

## Performance Tips

### For Large History

1. **Limit results:**
   ```javascript
   // In frontend, use pagination
   fetch('/history?limit=20&skip=0')
   ```

2. **Use filters:**
   ```javascript
   // Filter by language
   fetch('/history?language=hindi')
   ```

3. **Regular cleanup:**
   - Use "Clear All" button periodically
   - Or set up automatic cleanup (TTL index)

### Database Optimization

```javascript
// In MongoDB shell, create TTL index to auto-delete old records
use metaphor_detector
db.predictions.createIndex(
  { "timestamp": 1 },
  { expireAfterSeconds: 2592000 }  // 30 days
)
```

---

## Backup and Restore

### Backup
```bash
mongodump --db metaphor_detector --out ./backup
```

### Restore
```bash
mongorestore --db metaphor_detector ./backup/metaphor_detector
```

---

## Security Best Practices

1. **Never commit `.env` file** (already in .gitignore)
2. **Use strong passwords** for MongoDB users
3. **Restrict IP access** in production
4. **Enable authentication** for production MongoDB
5. **Use environment-specific databases:**
   - Development: `metaphor_detector_dev`
   - Production: `metaphor_detector_prod`

---

## Next Steps

1. ✅ Install MongoDB (local or Atlas)
2. ✅ Configure `.env` file
3. ✅ Test connection
4. ✅ Start the application
5. ✅ Try the history feature!

For issues, check the main `TROUBLESHOOTING.md` file.

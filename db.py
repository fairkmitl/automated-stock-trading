
from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://fair:ELgmZq3SRMA9kfmO@cluster0.gvxc6yw.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# define the database and collection
db = client['stock_trading']
collection = db['portfolio']

# Send a ping to confirm a successful connection
def test_connection():
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    
# test finction
def test_insert():
    document = {"name": "Fair", "age": 28, "city": "Bangkok"}
    result = collection.insert_one(document)
    print("Inserted document with ID:", result.inserted_id)

def test_findAll():
    for doc in collection.find():
        print(doc)
        
def test_filter():
    query_filter = {"city": "Bangkok"}
    for doc in collection.find(query_filter):
        print(doc)
        
def test_update():
    query_filter = {"name": "John", "city": "New York"}
    update_data = {"$set": {"age": 41}}
    result = collection.update_one(query_filter, update_data)
    print("Documents updated:", result.modified_count)

def test_delete():
    query_filter = {"name": "John"}
    result = collection.delete_one(query_filter)
    print("Documents deleted:", result.deleted_count)

def insert(document):
    result = collection.insert_one(document)
    print("Inserted document with ID:", result.inserted_id)
    
# test_connection()
# test_insert()
# test_findAll()
# test_filter()
# test_update()
# test_delete()

# close connection
# client.close()

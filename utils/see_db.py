from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_db")
print(client.get_collections())

info = client.count("activity_log")
print(f"Stored vectors: {info.count}")



records = client.scroll(
    collection_name="activity_log",
    limit=50,   # get first 50
)

for rec in records[0]:
    print(f"ID: {rec.id}")
    print(f"Payload: {rec.payload}")
    print("-----")




meta = client.get_collection("activity_log")
print(meta)



query = "coding on laptop"
hits = client.query(collection_name="activity_log",
                     query_text=query,  # dummy vector shape check
                     limit=5)

for hit in hits:
    print(hit.score)
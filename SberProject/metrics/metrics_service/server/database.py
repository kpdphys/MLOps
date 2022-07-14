import os
import logging
from urllib.parse import quote
import motor.motor_asyncio
from bson.objectid import ObjectId


def connect_to_mongodb():
    mongo_uris = os.getenv("MONGO_URIS", default="localhost:27017")
    user = os.getenv("MONGO_USER", default="user")
    password = os.getenv("MONGO_PASS", default="user")

    logger = logging.getLogger(__name__)
    logging.basicConfig(encoding="utf-8", level=logging.INFO)
    logger.info(f"MongoDB username: {user}")
    logger.info(f"MongoDB password: {password}")

    client = motor.motor_asyncio.AsyncIOMotorClient(f"mongodb://" +
                                                    quote(f"{user}") +
                                                    ":" +
                                                    quote(f"{password}") +
                                                    f"@{mongo_uris}")
    return client.search_results, database.get_collection("search_results_collection")


database, search_results_collection = connect_to_mongodb()


def search_result_helper(search_result) -> dict:
    print(search_result)
    return {
        "id": str(search_result["_id"]),
        "query": search_result["query"],
        "result": search_result["result"],
        "link_pos": search_result["link_pos"]
    }


async def add_index():
    return await search_results_collection.create_index("link_pos")


async def retrieve_search_results():
    search_results = []
    async for search_result in search_results_collection.find():
        search_results.append(search_result_helper(search_result))
    return search_results


async def retrieve_search_result(id: str) -> dict:
    search_result = await search_results_collection.find_one({"_id": ObjectId(id)})
    if search_result:
        return search_result_helper(search_result)


async def add_search_result(search_result_data: dict) -> dict:
    search_result = await search_results_collection.insert_one(search_result_data)
    new_search_result = await search_results_collection.find_one({"_id": search_result.inserted_id})
    return search_result_helper(new_search_result)


async def update_search_result(id: str, data: dict):
    # Return false if an empty request body is sent.
    if len(data) < 1:
        return False
    search_result = await search_results_collection.find_one({"_id": ObjectId(id)})
    if search_result:
        updated_search_result = await search_results_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_search_result:
            return True
    return False


async def delete_search_result(id: str):
    search_result = await search_results_collection.find_one({"_id": ObjectId(id)})
    if search_result:
        await search_results_collection.delete_one({"_id": ObjectId(id)})
        return True
    return False


async def retrieve_clicks_statistics():
    docs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    # Retrieve the count of docs with the defined clicks position
    pipeline = [
        {
            "$group": {
                "_id": "$link_pos",
                "count": {"$sum": 1}
            }
        }
    ]
    async for doc in search_results_collection.aggregate(pipeline):
        docs[doc["_id"]] = doc["count"]
    return docs

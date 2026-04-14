# =============================================================================
# ImmunoBiology RAG — MongoDB Connection Manager
# =============================================================================
# Adapted from Tesla RAG: src/client/mongodb_config.py
# Changes: English comments; default db_name → "immunology_rag";
#          reads host/port/db_name from config via environment variables
#          or falls back to src/constant.py values.

import os
from pprint import pprint
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError


class MongoConfig:
    """
    Singleton MongoDB connection manager.

    Uses class variables so only one connection pool is created per process,
    avoiding the overhead of repeatedly opening/closing connections.

    Configuration priority (highest to lowest):
      1. Environment variables: MONGO_HOST, MONGO_PORT, MONGO_DB_NAME, etc.
      2. Values in config.yaml (loaded via src/constant.py)
    """

    # Connection parameters — override via environment variables if needed
    _host = os.getenv("MONGO_HOST", "localhost")
    _port = int(os.getenv("MONGO_PORT", 27017))
    _db_name = os.getenv("MONGO_DB_NAME", "immunology_rag")
    _username = os.getenv("MONGO_USERNAME")
    _password = os.getenv("MONGO_PASSWORD")
    _auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin")

    # Connection pool settings
    _max_pool_size = 100
    _connect_timeout = 5000   # ms
    _socket_timeout = 3000    # ms

    # Singleton state
    _client = None
    _db = None

    @classmethod
    def _build_connection_uri(cls) -> str:
        """Build MongoDB URI with or without authentication credentials."""
        if cls._username and cls._password:
            return (
                f"mongodb://{cls._username}:{cls._password}"
                f"@{cls._host}:{cls._port}/?authSource={cls._auth_source}"
            )
        return f"mongodb://{cls._host}:{cls._port}"

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize the MongoDB connection pool.
        Idempotent: subsequent calls are no-ops if already connected.
        """
        if cls._client is None:
            try:
                cls._client = MongoClient(
                    cls._build_connection_uri(),
                    maxPoolSize=cls._max_pool_size,
                    connectTimeoutMS=cls._connect_timeout,
                    socketTimeoutMS=cls._socket_timeout,
                    serverSelectionTimeoutMS=5000,
                )
                # Verify connection with a ping
                cls._client.admin.command("ping")
                cls._db = cls._client[cls._db_name]
                print(f"[MongoDB] Connected to {cls._host}:{cls._port}/{cls._db_name}")
            except ConfigurationError as e:
                raise RuntimeError(f"[MongoDB] Configuration error: {e}")
            except ConnectionFailure as e:
                raise RuntimeError(f"[MongoDB] Connection failed: {e}")
            except Exception as e:
                raise RuntimeError(f"[MongoDB] Unexpected error: {e}")

    @classmethod
    def get_db(cls):
        """Return the database instance, initializing if necessary."""
        if cls._client is None:
            cls.initialize()
        return cls._db

    @classmethod
    def get_collection(cls, collection_name: str):
        """Return a MongoDB collection by name."""
        return cls.get_db()[collection_name]

    @classmethod
    def close(cls) -> None:
        """Close all connections and reset the singleton state."""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            print("[MongoDB] Connection closed.")


# Initialize connection when this module is imported
MongoConfig.initialize()


if __name__ == "__main__":
    collection = MongoConfig.get_collection("chunk_metadata")
    test_doc = {"unique_id": "test_001", "page_content": "Test immunology chunk."}
    collection.insert_one(test_doc)
    print("Inserted test document:")
    for doc in collection.find({"unique_id": "test_001"}):
        pprint(doc)
    collection.delete_one({"unique_id": "test_001"})
    print("Cleaned up test document.")

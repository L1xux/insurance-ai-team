import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

# 로깅 설정 (INFO 이상 출력)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class DatabaseManager:
    """
    Database connection management for PostgreSQL with pgvector.
    """

    def __init__(self):
        """Initialize database configuration."""
        self.connection_string = os.getenv("DB_CONNECTION_STRING")
        self._connection = None
        logger.info("DatabaseManager initialized")

    def get_connection(self):
        """Get database connection."""
        try:
            if self._connection is None or self._connection.closed:
                self._connection = psycopg2.connect(
                    self.connection_string,
                    cursor_factory=RealDictCursor
                )
                logger.info("Database connection established")
            return self._connection
        except Exception as e:
            import traceback
            logger.error(f"Failed to connect to database: {e}")
            logger.error("Traceback: %s", traceback.format_exc())
            raise

    @contextmanager
    def get_cursor(self, dict_cursor: bool = True):
        """Get database cursor with context management."""
        conn = self.get_connection()
        cursor = None
        try:
            if dict_cursor:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception as e:
            import traceback
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            logger.error("Traceback: %s", traceback.format_exc())
            raise
        finally:
            if cursor:
                cursor.close()

"""Custom exceptions for database operations in SCGO.

This module defines custom exception types for better error handling
and more specific error reporting in database operations.
"""


class DatabaseError(Exception):
    """Base exception for all database-related errors.

    All database exceptions inherit from this for easy catching.
    """


class DatabaseLockError(DatabaseError):
    """Raised when database is locked and operation cannot proceed.

    Usually indicates:
    - Another process has exclusive lock
    - Busy timeout exceeded
    - Concurrent write conflict
    """


class DatabaseSetupError(DatabaseError):
    """Raised when database setup or initialization fails.

    Usually indicates:
    - Failed to create database file
    - Failed to initialize schema
    - Invalid configuration
    """


class DatabaseMigrationError(DatabaseError):
    """Raised when database schema migration fails.

    Usually indicates:
    - Incompatible schema version
    - Failed migration operation
    - Missing migration handler
    """

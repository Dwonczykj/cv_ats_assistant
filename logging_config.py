from enum import Enum
import logging
import logging.handlers
import json
import os
from typing import Optional, Dict, Any
from pathlib import Path
import traceback
import queue
import threading
import sys
from logging.handlers import RotatingFileHandler
import psutil

LOGGER_IS_DEBUG = True


class PackageFilter(logging.Filter):
    """Filter to only allow logs from specific packages"""

    def filter(self, record, use_whitelist=True):
        # Block specific external packages
        if use_whitelist:
            allowed_packages = [
                'root',
                'whatsapp',
                'helpers',
                'models',
                'firebase',
                'chatbot',
                'testing'
            ]
            return any(record.name.startswith(pkg) for pkg in allowed_packages)
        else:
            blocked_packages = {
                'sqlalchemy',
                'httpcore',
                'urllib3',
                'httpx',
                'google',
                'apscheduler',
                'twilio',
                'uvicorn.access',  # block access logs but allow other uvicorn logs
            }

            # Block if the log is from any blocked package
            if any(record.name.startswith(pkg) for pkg in blocked_packages):
                return False

            # Allow all other logs
            return True


class StackTraceFilter(logging.Filter):
    """Filter to exclude specific lines from stack traces"""

    def filter(self, record):
        if record.exc_info:
            # Capture the stack trace
            stack_trace = traceback.format_exception(*record.exc_info)
            # Filter out lines that start with the specified path
            filtered_stack_trace = [
                line.replace("/Users/joey/Github_Keep/whatsapp_transcriber", ".") for line in stack_trace
                if not "/Users/joey/Github_Keep/whatsapp_transcriber/whatsapp_transcriber_venv" in line and
                "/Users/joey/Github_Keep/whatsapp_transcriber" in line
            ]
            # Update the record's message with the filtered stack trace
            record.msg = f"{record.getMessage()}\n{
                ''.join(filtered_stack_trace)}"
        # if record.exc_text:
        #     record.exc_text = record.exc_text.replace(
        #         "/Users/joey/Github_Keep/whatsapp_transcriber", "@")
        if record.stack_info:
            # Capture the stack trace from stack_info
            stack_info_lines = record.stack_info.splitlines()
            # Filter out lines that start with the specified path
            filtered_stack_info = [
                line.replace("/Users/joey/Github_Keep/whatsapp_transcriber", ".") for line in stack_info_lines
                if not "/Users/joey/Github_Keep/whatsapp_transcriber/whatsapp_transcriber_venv" in line and
                "/Users/joey/Github_Keep/whatsapp_transcriber" in line
            ]
            # Update the record's stack_info with the filtered stack trace
            record.stack_info = '\n'.join(filtered_stack_info)

        return True


class AsyncLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AsyncLogger, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return

        self.log_queue = queue.Queue(-1)
        self.queue_handler = logging.handlers.QueueHandler(self.log_queue)

        # Create handlers
        console_handler = self._create_console_handler()
        file_handler = self._create_file_handler()

        self.queue_listener = logging.handlers.QueueListener(
            self.log_queue,
            console_handler,
            file_handler,
            respect_handler_level=True
        )

        self.initialized = True

    def _create_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_formatter())
        console_handler.addFilter(PackageFilter())
        console_handler.addFilter(StackTraceFilter())
        return console_handler

    def _create_file_handler(self):
        file_handler = RotatingFileHandler(
            'app.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(self._get_formatter())
        file_handler.addFilter(PackageFilter())
        file_handler.addFilter(StackTraceFilter())
        return file_handler

    def _get_formatter(self):
        return logging.Formatter(
            '%(asctime)s [%(name)20s] [%(levelname)8s] %(message)s - %(filename)s:%(lineno)s %(funcName)s'
        )

    def setup_logging(self, debug_mode=False):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

        root_logger.handlers.clear()
        root_logger.addHandler(self.queue_handler)

        root_logger.addFilter(StackTraceFilter())

        self._configure_external_loggers()
        self.queue_listener.start()

        # Configure the profiling logger
        profiling_logger = logging.getLogger('local_profiling')
        profiling_logger.setLevel(
            logging.DEBUG if debug_mode else logging.INFO)

        # Add handlers for profiling logger if needed
        if not profiling_logger.handlers:
            # Add your handlers here
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            profiling_logger.addHandler(console_handler)

            # File handler for profiling logs
            if LOGGER_IS_DEBUG:
                file_handler = logging.FileHandler('profiling.log')
                file_handler.setFormatter(formatter)
                profiling_logger.addHandler(file_handler)

        return root_logger

    class NamedLoggers(Enum):
        ROOT = 'root'
        PROFILING = 'profiling'
        LOCAL_PROFILING = 'local_profiling'
        WHATSAPP = 'whatsapp'
        HELPERS = 'helpers'
        MODELS = 'models'
        FIREBASE = 'firebase'
        CHATBOT = 'chatbot'

    _internal_loggers = {
        NamedLoggers.ROOT: logging.DEBUG,
        NamedLoggers.PROFILING: logging.DEBUG,
        NamedLoggers.LOCAL_PROFILING: logging.DEBUG,
        NamedLoggers.WHATSAPP: logging.DEBUG,
        NamedLoggers.HELPERS: logging.DEBUG,
        NamedLoggers.MODELS: logging.DEBUG,
        NamedLoggers.FIREBASE: logging.DEBUG,
        NamedLoggers.CHATBOT: logging.DEBUG,
    }

    # Configure levels for external packages
    _external_loggers = {
        'sqlalchemy': logging.WARNING,
        'sqlalchemy.engine': logging.WARNING,
        'sqlalchemy.engine.base.Engine': logging.WARNING,
        'sqlalchemy.engine.Engine': logging.WARNING,
        'sqlalchemy.dialects': logging.WARNING,
        'sqlalchemy.pool': logging.WARNING,
        'sqlalchemy.orm': logging.WARNING,
        'httpcore': logging.WARNING,
        'urllib3': logging.WARNING,
        'urllib3.connectionpool': logging.WARNING,
        'urllib3.connectionpool.HTTPConnectionPool': logging.WARNING,
        'asyncio': logging.WARNING,
        'sentence_transformers': logging.WARNING,
        'sentence_transformers.SentenceTransformer': logging.WARNING,
        'httpx': logging.WARNING,
        'google': logging.WARNING,
        'google.auth': logging.WARNING,
        'google.cloud': logging.WARNING,
        'apscheduler': logging.WARNING,
        'twilio': logging.WARNING,
        'uvicorn': logging.INFO,
        'gunicorn': logging.INFO,
        'openai': logging.INFO,
        'langchain': logging.INFO
    }

    def _configure_external_loggers(self):
        # Configure levels for our packages

        # Set levels for our internal loggers
        for logger_name, level in self._internal_loggers.items():
            logger = logging.getLogger(logger_name.value)
            logger.setLevel(level)
            logger.propagate = True  # Allow propagation for our loggers

        # Configure external loggers with disabled state
        for logger_name, level in self._external_loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            logger.propagate = False

            if logger_name.startswith('sqlalchemy'):
                logger.disabled = True
                logger.handlers = [logging.NullHandler()]

        # Additional SQLAlchemy specific settings
        engine_logger = logging.getLogger('sqlalchemy.engine')
        engine_logger.disabled = True
        engine_logger.propagate = False
        engine_logger.addHandler(logging.NullHandler())

        # Handle the base sqlalchemy logger as well
        base_logger = logging.getLogger('sqlalchemy')
        base_logger.disabled = True
        base_logger.propagate = False
        base_logger.addHandler(logging.NullHandler())

    def shutdown(self):
        self.queue_listener.stop()


# Create a global instance
async_logger = AsyncLogger()


# Public API functions
def setup_logging(debug_mode=False) -> logging.Logger:
    """Main entry point for setting up logging"""
    return async_logger.setup_logging(debug_mode)


def configure_logging(log_level: Optional[int] = None) -> logging.Logger:
    """Backward compatibility wrapper for setup_logging"""
    debug_mode = log_level == logging.DEBUG if log_level is not None else LOGGER_IS_DEBUG
    return setup_logging(debug_mode)


def set_logger_levels():
    """Configure log levels for external packages"""
    async_logger._configure_external_loggers()


# Optional: Helper function for adding file loggers to specific modules
def add_file_logger(name: str, log_level: int) -> logging.Logger:
    """Add a file logger to a specific module"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    file_handler = RotatingFileHandler(
        f'{name}.log',
        maxBytes=10485760,
        backupCount=5
    )
    file_handler.setFormatter(async_logger._get_formatter())
    file_handler.addFilter(PackageFilter())

    logger.addHandler(file_handler)
    return logger

# alias for add_file_logger


def get_file_logger(name: str, log_level: int = logging.INFO):
    """Get a logger and add a file handler if not already added"""
    return add_file_logger(name=name, log_level=log_level)


def get_named_logger(name: AsyncLogger.NamedLoggers):
    """Get a named logger"""
    return logging.getLogger(name.value)


def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024
    logging.info(f"Memory usage: {memory_usage:.2f} MB")

# Add this to your periodic tasks or after processing each request


mem_prof_log_file = open('memory_profile.log', 'w+')

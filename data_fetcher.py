import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Optional, Union, Literal
from dataclasses import dataclass
import time
from functools import lru_cache
import logging
from pathlib import Path
import sqlite3
import json

# Optional import for tqdm with fallback
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback class if tqdm is not available
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', 0)
            self.desc = kwargs.get('desc', '')
            self.n = 0
            print(f"{self.desc} - Starting...")
            
        def update(self, n=1):
            self.n += n
            if self.total > 0:
                print(f"{self.desc} - Progress: {self.n}/{self.total} ({int(self.n/self.total*100)}%)")
            else:
                print(f"{self.desc} - Progress: {self.n}")
                
        def close(self):
            print(f"{self.desc} - Completed")
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
            
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.update()

MarketCategory = Literal['spot', 'linear', 'inverse']

@dataclass
class FundingRate:
    symbol: str
    funding_rate: float
    funding_time: datetime
    
@dataclass
class TimeRange:
    start: datetime
    end: datetime

    @property
    def start_timestamp(self) -> int:
        """Return the start timestamp of the TimeRange in milliseconds"""
        return int(self.start.timestamp() * 1000)

    @property
    def end_timestamp(self) -> int:
        """Return the end timestamp of the TimeRange in milliseconds"""
        return int(self.end.timestamp() * 1000)

@dataclass
class DataConfig:
    db_path: str = 'crypto_data.db'
    log_path: str = 'data_collection.log'
    checkpoint_path: str = 'checkpoints.json'
    max_workers: int = 10  # Increased from 5 for better parallelism
    request_interval: float = 0.1
    backup_enabled: bool = True
    batch_size: int = 1000  # For batch database operations

class BybitDataFetcher:
    def __init__(self, request_interval: float = 0.1, max_retries: int = 3):
        """
        Initialize the Bybit data fetcher.
        
        Args:
            request_interval: Minimum time between requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = "https://api.bybit.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'bybit-market-data-fetcher/1.0'
        })
        self.last_request_time = 0
        self.request_interval = request_interval
        self.max_retries = max_retries
        
        # Cache for API responses
        self._response_cache = {}

    def _make_request(self, method: str, url: str, params: dict) -> requests.Response:
        """Make an API request with rate limiting and error handling."""
        # Create cache key from URL and parameters
        cache_key = f"{url}:{json.dumps(params, sort_keys=True)}"
        
        # Check cache for recent results (10 minute cache)
        if cache_key in self._response_cache:
            cache_time, cache_data = self._response_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < 600:  # 10 minutes
                return cache_data
        
        # Apply rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_interval:
            time.sleep(self.request_interval - time_since_last_request)
        
        retries = 0
        while retries <= self.max_retries:
            try:
                response = self.session.request(method, url, params=params)
                self.last_request_time = time.time()
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logging.warning(f"Rate limit exceeded. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    retries += 1
                    continue
                
                response.raise_for_status()
                
                # Cache successful responses
                self._response_cache[cache_key] = (datetime.now(), response)
                
                # Cleanup old cache entries periodically
                if len(self._response_cache) > 1000:
                    self._cleanup_cache()
                    
                return response
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Request error: {e}")
                retries += 1
                
                if retries <= self.max_retries:
                    sleep_time = 2 ** retries  # Exponential backoff
                    logging.info(f"Retrying in {sleep_time} seconds... (Attempt {retries}/{self.max_retries})")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Max retries reached for request to {url}")
                    raise
        
        # This should never happen, but just in case
        raise Exception(f"Failed to make request after {self.max_retries} retries")

    def _cleanup_cache(self):
        """Remove old entries from the response cache"""
        now = datetime.now()
        old_keys = [
            key for key, (timestamp, _) in self._response_cache.items()
            if (now - timestamp).total_seconds() > 600  # 10 minutes
        ]
        
        for key in old_keys:
            del self._response_cache[key]

    def fetch_klines(self, symbol: str, time_range: TimeRange, 
                    interval: str = '1h', category: MarketCategory = 'spot') -> pd.DataFrame:
        """
        Fetch kline/candlestick data for a specific symbol and time range.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            time_range: TimeRange object with start and end times
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            category: Market category ('spot', 'linear', or 'inverse')
        """
        url = f"{self.base_url}/v5/market/kline"
        
        # Convert interval to Bybit format
        interval_map = {
            '1m': '1', 
            '5m': '5',
            '15m': '15',
            '1h': '60',
            '4h': '240',
            '1d': 'D'
        }
        
        params = {
            'category': category,
            'symbol': symbol,
            'interval': interval_map.get(interval, '60'),
            'start': time_range.start_timestamp,
            'end': time_range.end_timestamp,
            'limit': 1000
        }
        
        try:
            response = self._make_request('GET', url, params)
            data = response.json()
            
            if data['retCode'] != 0:
                logging.error(f"Error fetching {symbol}: {data['retMsg']}")
                return pd.DataFrame()

            klines = data['result'].get('list', [])
            if not klines:
                logging.info(f"No data available for {symbol} in specified time range")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert types safely
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            float_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in float_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['symbol'] = symbol
            df['category'] = category
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

    @lru_cache(maxsize=100)
    def _split_time_range(self, start_time_str: str, end_time_str: str, 
                         interval: str) -> List[tuple]:
        """
        Split time range into chunks based on interval.
        
        Uses string representation of datetimes for caching.
        Returns a list of (start, end) tuples as strings.
        """
        start_time = datetime.fromisoformat(start_time_str)
        end_time = datetime.fromisoformat(end_time_str)
        
        chunk_sizes = {
            '1m': timedelta(hours=6),
            '5m': timedelta(hours=12),
            '15m': timedelta(days=1),
            '1h': timedelta(days=3),
            '4h': timedelta(days=7),
            '1d': timedelta(days=30)
        }
        
        chunk_size = chunk_sizes.get(interval, timedelta(days=1))
        chunks = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + chunk_size, end_time)
            chunks.append((current_start.isoformat(), current_end.isoformat()))
            current_start = current_end
            
        return chunks

    def fetch_multi_symbol_data(self, symbols: List[str], start_time: datetime, 
                              end_time: datetime, interval: str = '1h',
                              category: MarketCategory = 'spot',
                              max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Fetch kline data for multiple symbols concurrently with progress bar.
        
        Args:
            symbols: List of trading pair symbols
            start_time: Start datetime
            end_time: End datetime
            interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
            category: Market category ('spot', 'linear', or 'inverse')
            max_workers: Maximum number of concurrent requests
        """
        all_data = {}
        
        # Calculate total number of requests
        time_chunks = self._split_time_range(
            start_time.isoformat(), 
            end_time.isoformat(), 
            interval
        )
        total_requests = len(symbols) * len(time_chunks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {}
            
            # Create progress bar
            pbar = tqdm(total=total_requests, 
                       desc=f"Fetching {category} market data",
                       unit='request')
            
            # Submit all tasks
            for symbol in symbols:
                for start_str, end_str in time_chunks:
                    start = datetime.fromisoformat(start_str)
                    end = datetime.fromisoformat(end_str)
                    time_range = TimeRange(start, end)
                    
                    future = executor.submit(
                        self.fetch_klines, 
                        symbol, 
                        time_range, 
                        interval,
                        category
                    )
                    future_to_params[future] = (symbol, time_range, interval)
            
            # Process completed tasks
            for future in as_completed(future_to_params):
                symbol, time_range, interval = future_to_params[future]
                try:
                    df = future.result()
                    if not df.empty:
                        df['interval'] = interval
                        if symbol not in all_data:
                            all_data[symbol] = df
                        else:
                            all_data[symbol] = pd.concat([all_data[symbol], df], 
                                                       ignore_index=True)
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")
                finally:
                    pbar.update(1)
            
            pbar.close()
        
        # Post-process the data
        logging.info("Post-processing data...")
        for symbol in all_data:
            if not all_data[symbol].empty:
                all_data[symbol] = (all_data[symbol]
                                  .sort_values('timestamp')
                                  .drop_duplicates(subset=['timestamp', 'interval'])
                                  .reset_index(drop=True))
        
        # Print summary
        for symbol in symbols:
            if symbol in all_data:
                logging.info(f"{symbol}: {len(all_data[symbol])} data points fetched")
            else:
                logging.info(f"{symbol}: No data fetched")
        
        return all_data

    def fetch_funding_rates(self, symbol: str, time_range: TimeRange,
                            category: MarketCategory = 'linear') -> pd.DataFrame:
        """
        Fetch funding rate history for a symbol based on Bybit's API.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            time_range: TimeRange object with start and end times
            category: Market category ('linear' or 'inverse')

        Returns:
            DataFrame with funding rate data
        """
        url = f"{self.base_url}/v5/market/funding/history"
        all_data = []

        # According to docs, we need both startTime and endTime or just endTime
        # We'll use pagination by adjusting endTime
        current_end_time = time_range.end_timestamp

        # Maximum number of iterations to prevent infinite loops
        max_iterations = 30
        iterations = 0

        logging.info(f"Fetching funding rates for {symbol} from {time_range.start} to {time_range.end}")

        while iterations < max_iterations:
            iterations += 1

            # Set up parameters according to the API docs
            params = {
                'category': category,
                'symbol': symbol,
                'limit': 200,
                'endTime': current_end_time
            }

            # Only include startTime if we're on the first request
            # Otherwise, we'll use pagination with endTime only
            if iterations == 1:
                params['startTime'] = time_range.start_timestamp

            try:
                logging.debug(f"Funding rate request {iterations}: params={params}")
                response = self._make_request('GET', url, params)
                data = response.json()

                # Check for API errors
                if data['retCode'] != 0:
                    logging.warning(f"API error when fetching funding rates: {data['retMsg']}")
                    break

                # Check for empty results
                funding_records = data['result'].get('list', [])
                if not funding_records:
                    logging.info(f"No more funding data available for {symbol}")
                    break

                records_count = len(funding_records)
                logging.debug(f"Retrieved {records_count} funding rate records")

                # Add to our collection
                all_data.extend(funding_records)

                # If we got fewer than the maximum records, we've reached the end
                if records_count < 200:
                    break

                # Get the earliest timestamp in this batch for pagination
                earliest_record = min(funding_records, key=lambda x: int(x['fundingRateTimestamp']))
                earliest_timestamp = int(earliest_record['fundingRateTimestamp'])

                # Update for next pagination request (use timestamp - 1ms)
                current_end_time = earliest_timestamp - 1

                # If we've gone far enough back in time, stop
                if earliest_timestamp <= time_range.start_timestamp:
                    logging.info(f"Reached requested start time for {symbol} funding rates")
                    break

                # Short pause between requests
                time.sleep(0.2)

            except Exception as e:
                logging.error(f"Error fetching funding rates for {symbol}: {e}")
                break

        # Process the data we collected
        if all_data:
            try:
                df = pd.DataFrame(all_data)

                # Convert types
                df['fundingRateTimestamp'] = pd.to_datetime(df['fundingRateTimestamp'].astype(float), unit='ms')
                df['fundingRate'] = df['fundingRate'].astype(float)
                df['symbol'] = symbol
                df['category'] = category

                # Sort chronologically
                df = df.sort_values('fundingRateTimestamp').reset_index(drop=True)

                return df

            except Exception as e:
                logging.error(f"Error processing funding rate data: {e}")

        # Return empty DataFrame if no data
        return pd.DataFrame()

    def fetch_open_interest(self, symbol: str, time_range: TimeRange,
                        interval: str = '1h', category: MarketCategory = 'linear') -> pd.DataFrame:
        """
        Fetch open interest history for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            time_range: TimeRange with start and end times
            interval: Time interval ('5min', '15min', '30min', '1h', '4h', '1d')
            category: Market category ('linear' or 'inverse')
        """
        url = f"{self.base_url}/v5/market/open-interest"
        
        # Convert interval to Bybit format
        interval_map = {
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        api_interval = interval_map.get(interval, interval)
        
        params = {
            'category': category,
            'symbol': symbol,
            'intervalTime': api_interval,
            'limit': 200,
            'startTime': time_range.start_timestamp,
            'endTime': time_range.end_timestamp
        }
        
        try:
            response = self._make_request('GET', url, params)
            data = response.json()
            
            if data['retCode'] != 0 or not data['result']['list']:
                return pd.DataFrame()

            oi_data = data['result']['list']
            
            if not oi_data:
                logging.info(f"No data available for {symbol} in specified time range")
                return pd.DataFrame()
            
            # Convert the data to a DataFrame
            df = pd.DataFrame(oi_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df['openInterest'] = df['openInterest'].astype(float)
            df['symbol'] = symbol
            df['interval'] = interval
            df['category'] = category
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching OI for {symbol} {category}: {e}")
            return pd.DataFrame()

    def fetch_multi_open_interest(self, symbols: List[str], start_time: datetime,
                                 end_time: datetime, interval: str = '1h',
                                 category: MarketCategory = 'linear',
                                 max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Fetch open interest for multiple symbols concurrently.
        """
        all_data = {}
        
        # Calculate total number of requests
        time_chunks = self._split_time_range(
            start_time.isoformat(), 
            end_time.isoformat(), 
            interval
        )
        total_requests = len(symbols) * len(time_chunks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_params = {}
            
            # Create progress bar
            pbar = tqdm(total=total_requests, 
                       desc=f"Fetching {category} open interest data",
                       unit='request')
            
            # Submit all tasks
            for symbol in symbols:
                for start_str, end_str in time_chunks:
                    start = datetime.fromisoformat(start_str)
                    end = datetime.fromisoformat(end_str)
                    time_range = TimeRange(start, end)
                    
                    future = executor.submit(
                        self.fetch_open_interest, 
                        symbol, 
                        time_range, 
                        interval,
                        category
                    )
                    future_to_params[future] = (symbol, time_range, interval)
            
            # Process completed tasks
            for future in as_completed(future_to_params):
                symbol, time_range, interval = future_to_params[future]
                try:
                    df = future.result()
                    if not df.empty:
                        if symbol not in all_data:
                            all_data[symbol] = df
                        else:
                            all_data[symbol] = pd.concat([all_data[symbol], df], 
                                                       ignore_index=True)
                except Exception as e:
                    logging.error(f"Error processing {symbol} OI: {e}")
                finally:
                    pbar.update(1)
            
            pbar.close()
        
        # Post-process the data
        logging.info("Post-processing open interest data...")
        for symbol in all_data:
            if not all_data[symbol].empty:
                all_data[symbol] = (all_data[symbol]
                                  .sort_values('timestamp')
                                  .drop_duplicates(subset=['timestamp', 'interval'])
                                  .reset_index(drop=True))
        
        # Print summary
        for symbol in symbols:
            if symbol in all_data:
                logging.info(f"{symbol} OI: {len(all_data[symbol])} data points fetched")
            else:
                logging.info(f"{symbol} OI: No data fetched")
        
        return all_data


class CryptoDataCollector:
    def __init__(self, config: DataConfig):
        """Initialize the crypto data collector with configuration."""
        self.config = config

        # Initialize database connection pool with WAL + performance PRAGMAs
        self.conn_pool = []
        for _ in range(5):
            c = sqlite3.connect(self.config.db_path, timeout=30)
            c.execute('PRAGMA journal_mode = WAL')
            c.execute('PRAGMA synchronous = NORMAL')
            c.execute('PRAGMA cache_size = -262144')   # 256 MB
            c.execute('PRAGMA temp_store = MEMORY')
            c.execute('PRAGMA mmap_size = 268435456')   # 256 MB
            c.execute('PRAGMA busy_timeout = 15000')
            self.conn_pool.append(c)

        self.setup_logging()
        self.setup_database()
        self.load_checkpoints()
        self.fetcher = BybitDataFetcher(request_interval=config.request_interval)
        
        # Default start dates for different assets
        self.default_start_dates = {
            'BTCUSDT': datetime(2020, 3, 25),  # First funding rate date
            'ETHUSDT': datetime(2020, 10, 21),  # First funding rate date
            'default': datetime(2021, 1, 1)     # Safe default for other assets
        }
        
        # Define available intervals
        self.available_intervals = {
            'open_interest': ['5min', '15min', '30min', '1h', '4h', '1d'],
            'funding_rates': ['8h'],  # Fixed by Bybit
            'price_data': ['5m', '15m', '1h', '4h', '1d']
        }
        
        # Create cache for database operations
        self.cache = {
            'price_data': {},
            'funding_rates': {},
            'open_interest': {}
        }

            
        # Validate database on startup
        if Path(self.config.db_path).exists():
            validation = self.validate_database()
            if not all(validation.values()):
                self.logger.warning("Database validation failed. Attempting repair...")
                self.repair_database()

    def __del__(self):
        """Clean up resources when object is destroyed"""
        for conn in self.conn_pool:
            try:
                conn.close()
            except:
                pass

    def _get_connection(self):
        """Get a database connection from the pool"""
        if not self.conn_pool:
            return sqlite3.connect(self.config.db_path)
        return self.conn_pool.pop()
        
    def _return_connection(self, conn):
        """Return a connection to the pool"""
        if len(self.conn_pool) < 5:
            self.conn_pool.append(conn)
        else:
            conn.close()

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_database(self):
        """Initialize SQLite database with schema"""
        try:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                
                # Price data table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS price_data (
                        symbol TEXT,
                        timestamp DATETIME,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        interval TEXT,
                        category TEXT,
                        PRIMARY KEY (symbol, timestamp, interval, category)
                    )
                ''')
                
                # Funding rates table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS funding_rates (
                        symbol TEXT,
                        timestamp DATETIME,
                        funding_rate REAL,
                        category TEXT,
                        PRIMARY KEY (symbol, timestamp, category)
                    )
                ''')
                
                # Open interest table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS open_interest (
                        symbol TEXT,
                        timestamp DATETIME,
                        openInterest REAL,
                        interval TEXT,
                        category TEXT,
                        PRIMARY KEY (symbol, timestamp, interval, category)
                    )
                ''')
                
                # Create indexes
                cur.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_time ON price_data(symbol, timestamp, interval, category)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_funding_symbol_time ON funding_rates(symbol, timestamp, category)')
                cur.execute('CREATE INDEX IF NOT EXISTS idx_oi_symbol_time ON open_interest(symbol, timestamp, interval, category)')
                
                # Add PRAGMA statements for better performance
                cur.execute('PRAGMA journal_mode = WAL')
                cur.execute('PRAGMA synchronous = NORMAL')
                cur.execute('PRAGMA cache_size = -102400')  # 100MB cache
                cur.execute('PRAGMA temp_store = MEMORY')
                
                conn.commit()
                
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Database setup error: {e}")
            raise

    def _backup_database(self):
        """Create a backup of the database if it exists."""
        if not self.config.backup_enabled:
            return
            
        db_path = Path(self.config.db_path)
        if db_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
            try:
                import shutil
                shutil.copy2(db_path, backup_path)
                self.logger.info(f"Created database backup: {backup_path}")
            except Exception as e:
                self.logger.error(f"Failed to create database backup: {e}")

    def validate_database(self) -> Dict[str, bool]:
        """
        Validate database integrity and schema.
        Returns dict of validation results.
        """
        validations = {
            'tables_exist': True,
            'indexes_exist': True,
            'data_integrity': True,
            'schema_valid': True
        }
        
        try:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                
                # Check tables exist
                tables = ['price_data', 'funding_rates', 'open_interest']
                for table in tables:
                    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                    if not cur.fetchone():
                        validations['tables_exist'] = False
                        self.logger.error(f"Missing table: {table}")
                
                # Check indexes exist
                indexes = ['idx_price_symbol_time', 'idx_funding_symbol_time', 'idx_oi_symbol_time']
                for index in indexes:
                    cur.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{index}'")
                    if not cur.fetchone():
                        validations['indexes_exist'] = False
                        self.logger.error(f"Missing index: {index}")
                
                # Check data integrity
                for table in tables:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                    except sqlite3.DatabaseError:
                        validations['data_integrity'] = False
                        self.logger.error(f"Data integrity issue in table: {table}")
                
                # Validate schema
                expected_columns = {
                    'price_data': {'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'interval', 'category'},
                    'funding_rates': {'symbol', 'timestamp', 'funding_rate', 'category'},
                    'open_interest': {'symbol', 'timestamp', 'openInterest', 'interval', 'category'}
                }
                
                for table, expected in expected_columns.items():
                    cur.execute(f"PRAGMA table_info({table})")
                    columns = {row[1] for row in cur.fetchall()}
                    if columns != expected:
                        validations['schema_valid'] = False
                        self.logger.error(f"Schema mismatch in {table}. Missing: {expected - columns}")
            
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Database validation error: {e}")
            return {k: False for k in validations}
            
        return validations

    def repair_database(self) -> bool:
        """
        Attempt to repair database issues.
        Returns True if successful.
        """
        try:
            validations = self.validate_database()
            
            if all(validations.values()):
                self.logger.info("Database is healthy, no repairs needed")
                return True
            
            # Backup before repair
            self._backup_database()
                
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                
                # Recreate missing tables
                if not validations['tables_exist']:
                    self.setup_database()
                
                # Recreate missing indexes
                if not validations['indexes_exist']:
                    cur.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_time ON price_data(symbol, timestamp, interval, category)')
                    cur.execute('CREATE INDEX IF NOT EXISTS idx_funding_symbol_time ON funding_rates(symbol, timestamp, category)')
                    cur.execute('CREATE INDEX IF NOT EXISTS idx_oi_symbol_time ON open_interest(symbol, timestamp, interval, category)')
                
                # Vacuum database
                cur.execute("VACUUM")
                conn.commit()
                
            finally:
                self._return_connection(conn)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Database repair failed: {e}")
            return False

    def optimize_database(self):
        """Optimize the database for better performance."""
        try:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                
                # Analyze tables for query optimization
                cur.execute("ANALYZE")
                
                # Rebuild indexes
                cur.execute("REINDEX")
                
                # Compact the database
                cur.execute("VACUUM")
                
                # Update statistics
                cur.execute("PRAGMA optimize")
                
                self.logger.info("Database optimization completed")
                
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Database optimization failed: {e}")

    def load_checkpoints(self):
        """Load last successful data collection checkpoints"""
        self.checkpoints = {
            'price_data': {},
            'funding_rates': {},
            'open_interest': {}
        }
        
        checkpoint_path = Path(self.config.checkpoint_path)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    self.checkpoints = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading checkpoints: {e}")

    def save_checkpoints(self):
        """Save current checkpoints"""
        try:
            with open(self.config.checkpoint_path, 'w') as f:
                json.dump(self.checkpoints, f)
        except Exception as e:
            self.logger.error(f"Error saving checkpoints: {e}")

    def get_default_start_date(self, symbol: str) -> datetime:
        """Get the default start date for a symbol."""
        return self.default_start_dates.get(symbol, self.default_start_dates['default'])

    def get_last_timestamp(self, data_type: str, symbol: str, interval: str = None, 
                          category: str = 'linear') -> Optional[datetime]:
        """Get last stored timestamp for a symbol and data type"""
        try:
            conn = self._get_connection()
            try:
                if data_type == 'price_data':
                    query = """
                        SELECT MAX(timestamp) 
                        FROM price_data 
                        WHERE symbol = ? AND interval = ? AND category = ?
                    """
                    result = conn.execute(query, (symbol, interval, category)).fetchone()[0]
                elif data_type == 'funding_rates':
                    query = """
                        SELECT MAX(timestamp)
                        FROM funding_rates
                        WHERE symbol = ? AND category = ?
                    """
                    result = conn.execute(query, (symbol, category)).fetchone()[0]
                elif data_type == 'open_interest':
                    query = """
                        SELECT MAX(timestamp)
                        FROM open_interest
                        WHERE symbol = ? AND interval = ? AND category = ?
                    """
                    result = conn.execute(query, (symbol, interval, category)).fetchone()[0]
                    
                if result:
                    return pd.to_datetime(result)
            finally:
                self._return_connection(conn)
        except Exception as e:
            self.logger.error(f"Error getting last timestamp: {e}")
            
            # Fall back to checkpoints if database query fails
            if data_type == 'price_data':
                checkpoint_key = f"{symbol}_{interval}_{category}"
            elif data_type == 'funding_rates':
                checkpoint_key = f"{symbol}_{category}"
            elif data_type == 'open_interest':
                checkpoint_key = f"{symbol}_{interval}_{category}"
                
            if checkpoint_key in self.checkpoints[data_type]:
                checkpoint_time = self.checkpoints[data_type][checkpoint_key]
                return pd.to_datetime(checkpoint_time)
                
        return None

    def _normalize_timestamps(self, df, timestamp_col='timestamp'):
        """Normalize timestamps to be timezone-naive for consistent comparisons"""
        if timestamp_col in df.columns:
            df = df.copy()
            if df[timestamp_col].dt.tz is not None:
                df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
        return df

    def store_price_data(self, klines_data: Dict[str, pd.DataFrame], symbol: str, interval: str, category: str):
        """Store price data in database using batch operations for better performance"""
        if symbol not in klines_data or klines_data[symbol].empty:
            return

        try:
            df = klines_data[symbol]
            df = self._normalize_timestamps(df, 'timestamp')

            # Check for NaT values and handle them
            if df['timestamp'].isna().any():
                self.logger.warning(f"Found NaT timestamps in {symbol} data. Filtering...")
                df = df.dropna(subset=['timestamp'])

            if df.empty:
                self.logger.warning(f"No valid data left for {symbol} after filtering NaT values")
                return

            conn = self._get_connection()
            try:
                # Get existing timestamps to avoid duplicates (using index for speed)
                query = """
                    SELECT timestamp 
                    FROM price_data 
                    WHERE symbol = ? AND interval = ? AND category = ?
                """
                existing_df = pd.read_sql_query(query, conn, params=(symbol, interval, category))

                if not existing_df.empty:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                    existing_timestamps = set(existing_df['timestamp'])

                    # For incomplete candles (current period), allow updates
                    # Determine what constitutes "current period" based on interval
                    now = datetime.now()

                    # Calculate the start time of the current incomplete candle
                    if interval == '1d':
                        current_candle_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    elif interval == '4h':
                        hour_groups = now.hour // 4
                        current_candle_start = now.replace(hour=hour_groups * 4, minute=0, second=0, microsecond=0)
                    elif interval == '1h':
                        current_candle_start = now.replace(minute=0, second=0, microsecond=0)
                    elif interval == '15m':
                        minute_groups = now.minute // 15
                        current_candle_start = now.replace(minute=minute_groups * 15, second=0, microsecond=0)
                    elif interval == '5m':
                        minute_groups = now.minute // 5
                        current_candle_start = now.replace(minute=minute_groups * 5, second=0, microsecond=0)
                    else:
                        # Default: allow updates for candles from the last 24 hours
                        current_candle_start = now - timedelta(days=1)

                    # Convert current_candle_start to match the timezone of df timestamps
                    if hasattr(current_candle_start, 'tz_localize') and df['timestamp'].dt.tz is None:
                        current_candle_start = current_candle_start.tz_localize(None)
                    elif df['timestamp'].dt.tz is not None and current_candle_start.tzinfo is None:
                        current_candle_start = current_candle_start.replace(tzinfo=df['timestamp'].dt.tz)

                    # Filter out existing records, but allow updates for current/recent incomplete candles
                    df_timestamps = df['timestamp']
                    new_records_mask = ~df_timestamps.isin(existing_timestamps) | (
                            df_timestamps >= pd.Timestamp(current_candle_start))
                    new_df = df[new_records_mask]
                else:
                    new_df = df

                if new_df.empty:
                    self.logger.info(f"No new price data for {symbol} {interval}")
                    return

                # Vectorised row preparation (avoids iterrows overhead)
                ts_strs = new_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values
                opens = new_df['open'].values
                highs = new_df['high'].values
                lows = new_df['low'].values
                closes = new_df['close'].values
                vols = new_df['volume'].values
                new_data = [
                    (symbol, ts_strs[i], float(opens[i]), float(highs[i]),
                     float(lows[i]), float(closes[i]), float(vols[i]),
                     interval, category)
                    for i in range(len(new_df))
                ]

                # Single transaction for all batches
                conn.execute('BEGIN')
                batch_size = self.config.batch_size
                for i in range(0, len(new_data), batch_size):
                    conn.executemany(
                        """INSERT OR REPLACE INTO price_data
                           (symbol, timestamp, open, high, low, close, volume, interval, category)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        new_data[i:i + batch_size],
                    )
                conn.commit()

                # Update checkpoint
                max_timestamp = df['timestamp'].max()
                if pd.notnull(max_timestamp):
                    self.checkpoints['price_data'][f"{symbol}_{interval}_{category}"] = max_timestamp.strftime(
                        '%Y-%m-%d %H:%M:%S')
                    self.save_checkpoints()

                self.logger.info(f"Stored {len(new_data)} new/updated price records for {symbol} {interval}")

            finally:
                self._return_connection(conn)

        except Exception as e:
            self.logger.error(f"Error storing price data for {symbol}: {e}")

    def store_open_interest(self, df_dict: Dict[str, pd.DataFrame], symbol: str, interval: str, category: str):
        """Store open interest data in database using efficient batch operations"""
        if symbol not in df_dict or df_dict[symbol].empty:
            return

        try:
            df = df_dict[symbol]
            df = self._normalize_timestamps(df, 'timestamp')

            # Check for NaT values and handle them
            if df['timestamp'].isna().any():
                self.logger.warning(f"Found NaT timestamps in {symbol} OI data. Filtering...")
                df = df.dropna(subset=['timestamp'])

            if df.empty:
                self.logger.warning(f"No valid OI data left for {symbol} after filtering NaT values")
                return

            conn = self._get_connection()
            try:
                # Get existing timestamps to avoid duplicates
                query = """
                    SELECT timestamp 
                    FROM open_interest 
                    WHERE symbol = ? AND interval = ? AND category = ?
                """
                existing_df = pd.read_sql_query(query, conn, params=(symbol, interval, category))

                if not existing_df.empty:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                    existing_timestamps = set(existing_df['timestamp'])

                    # For incomplete candles (current period), allow updates
                    # Determine what constitutes "current period" based on interval
                    now = datetime.now()

                    # Calculate the start time of the current incomplete candle
                    if interval == '1d':
                        current_candle_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    elif interval == '4h':
                        hour_groups = now.hour // 4
                        current_candle_start = now.replace(hour=hour_groups * 4, minute=0, second=0, microsecond=0)
                    elif interval == '1h':
                        current_candle_start = now.replace(minute=0, second=0, microsecond=0)
                    elif interval in ['15min', '15m']:
                        minute_groups = now.minute // 15
                        current_candle_start = now.replace(minute=minute_groups * 15, second=0, microsecond=0)
                    elif interval in ['5min', '5m']:
                        minute_groups = now.minute // 5
                        current_candle_start = now.replace(minute=minute_groups * 5, second=0, microsecond=0)
                    elif interval in ['30min', '30m']:
                        minute_groups = now.minute // 30
                        current_candle_start = now.replace(minute=minute_groups * 30, second=0, microsecond=0)
                    else:
                        # Default: allow updates for candles from the last 24 hours
                        current_candle_start = now - timedelta(days=1)

                    # Convert current_candle_start to match the timezone of df timestamps
                    if hasattr(current_candle_start, 'tz_localize') and df['timestamp'].dt.tz is None:
                        current_candle_start = current_candle_start.tz_localize(None)
                    elif df['timestamp'].dt.tz is not None and current_candle_start.tzinfo is None:
                        current_candle_start = current_candle_start.replace(tzinfo=df['timestamp'].dt.tz)

                    # Filter out existing records, but allow updates for current/recent incomplete candles
                    df_timestamps = df['timestamp']
                    new_records_mask = ~df_timestamps.isin(existing_timestamps) | (
                            df_timestamps >= pd.Timestamp(current_candle_start))
                    new_df = df[new_records_mask]
                else:
                    new_df = df

                if new_df.empty:
                    self.logger.info(f"No new open interest data for {symbol} {interval}")
                    return

                # Prepare data for insertion - use INSERT OR REPLACE to handle updates
                new_data = []
                for _, row in new_df.iterrows():
                    new_data.append((
                        symbol,
                        row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['timestamp'], 'strftime') else str(row['timestamp']),
                        row['openInterest'],
                        interval,
                        category
                    ))

                # Insert in batches for better performance
                batch_size = self.config.batch_size
                for i in range(0, len(new_data), batch_size):
                    batch = new_data[i:i + batch_size]
                    conn.executemany(
                        """INSERT OR REPLACE INTO open_interest 
                           (symbol, timestamp, openInterest, interval, category) 
                           VALUES (?, ?, ?, ?, ?)""",
                        batch
                    )
                    conn.commit()

                # Update checkpoint
                max_timestamp = df['timestamp'].max()
                if pd.notnull(max_timestamp):
                    self.checkpoints['open_interest'][f"{symbol}_{interval}_{category}"] = max_timestamp.strftime(
                        '%Y-%m-%d %H:%M:%S')
                    self.save_checkpoints()

                self.logger.info(f"Stored {len(new_data)} new/updated open interest records for {symbol} {interval}")

            finally:
                self._return_connection(conn)

        except Exception as e:
            self.logger.error(f"Error storing open interest for {symbol}: {e}")

    def store_funding_rates(self, df_dict: Dict[str, pd.DataFrame], symbol: str, category: str):
        """Store funding rates in database using efficient batch operations"""
        if symbol not in df_dict or df_dict[symbol].empty:
            return

        try:
            df = df_dict[symbol]
            df = self._normalize_timestamps(df, 'fundingRateTimestamp')


            # Check for NaT values and handle them
            if df['fundingRateTimestamp'].isna().any():
                self.logger.warning(f"Found NaT timestamps in {symbol} funding rate data. Filtering...")
                df = df.dropna(subset=['fundingRateTimestamp'])

            if df.empty:
                self.logger.warning(f"No valid funding rate data left for {symbol} after filtering NaT values")
                return

            conn = self._get_connection()
            try:
                # Get existing timestamps to avoid duplicates
                query = """
                    SELECT timestamp 
                    FROM funding_rates 
                    WHERE symbol = ? AND category = ?
                """
                existing_df = pd.read_sql_query(query, conn, params=(symbol, category))

                if not existing_df.empty:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                    existing_timestamps = set(existing_df['timestamp'])

                    # Convert dataframe timestamps for comparison
                    df_timestamps = set(df['fundingRateTimestamp'])

                    # Find unique new timestamps - funding rates are fixed every 8h so less concern about updates
                    # But still allow recent updates in case of corrections
                    now = datetime.now()
                    recent_cutoff = pd.Timestamp(now - timedelta(days=1))

                    # Find timestamps that are either new or recent enough to potentially update
                    new_timestamps = df_timestamps - existing_timestamps
                    recent_existing = {ts for ts in existing_timestamps if pd.Timestamp(ts) >= recent_cutoff}
                    updateable_timestamps = new_timestamps | (df_timestamps & recent_existing)

                    # Filter to only new or updateable records
                    new_df = df[df['fundingRateTimestamp'].isin(updateable_timestamps)]
                else:
                    new_df = df

                if new_df.empty:
                    self.logger.info(f"No new funding rate data for {symbol}")
                    return

                # Prepare data for insertion - use INSERT OR REPLACE to handle updates
                new_data = []
                for _, row in new_df.iterrows():
                    new_data.append((
                        symbol,
                        row['fundingRateTimestamp'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['fundingRateTimestamp'], 'strftime') else str(row['fundingRateTimestamp']),
                        row['fundingRate'],
                        category
                    ))

                # Insert in batches for better performance
                batch_size = self.config.batch_size
                for i in range(0, len(new_data), batch_size):
                    batch = new_data[i:i + batch_size]
                    conn.executemany(
                        """INSERT OR REPLACE INTO funding_rates 
                           (symbol, timestamp, funding_rate, category) 
                           VALUES (?, ?, ?, ?)""",
                        batch
                    )
                    conn.commit()

                # Update checkpoint
                max_timestamp = df['fundingRateTimestamp'].max()
                if pd.notnull(max_timestamp):
                    self.checkpoints['funding_rates'][f"{symbol}_{category}"] = max_timestamp.strftime(
                        '%Y-%m-%d %H:%M:%S')
                    self.save_checkpoints()

                self.logger.info(f"Stored {len(new_data)} new/updated funding rate records for {symbol}")

            finally:
                self._return_connection(conn)

        except Exception as e:
            self.logger.error(f"Error storing funding rates for {symbol}: {e}")

    def update_price_data(self, symbols: List[str], category: str = 'linear', intervals: List[str] = None):
        """Update price data for multiple symbols and intervals in parallel"""
        if intervals is None:
            intervals = self.available_intervals['price_data']
                
        # Get start times for all symbol-interval combinations
        symbol_times = {}
        for symbol in symbols:
            symbol_times[symbol] = {}
            for interval in intervals:
                last_timestamp = self.get_last_timestamp('price_data', symbol, interval, category)
                # Use a small overlap to ensure continuity
                symbol_times[symbol][interval] = last_timestamp - timedelta(minutes=20) if last_timestamp else self.get_default_start_date(symbol)
        
        end_time = datetime.now()
        
        # Process in smaller groups to reduce memory usage
        max_symbols_per_batch = 5
        for i in range(0, len(symbols), max_symbols_per_batch):
            symbol_batch = symbols[i:i+max_symbols_per_batch]
            self.logger.info(f"Processing price data batch {i//max_symbols_per_batch + 1} of {(len(symbols)-1)//max_symbols_per_batch + 1}")
            
            for interval in intervals:
                self.logger.info(f"Fetching {interval} price data for {len(symbol_batch)} symbols")
                
                # Create tasks for this interval
                tasks = []
                for symbol in symbol_batch:
                    start_time = symbol_times[symbol][interval]
                    tasks.append((symbol, start_time, end_time, interval, category))
                
                # Process in parallel
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = {}
                    
                    # Submit all tasks
                    for symbol, start_time, end_time, interval, category in tasks:
                        future = executor.submit(
                            self.fetcher.fetch_multi_symbol_data,
                            symbols=[symbol],
                            start_time=start_time,
                            end_time=end_time,
                            interval=interval,
                            category=category,
                            max_workers=1  # Since we're already parallelizing at the symbol level
                        )
                        futures[future] = symbol
                    
                    # Process results as they complete
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            klines_data = future.result()
                            self.store_price_data(klines_data, symbol, interval, category)
                        except Exception as e:
                            self.logger.error(f"Error processing {symbol} {interval}: {e}")

    def update_open_interest(self, symbols: List[str], category: str = 'linear', intervals: List[str] = None):
        """Update open interest for multiple symbols and intervals in parallel"""
        if intervals is None:
            intervals = self.available_intervals['open_interest']
        
        # Get start times for all symbol-interval combinations
        symbol_times = {}
        for symbol in symbols:
            symbol_times[symbol] = {}
            for interval in intervals:
                last_timestamp = self.get_last_timestamp('open_interest', symbol, interval, category)
                # Use a small overlap to ensure continuity
                symbol_times[symbol][interval] = last_timestamp - timedelta(minutes=20) if last_timestamp else self.get_default_start_date(symbol)
        
        end_time = datetime.now()
        
        # Process in smaller groups to reduce memory usage
        max_symbols_per_batch = 5
        for i in range(0, len(symbols), max_symbols_per_batch):
            symbol_batch = symbols[i:i+max_symbols_per_batch]
            self.logger.info(f"Processing open interest batch {i//max_symbols_per_batch + 1} of {(len(symbols)-1)//max_symbols_per_batch + 1}")
            
            for interval in intervals:
                self.logger.info(f"Fetching {interval} open interest for {len(symbol_batch)} symbols")
                
                # Create tasks for this interval
                tasks = []
                for symbol in symbol_batch:
                    start_time = symbol_times[symbol][interval]
                    tasks.append((symbol, start_time, end_time, interval, category))
                
                # Process in parallel
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = {}
                    
                    # Submit all tasks
                    for symbol, start_time, end_time, interval, category in tasks:
                        future = executor.submit(
                            self.fetcher.fetch_multi_open_interest,
                            symbols=[symbol],
                            start_time=start_time,
                            end_time=end_time,
                            interval=interval,
                            category=category,
                            max_workers=1  # Since we're already parallelizing at the symbol level
                        )
                        futures[future] = symbol
                    
                    # Process results as they complete
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            oi_data = future.result()
                            self.store_open_interest(oi_data, symbol, interval, category)
                        except Exception as e:
                            self.logger.error(f"Error processing OI {symbol} {interval}: {e}")

    def update_funding_rates(self, symbols: List[str], category: str = 'linear'):
        """Update funding rates for multiple symbols in parallel"""
        # Get start times for all symbols
        symbol_times = {}
        for symbol in symbols:
            last_timestamp = self.get_last_timestamp('funding_rates', symbol, category=category)
            # Use a small overlap to ensure continuity
            symbol_times[symbol] = last_timestamp - timedelta(minutes=20) if last_timestamp else self.get_default_start_date(symbol)
        
        end_time = datetime.now()
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            
            # Submit all tasks
            for symbol in symbols:
                start_time = symbol_times[symbol]
                time_range = TimeRange(start_time, end_time)
                
                future = executor.submit(
                    self.fetcher.fetch_funding_rates,
                    symbol=symbol,
                    time_range=time_range,
                    category=category
                )
                futures[future] = symbol
            
            # Process results as they complete
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    funding_data = future.result()
                    
                    # Format data for storage
                    if not funding_data.empty:
                        data_dict = {symbol: funding_data}
                        self.store_funding_rates(data_dict, symbol, category)
                except Exception as e:
                    self.logger.error(f"Error processing funding rates for {symbol}: {e}")

    def update_all_data(self, symbols: List[str], category: str = 'linear', 
                       oi_intervals: List[str] = None,
                       price_intervals: List[str] = None):
        """Update all data types for all symbols with specific intervals."""
        self.logger.info(f"Starting data update for {len(symbols)} symbols")
        
        try:
            # Update price data with specified intervals
            self.logger.info("Updating price data...")
            self.update_price_data(symbols, category, intervals=price_intervals)
            
            # Update funding rates (fixed 8h interval)
            self.logger.info("Updating funding rates...")
            self.update_funding_rates(symbols, category)
            
            # Update open interest with specified intervals
            self.logger.info("Updating open interest...")
            self.update_open_interest(symbols, category, intervals=oi_intervals)
            
            # Optimize database after updates
            self.logger.info("Optimizing database...")
            self.optimize_database()
            
            self.logger.info("Data update completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during data update: {e}")
            
    def schedule_updates(self, symbols: List[str], schedule_time: str = "00:00"):
        """Schedule daily data updates (requires 'pip install schedule')."""
        import schedule as sched
        def job():
            self.logger.info("Running scheduled data update")
            self._backup_database()  # Backup before update
            self.update_all_data(symbols)

        sched.every().day.at(schedule_time).do(job)

        self.logger.info(f"Scheduled daily updates for {len(symbols)} symbols at {schedule_time}")

        while True:
            sched.run_pending()
            time.sleep(60)
            
    def list_symbols(self) -> Dict[str, List[str]]:
        """Get all symbols currently in the database for each data type."""
        try:
            conn = self._get_connection()
            try:
                symbols = {}
                
                # Query each table
                for table in ['price_data', 'funding_rates', 'open_interest']:
                    query = f"SELECT DISTINCT symbol FROM {table}"
                    df = pd.read_sql_query(query, conn)
                    symbols[table] = df['symbol'].tolist()
                
                return symbols
                
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Error listing symbols: {e}")
            return {}
            
    def get_data_ranges(self) -> Dict[str, Dict[str, Dict[str, datetime]]]:
        """Get the date range for each symbol and data type."""
        try:
            conn = self._get_connection()
            try:
                ranges = {}
                
                # Get price data ranges
                query = """
                SELECT 
                    symbol,
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date,
                    GROUP_CONCAT(DISTINCT interval) as intervals
                FROM price_data
                GROUP BY symbol
                """
                df = pd.read_sql_query(query, conn)
                for _, row in df.iterrows():
                    if row['symbol'] not in ranges:
                        ranges[row['symbol']] = {}
                    ranges[row['symbol']]['price_data'] = {
                        'start': pd.to_datetime(row['start_date']),
                        'end': pd.to_datetime(row['end_date']),
                        'intervals': row['intervals'].split(',')
                    }
                
                # Get funding rate ranges
                query = """
                SELECT 
                    symbol,
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date
                FROM funding_rates
                GROUP BY symbol
                """
                df = pd.read_sql_query(query, conn)
                for _, row in df.iterrows():
                    if row['symbol'] not in ranges:
                        ranges[row['symbol']] = {}
                    ranges[row['symbol']]['funding_rates'] = {
                        'start': pd.to_datetime(row['start_date']),
                        'end': pd.to_datetime(row['end_date'])
                    }
                
                # Get open interest ranges
                query = """
                SELECT 
                    symbol,
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date,
                    GROUP_CONCAT(DISTINCT interval) as intervals
                FROM open_interest
                GROUP BY symbol
                """
                df = pd.read_sql_query(query, conn)
                for _, row in df.iterrows():
                    if row['symbol'] not in ranges:
                        ranges[row['symbol']] = {}
                    ranges[row['symbol']]['open_interest'] = {
                        'start': pd.to_datetime(row['start_date']),
                        'end': pd.to_datetime(row['end_date']),
                        'intervals': row['intervals'].split(',')
                    }
                
                return ranges
                
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Error getting data ranges: {e}")
            return {}

    def remove_symbol(self, symbol: str) -> bool:
        """Remove all data for a specific symbol from the database."""
        try:
            # Backup before making major changes
            if self.config.backup_enabled:
                self._backup_database()
                
            conn = self._get_connection()
            try:
                # Remove from all tables
                for table in ['price_data', 'funding_rates', 'open_interest']:
                    conn.execute(f"DELETE FROM {table} WHERE symbol = ?", (symbol,))
                
                conn.commit()
                
                # Remove from checkpoints
                for data_type in self.checkpoints:
                    keys_to_remove = []
                    for key in self.checkpoints[data_type]:
                        if key.startswith(f"{symbol}_"):
                            keys_to_remove.append(key)
                    for key in keys_to_remove:
                        del self.checkpoints[data_type][key]
                
                self.save_checkpoints()
                self.logger.info(f"Successfully removed symbol {symbol} from database")
                return True
                
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Error removing symbol {symbol}: {e}")
            return False

    def verify_data_continuity(self, symbol: str, start_date: datetime = None, 
                             end_date: datetime = None) -> Dict[str, Dict[str, List[datetime]]]:
        """Check for gaps in the data for a given symbol."""
        try:
            if start_date is None:
                start_date = self.get_default_start_date(symbol)
            if end_date is None:
                end_date = datetime.now()
                
            conn = self._get_connection()
            try:
                gaps = {'price_data': {}, 'funding_rates': {}, 'open_interest': {}}
                
                # Check price data
                query = "SELECT DISTINCT interval FROM price_data WHERE symbol = ?"
                intervals = pd.read_sql_query(query, conn, params=(symbol,))['interval'].tolist()
                
                for interval in intervals:
                    query = """
                        SELECT timestamp 
                        FROM price_data 
                        WHERE symbol = ? AND interval = ?
                        ORDER BY timestamp
                    """
                    df = pd.read_sql_query(query, conn, params=(symbol, interval))
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    gaps['price_data'][interval] = self._find_gaps(df['timestamp'], interval)
                
                # Check funding rates (8h intervals)
                query = "SELECT timestamp FROM funding_rates WHERE symbol = ? ORDER BY timestamp"
                df = pd.read_sql_query(query, conn, params=(symbol,))
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    gaps['funding_rates']['8h'] = self._find_gaps(df['timestamp'], '8h')
                
                # Check open interest
                query = "SELECT DISTINCT interval FROM open_interest WHERE symbol = ?"
                intervals = pd.read_sql_query(query, conn, params=(symbol,))['interval'].tolist()
                
                for interval in intervals:
                    query = """
                        SELECT timestamp 
                        FROM open_interest 
                        WHERE symbol = ? AND interval = ?
                        ORDER BY timestamp
                    """
                    df = pd.read_sql_query(query, conn, params=(symbol, interval))
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    gaps['open_interest'][interval] = self._find_gaps(df['timestamp'], interval)
                
                return gaps
                
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Error verifying data continuity: {e}")
            return {}

    def _find_gaps(self, timestamps: pd.Series, interval: str) -> List[datetime]:
        """Helper function to find gaps in timestamp series."""
        if timestamps.empty:
            return []
            
        interval_map = {
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '8h': timedelta(hours=8),
            '1d': timedelta(days=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30)
        }
        
        expected_diff = interval_map.get(interval, timedelta(hours=1))
        gaps = []
        
        # Sort timestamps to ensure sequential analysis
        sorted_timestamps = timestamps.sort_values().reset_index(drop=True)
        
        for i in range(len(sorted_timestamps) - 1):
            diff = sorted_timestamps.iloc[i + 1] - sorted_timestamps.iloc[i]
            if diff > expected_diff * 1.5:  # Allow some tolerance
                # Record the gap start and end
                gap_start = sorted_timestamps.iloc[i]
                gap_end = sorted_timestamps.iloc[i + 1]
                gaps.append((gap_start, gap_end, diff))
                
        return gaps

    def get_database_stats(self) -> Dict:
        """Get statistics about the database."""
        try:
            conn = self._get_connection()
            try:
                stats = {
                    'symbols': self.list_symbols(),
                    'date_ranges': self.get_data_ranges(),
                    'record_counts': {},
                    'disk_usage': Path(self.config.db_path).stat().st_size / (1024 * 1024)  # MB
                }
                
                # Get record counts for each table
                for table in ['price_data', 'funding_rates', 'open_interest']:
                    count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", conn)
                    stats['record_counts'][table] = count['count'].iloc[0]
                    
                # Get count by symbol for each table
                for table in ['price_data', 'funding_rates', 'open_interest']:
                    query = f"""
                        SELECT symbol, COUNT(*) as count 
                        FROM {table} 
                        GROUP BY symbol
                    """
                    df = pd.read_sql_query(query, conn)
                    if not df.empty:
                        symbol_counts = dict(zip(df['symbol'], df['count']))
                        stats[f'{table}_by_symbol'] = symbol_counts
                
                return stats
                
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}

    def get_available_intervals(self, data_type: str = None) -> Dict[str, List[str]]:
        """Get available intervals for each data type or specific data type."""
        if data_type:
            return {data_type: self.available_intervals[data_type]}
        return self.available_intervals.copy()
    
    def query_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None,
                 interval: str = None, data_type: str = 'price_data',
                 category: str = 'linear') -> pd.DataFrame:
        """
        Query data from the database with various filters.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for query (default: symbol's first available data)
            end_date: End date for query (default: now)
            interval: Time interval (for price_data and open_interest)
            data_type: Type of data to query ('price_data', 'funding_rates', 'open_interest')
            category: Market category ('spot', 'linear', 'inverse')
            
        Returns:
            DataFrame with requested data
        """
        try:
            if start_date is None:
                start_date = self.get_default_start_date(symbol)
            if end_date is None:
                end_date = datetime.now()
                
            conn = self._get_connection()
            try:
                if data_type == 'price_data':
                    query = """
                        SELECT * FROM price_data 
                        WHERE symbol = ? AND category = ?
                        AND timestamp BETWEEN ? AND ?
                    """
                    params = [symbol, category, start_date, end_date]
                    
                    if interval:
                        query += " AND interval = ?"
                        params.append(interval)
                        
                elif data_type == 'funding_rates':
                    query = """
                        SELECT * FROM funding_rates 
                        WHERE symbol = ? AND category = ?
                        AND timestamp BETWEEN ? AND ?
                    """
                    params = [symbol, category, start_date, end_date]
                    
                elif data_type == 'open_interest':
                    query = """
                        SELECT * FROM open_interest 
                        WHERE symbol = ? AND category = ?
                        AND timestamp BETWEEN ? AND ?
                    """
                    params = [symbol, category, start_date, end_date]
                    
                    if interval:
                        query += " AND interval = ?"
                        params.append(interval)
                        
                query += " ORDER BY timestamp"
                df = pd.read_sql_query(query, conn, params=params)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                return df
                
            finally:
                self._return_connection(conn)
                
        except Exception as e:
            self.logger.error(f"Error querying data: {e}")
            return pd.DataFrame()
    
    def export_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None,
                  data_types: List[str] = None, intervals: Dict[str, List[str]] = None,
                  file_format: str = 'csv', output_dir: str = 'exports') -> Dict[str, str]:
        """
        Export data to files.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for export
            end_date: End date for export
            data_types: List of data types to export (default: all)
            intervals: Dict mapping data types to intervals (default: all available)
            file_format: Export format ('csv' or 'parquet')
            output_dir: Directory to save exported files
            
        Returns:
            Dict mapping data types to exported file paths
        """
        try:
            if data_types is None:
                data_types = ['price_data', 'funding_rates', 'open_interest']
                
            if intervals is None:
                intervals = {
                    'price_data': self.available_intervals['price_data'],
                    'open_interest': self.available_intervals['open_interest'],
                    'funding_rates': ['8h']  # Fixed by Bybit
                }
                
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            exported_files = {}
            
            for data_type in data_types:
                if data_type == 'funding_rates':
                    # Funding rates don't have intervals
                    df = self.query_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        data_type=data_type
                    )
                    
                    if not df.empty:
                        filename = f"{symbol}_{data_type}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                        filepath = output_path / f"{filename}.{file_format}"
                        
                        if file_format == 'csv':
                            df.to_csv(filepath, index=False)
                        elif file_format == 'parquet':
                            df.to_parquet(filepath, index=False)
                            
                        exported_files[data_type] = str(filepath)
                        
                else:
                    # Price data and open interest have intervals
                    for interval in intervals.get(data_type, []):
                        df = self.query_data(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            interval=interval,
                            data_type=data_type
                        )
                        
                        if not df.empty:
                            filename = f"{symbol}_{data_type}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                            filepath = output_path / f"{filename}.{file_format}"
                            
                            if file_format == 'csv':
                                df.to_csv(filepath, index=False)
                            elif file_format == 'parquet':
                                df.to_parquet(filepath, index=False)
                                
                            if data_type not in exported_files:
                                exported_files[data_type] = {}
                                
                            exported_files[data_type][interval] = str(filepath)
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return {}

def main():
    """Fetch daily price data for the configured symbol list."""
    try:
        import config as cfg
        symbols = cfg.DEFAULT_SYMBOLS
        db_path = str(cfg.DB_PATH)
    except ImportError:
        symbols = [
            'AVAXUSDT', 'BNBUSDT', 'SOLUSDT', 'BTCUSDT', 'DOGEUSDT',
            'ETHUSDT', 'HYPEUSDT', 'ADAUSDT', 'LINKUSDT', 'TRXUSDT',
            'SUIUSDT', 'BCHUSDT', 'XLMUSDT', 'XRPUSDT', 'TONUSDT',
        ]
        db_path = 'crypto_data.db'

    config = DataConfig(
        db_path=db_path, log_path='collection.log',
        checkpoint_path='checkpoints.json',
        max_workers=10, batch_size=1000, backup_enabled=True,
    )
    collector = CryptoDataCollector(config)
    collector.update_price_data(symbols, category='linear', intervals=['1d'])

if __name__ == "__main__":
    main()

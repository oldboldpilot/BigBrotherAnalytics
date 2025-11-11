# Agent Coding Guide - BigBrotherAnalytics

This guide provides code templates and patterns for agents to create code matching the existing codebase style.

## Quick Start for New Code

### 1. Data Collection Script Template

```python
#!/usr/bin/env python3
"""
BigBrotherAnalytics: [Feature Name]
[Description of what this script does]

API Documentation: [API docs link]
Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import duckdb
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureCollector:
    """Collects [data type] from [API source]."""
    
    API_URL = "https://api.example.com/endpoint"
    
    def __init__(self, api_key: Optional[str] = None, db_path: str = "data/bigbrother.duckdb"):
        """
        Initialize collector.
        
        Args:
            api_key: API key (optional, reads from api_keys.yaml or env variable)
            db_path: Path to DuckDB database
        """
        # Try to load from api_keys.yaml first, then env variable
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._load_api_key_from_config() or os.getenv('FEATURE_API_KEY')
        
        self.db_path = Path(db_path)
        
        # Ensure database exists
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True)
        
        logger.info("Feature Collector initialized")
    
    def _load_api_key_from_config(self) -> Optional[str]:
        """Load API key from api_keys.yaml configuration file."""
        try:
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'api_keys.yaml'
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('feature_api_key')
        except Exception as e:
            logger.debug(f"Could not load API key from config: {e}")
        
        return None
    
    def fetch_data(self, params: Dict) -> Dict:
        """
        Fetch data from API.
        
        Args:
            params: Request parameters
        
        Returns:
            API response as dictionary
        """
        try:
            response = requests.get(self.API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Successfully fetched data")
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            return {}
    
    def parse_data(self, raw_data: Dict) -> List[Tuple]:
        """
        Parse API response into database row format.
        
        Args:
            raw_data: Raw API response
        
        Returns:
            List of tuples ready for database insertion
        """
        rows = []
        
        for item in raw_data.get('items', []):
            # Parse item and create tuple
            try:
                row = (item.get('field1'), item.get('field2'))
                rows.append(row)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse item: {e}")
                continue
        
        return rows
    
    def store_data(self, rows: List[Tuple]) -> None:
        """
        Store parsed data in DuckDB.
        
        Args:
            rows: List of (field1, field2, ...) tuples
        """
        conn = duckdb.connect(str(self.db_path))
        
        try:
            for row in rows:
                conn.execute(
                    "INSERT INTO table_name (col1, col2) VALUES (?, ?)",
                    row
                )
            
            logger.info(f"Stored {len(rows)} rows")
        
        except Exception as e:
            logger.error(f"Error storing data: {e}")
        
        finally:
            conn.close()
    
    def collect(self) -> None:
        """Main collection method."""
        logger.info("Starting data collection")
        
        params = {
            'param1': 'value1',
            'param2': 'value2'
        }
        
        data = self.fetch_data(params)
        if data:
            rows = self.parse_data(data)
            self.store_data(rows)
        
        logger.info("Collection complete")


def main():
    """Main entry point."""
    logger.info("╔════════════════════════════════════════╗")
    logger.info("║    Feature Collection Started          ║")
    logger.info("╚════════════════════════════════════════╝")
    
    collector = FeatureCollector()
    collector.collect()
    
    logger.info("Process complete")


if __name__ == "__main__":
    main()
```

### 2. Dashboard View Function Template

```python
def show_feature_view():
    """Display feature-specific dashboard view."""
    st.header("Feature View Header")
    
    # Get database connection
    conn = get_db_connection()
    
    # Load data from database
    query = """
        SELECT col1, col2, col3
        FROM table_name
        WHERE date >= CURRENT_DATE - INTERVAL 30 DAYS
        ORDER BY date DESC
    """
    
    df = conn.execute(query).df()
    
    if df.empty:
        st.info("No data available")
        return
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Metric 1", f"${df['col1'].sum():,.2f}")
    
    with col2:
        st.metric("Metric 2", len(df))
    
    with col3:
        st.metric("Metric 3", f"{df['col2'].mean():.2f}")
    
    st.divider()
    
    # Display visualization
    fig = px.line(df, x='col1', y='col2', title='Feature Trends')
    st.plotly_chart(fig, use_container_width=True)
    
    # Display data table
    st.subheader("Data Table")
    st.dataframe(df, use_container_width=True, hide_index=True)
```

### 3. Automated Update Script Template

```python
#!/usr/bin/env python3
"""
BigBrotherAnalytics: [Feature] Daily Update
[Description of automated update process]

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
DB_PATH = BASE_DIR / 'data' / 'bigbrother.duckdb'
LOG_DIR = BASE_DIR / 'logs' / 'automated_updates'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / f'feature_update_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FeatureUpdater:
    """Handles daily feature updates."""
    
    def __init__(self, test_mode: bool = False):
        """
        Initialize updater.
        
        Args:
            test_mode: If True, report only without modifying database
        """
        self.db_path = DB_PATH
        self.test_mode = test_mode
        self.update_summary = {
            'timestamp': datetime.now().isoformat(),
            'updates_performed': [],
            'errors': []
        }
        
        logger.info("=" * 80)
        logger.info("Feature Daily Update")
        logger.info(f"Mode: {'TEST' if test_mode else 'LIVE'}")
        logger.info(f"Database: {self.db_path}")
        logger.info("=" * 80)
    
    def should_run(self) -> bool:
        """Check if update should run today."""
        # Example: Run on specific day of week
        today = datetime.now().weekday()
        return today == 4  # Friday
    
    def update_feature(self) -> bool:
        """
        Perform feature update.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting feature update...")
            
            # Import collector
            from scripts.data_collection.feature_collector import FeatureCollector
            
            # Fetch data
            collector = FeatureCollector(db_path=str(self.db_path))
            collector.collect()
            
            logger.info("Feature update completed successfully")
            self.update_summary['updates_performed'].append('feature_update')
            return True
        
        except Exception as e:
            logger.error(f"Feature update failed: {e}")
            self.update_summary['errors'].append(f"feature_update: {str(e)}")
            return False
    
    def run(self) -> None:
        """Run all updates."""
        if not self.should_run():
            logger.info("Not scheduled to run today")
            return
        
        logger.info("Running daily updates...")
        
        # Run updates
        success = self.update_feature()
        
        # Generate report
        logger.info("=" * 80)
        if success:
            logger.info("All updates completed successfully")
        else:
            logger.warning(f"Updates completed with errors: {self.update_summary['errors']}")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature daily update script')
    parser.add_argument('--test', action='store_true', help='Run in test mode (report only)')
    args = parser.parse_args()
    
    updater = FeatureUpdater(test_mode=args.test)
    updater.run()


if __name__ == "__main__":
    main()
```

### 4. Configuration Loading Pattern

```python
import yaml
from pathlib import Path
import os

class ConfigManager:
    """Manage configuration loading from YAML files."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            base_dir: Base directory (defaults to project root)
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent
        else:
            self.base_dir = base_dir
        
        self.config_dir = self.base_dir / 'configs'
    
    def load_api_keys(self) -> Dict:
        """Load API keys from configs/api_keys.yaml."""
        api_keys_file = self.config_dir / 'api_keys.yaml'
        
        if api_keys_file.exists():
            with open(api_keys_file, 'r') as f:
                return yaml.safe_load(f) or {}
        
        return {}
    
    def load_config(self, config_name: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_name: Config file name (e.g., 'config.yaml')
        
        Returns:
            Configuration dictionary
        """
        config_file = self.config_dir / config_name
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        
        return {}
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get API key by name.
        
        Priority:
        1. configs/api_keys.yaml
        2. Environment variable (uppercase with underscores)
        
        Args:
            key_name: Name of API key (e.g., 'fred_api_key')
        
        Returns:
            API key value or None
        """
        # Try config file
        api_keys = self.load_api_keys()
        if key_name in api_keys:
            return api_keys[key_name]
        
        # Try environment variable
        env_var_name = key_name.upper()
        return os.getenv(env_var_name)
```

### 5. Database Operations Template

```python
import duckdb
from pathlib import Path
from typing import List, Dict

class DatabaseManager:
    """Manage DuckDB operations."""
    
    def __init__(self, db_path: str = "data/bigbrother.duckdb"):
        """Initialize database manager."""
        self.db_path = Path(db_path)
    
    def connect(self, read_only: bool = False):
        """Create database connection."""
        return duckdb.connect(str(self.db_path), read_only=read_only)
    
    def execute_query(self, query: str, params: List = None) -> List:
        """
        Execute query and return results.
        
        Args:
            query: SQL query
            params: Query parameters
        
        Returns:
            List of tuples
        """
        conn = self.connect(read_only=True)
        try:
            if params:
                result = conn.execute(query, params).fetchall()
            else:
                result = conn.execute(query).fetchall()
            return result
        finally:
            conn.close()
    
    def execute_query_df(self, query: str, params: List = None):
        """
        Execute query and return pandas DataFrame.
        
        Args:
            query: SQL query
            params: Query parameters
        
        Returns:
            pandas DataFrame
        """
        conn = self.connect(read_only=True)
        try:
            if params:
                return conn.execute(query, params).df()
            else:
                return conn.execute(query).df()
        finally:
            conn.close()
    
    def insert_data(self, table: str, columns: List[str], rows: List[Tuple]) -> int:
        """
        Insert data into table.
        
        Args:
            table: Table name
            columns: Column names
            rows: List of (val1, val2, ...) tuples
        
        Returns:
            Number of rows inserted
        """
        conn = self.connect(read_only=False)
        try:
            placeholders = ','.join(['?'] * len(columns))
            col_names = ','.join(columns)
            query = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
            
            count = 0
            for row in rows:
                conn.execute(query, row)
                count += 1
            
            return count
        finally:
            conn.close()
```

## Common Patterns

### Error Handling for API Calls

```python
import requests
import logging

logger = logging.getLogger(__name__)

def fetch_with_retry(url: str, params: Dict, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch data from API with retry logic.
    
    Args:
        url: API endpoint URL
        params: Query parameters
        max_retries: Maximum number of retry attempts
    
    Returns:
        Response JSON or None if failed
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"All retry attempts failed for {url}")
                return None
```

### Date Range Queries

```python
from datetime import datetime, timedelta

def get_recent_data(conn, days_back: int = 7) -> DataFrame:
    """Get data from last N days."""
    query = f"""
        SELECT *
        FROM table_name
        WHERE date >= CURRENT_DATE - INTERVAL {days_back} DAYS
        ORDER BY date DESC
    """
    return conn.execute(query).df()

def get_month_to_date(conn) -> DataFrame:
    """Get data from beginning of current month."""
    query = """
        SELECT *
        FROM table_name
        WHERE date >= DATE_TRUNC('month', CURRENT_DATE)
        ORDER BY date DESC
    """
    return conn.execute(query).df()
```

## Key Files to Reference

When creating code, reference these existing files:

1. **Data Collection Pattern**: `/home/muyiwa/Development/BigBrotherAnalytics/scripts/data_collection/bls_employment.py`
2. **Dashboard Pattern**: `/home/muyiwa/Development/BigBrotherAnalytics/dashboard/app.py`
3. **Automated Updates**: `/home/muyiwa/Development/BigBrotherAnalytics/scripts/automated_updates/daily_update.py`
4. **Configuration**: `/home/muyiwa/Development/BigBrotherAnalytics/configs/config.yaml`

## Testing Your Code

```python
#!/usr/bin/env python3
"""Test script for new code."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_feature():
    """Test feature collector."""
    from scripts.data_collection.feature_collector import FeatureCollector
    
    # Test with small data set
    collector = FeatureCollector()
    
    # Verify connection
    assert collector.api_key is not None, "API key not loaded"
    assert collector.db_path.parent.exists(), "DB directory doesn't exist"
    
    # Test fetch
    data = collector.fetch_data({'param': 'value'})
    assert isinstance(data, dict), "Fetch didn't return dict"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_feature()
```

## C++ Module Development (C++23)

### Module Template

```cpp
/**
 * BigBrotherAnalytics - [Feature Name] Module (C++23)
 *
 * [Description of what this module does]
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 * Phase: [Phase number]
 *
 * Following C++ Core Guidelines:
 * - I.2: Avoid non-const global variables
 * - F.51: Prefer default arguments over overloading
 * - Trailing return type syntax
 */

// Global module fragment
module;

#include <string>
#include <vector>
#include <expected>

// Module declaration
export module bigbrother.feature.name;

// Import dependencies
import bigbrother.utils.types;
import bigbrother.utils.logger;

export namespace bigbrother::feature {

using namespace bigbrother::types;
using namespace bigbrother::utils;

// ============================================================================
// Data Types
// ============================================================================

/**
 * [Description]
 * C.1: Use struct for passive data
 */
struct DataType {
    std::string field1;
    int field2{0};
};

// ============================================================================
// Main Class
// ============================================================================

/**
 * [Description of class purpose]
 *
 * C.2: Use class when invariants exist
 */
class FeatureClass {
public:
    /**
     * Constructor
     *
     * @param param1 Description
     */
    explicit FeatureClass(std::string param1)
        : param1_(std::move(param1)) {
        Logger::getInstance().info("Feature initialized");
    }

    /**
     * Main method
     *
     * @param input Description
     * @return Result or error
     *
     * F.20: Return by value
     * [[nodiscard]]: Result must be used
     */
    [[nodiscard]] auto process(std::string const& input)
        -> std::expected<DataType, Error> {

        if (input.empty()) {
            return std::unexpected(Error::make(
                ErrorCode::InvalidArgument,
                "Input cannot be empty"
            ));
        }

        // Process and return
        DataType result{input, 42};
        return result;
    }

    // Rule of Five (C.21)
    ~FeatureClass() = default;
    FeatureClass(FeatureClass const&) = delete;
    FeatureClass& operator=(FeatureClass const&) = delete;
    FeatureClass(FeatureClass&&) = delete;
    FeatureClass& operator=(FeatureClass&&) = delete;

private:
    std::string param1_;
};

} // namespace bigbrother::feature
```

### CMake Module Configuration

```cmake
# Add library with C++23 modules
add_library(feature_lib)
target_sources(feature_lib
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/feature/feature_name.cppm
)

# Link dependencies
target_link_libraries(feature_lib
    PUBLIC utils
    PRIVATE CURL::libcurl  # Example external lib
)

# Set C++23 standard
target_compile_features(feature_lib PUBLIC cxx_std_23)
```

### Building C++23 Modules with Ninja

**Prerequisites**:
- CMake 3.28+
- Ninja build system
- Clang 21+ with libc++
- clang-tidy (for pre-build validation)

**Build Commands**:
```bash
# Configure with Ninja (auto-detects clang and libc++)
cd build
cmake -G Ninja ..

# Build all modules
ninja

# Build specific target
ninja feature_lib

# Run tests
ninja test

# Module files generated at: build/modules/*.pcm
```

**Module Dependency Order**:
Ninja automatically handles module dependencies based on imports:
1. types (no dependencies)
2. logger (depends on types)
3. database (depends on types, logger)
4. sentiment (depends on types, logger)
5. news (depends on types, logger, database, sentiment)
6. bindings (depends on news)

**clang-tidy Validation**:
CMake automatically runs clang-tidy before building:
- Checks C++ Core Guidelines compliance
- Validates trailing return types
- Enforces [[nodiscard]] on query methods
- Build fails if violations found (blocking)

### pybind11 Python Bindings

**Template**:
```cpp
/**
 * BigBrotherAnalytics - Python Bindings for [Feature]
 *
 * Exposes C++ classes to Python via pybind11
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-10
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For std::vector, std::string
#include <pybind11/chrono.h>  // For timestamps

namespace py = pybind11;

// Import C++ modules
import bigbrother.feature.name;

using namespace bigbrother::feature;

PYBIND11_MODULE(feature_py, m) {
    m.doc() = "Python bindings for feature";

    // Expose struct
    py::class_<DataType>(m, "DataType")
        .def(py::init<>())
        .def_readwrite("field1", &DataType::field1)
        .def_readwrite("field2", &DataType::field2);

    // Expose class
    py::class_<FeatureClass>(m, "FeatureClass")
        .def(py::init<std::string>(), py::arg("param1"))
        .def("process", &FeatureClass::process, py::arg("input"))
        .def("__repr__", [](FeatureClass const& fc) {
            return "<FeatureClass>";
        });

    // Expose enum (if any)
    py::enum_<ErrorCode>(m, "ErrorCode")
        .value("Success", ErrorCode::Success)
        .value("InvalidArgument", ErrorCode::InvalidArgument);
}
```

**CMake Configuration**:
```cmake
# Python bindings module
pybind11_add_module(feature_py src/python_bindings/feature_bindings.cpp)
target_link_libraries(feature_py PRIVATE feature_lib)
```

**Using in Python**:
```python
import feature_py

# Create instance
obj = feature_py.FeatureClass("param_value")

# Call methods
result = obj.process("input_data")

# Access struct fields
data = feature_py.DataType()
data.field1 = "value"
data.field2 = 42
```

### Error Handling Pattern

**C++ Side**:
```cpp
auto fetch_data() -> std::expected<Data, Error> {
    if (error_condition) {
        return std::unexpected(Error::make(
            ErrorCode::NetworkError,
            "Failed to connect to API"
        ));
    }

    Data result{...};
    return result;
}

// Chaining operations
auto result = fetch_data()
    .and_then([](Data const& d) { return process_data(d); })
    .or_else([](Error const& e) {
        Logger::getInstance().error("Error: {}", e.message);
        return std::expected<ProcessedData, Error>(
            std::unexpected(e)
        );
    });
```

**Python Side** (via bindings):
```python
try:
    result = collector.fetch_data()
    if not result:  # Check for error
        logger.error(f"Failed: {result.error()}")
        return

    # Success - use result
    data = result.value()
except Exception as e:
    logger.error(f"Exception: {e}")
```

### Best Practices for C++23 Modules

1. **Always Use Trailing Return Type**:
   ```cpp
   // Good
   auto calculate() -> double { return 42.0; }

   // Bad
   double calculate() { return 42.0; }
   ```

2. **Use [[nodiscard]] for Query Methods**:
   ```cpp
   // Good
   [[nodiscard]] auto get_value() const -> int { return value_; }

   // Bad
   int get_value() const { return value_; }
   ```

3. **Prefer std::expected Over Exceptions**:
   ```cpp
   // Good
   auto parse(std::string const& s) -> std::expected<int, Error>;

   // Bad
   int parse(std::string const& s);  // throws on error
   ```

4. **Follow Rule of Five**:
   ```cpp
   class Resource {
   public:
       ~Resource();                               // Destructor
       Resource(Resource const&) = delete;        // Copy constructor
       Resource& operator=(Resource const&) = delete;  // Copy assignment
       Resource(Resource&&) = delete;             // Move constructor
       Resource& operator=(Resource&&) = delete;  // Move assignment
   };
   ```

5. **Use const& for Input Parameters**:
   ```cpp
   // Good
   auto process(std::string const& input) -> Result;

   // Bad
   auto process(std::string input) -> Result;  // Unnecessary copy
   ```

6. **Return by Value for Output**:
   ```cpp
   // Good
   auto get_articles() -> std::vector<Article>;

   // Bad
   auto get_articles() -> std::vector<Article> const&;  // Dangling reference
   ```

### Module Dependency Management

**Declare Dependencies Correctly**:
```cpp
// In news_ingestion.cppm
export module bigbrother.market_intelligence.news;

// Import ALL dependencies used in this module
import bigbrother.utils.types;        // For Timestamp, ErrorCode
import bigbrother.utils.logger;       // For Logger
import bigbrother.utils.database;     // For DatabaseManager
import bigbrother.market_intelligence.sentiment;  // For SentimentAnalyzer
```

**Avoid Circular Dependencies**:
- Base utilities (types, logger) should not import anything
- Higher-level modules import base modules
- Never have A import B while B imports A

**Build Order Matters**:
CMake + Ninja handles this automatically, but be aware:
1. Base modules (types, logger) build first
2. Utility modules (database, config) build next
3. Feature modules (sentiment, news) build last
4. Python bindings build after all C++ modules

## Documentation Requirements

Every module should have:

1. Module docstring with description and author
2. Module-level logger
3. Class docstrings with purpose
4. Method docstrings with Args and Returns
5. Error handling with logging

### Python Documentation

```python
#!/usr/bin/env python3
"""
BigBrotherAnalytics: [Feature Name]
[Detailed description of what this module does]

Features:
- List key features

Data Sources:
- Source 1: [Description]
- Source 2: [Description]

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
"""

import logging

logger = logging.getLogger(__name__)

class MyClass:
    """
    [Description of class purpose]

    Attributes:
        attr1: [Description]
        attr2: [Description]
    """

    def method1(self, param1: str) -> bool:
        """
        [Description of method]

        Args:
            param1: [Description]

        Returns:
            [Description of return value]

        Raises:
            [Exception types that might be raised]
        """
        pass
```

### C++ Documentation

```cpp
/**
 * Brief description
 *
 * Detailed description with multiple lines if needed.
 *
 * @param param1 Description
 * @param param2 Description
 * @return Description of return value
 *
 * Example usage:
 * @code
 * auto result = function(arg1, arg2);
 * @endcode
 *
 * Following C++ Core Guidelines:
 * - Guideline 1
 * - Guideline 2
 */
auto function(Type1 const& param1, Type2 param2) -> ReturnType;
```


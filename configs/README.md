# Configuration Directory

Configuration files for the trading platform.

## Files

### config.yaml
Main configuration file
- Database paths
- API endpoints
- Feature flags
- System parameters

### logging.yaml
Logging configuration
- Log levels per module
- Output formats
- File rotation settings

### strategies.yaml
Trading strategy configurations
- Strategy parameters
- Risk limits
- Position sizing rules

### api_keys.yaml.template
Template for API credentials (NEVER commit actual keys)
- Schwab API credentials
- News API keys
- Data provider keys

## Security Best Practices

1. **NEVER commit actual API keys or credentials**
2. Use `api_keys.yaml.template` as template
3. Actual keys should be in `api_keys.yaml` (add to .gitignore)
4. Use environment variables for sensitive data in production
5. Encrypt credentials at rest

## Configuration Loading

```python
from src.utils.config import load_config

config = load_config('configs/config.yaml')
```

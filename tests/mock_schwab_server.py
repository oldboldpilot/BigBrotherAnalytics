"""
Mock Schwab API Server for Testing

Provides mock implementations of Schwab API endpoints without requiring
real API credentials. All responses are based on actual Schwab API response formats.

Endpoints mocked:
- POST /oauth/token - OAuth token exchange and refresh
- GET /marketdata/v1/quotes - Single/multiple symbol quotes
- GET /marketdata/v1/chains - Options chains
- GET /marketdata/v1/pricehistory - Historical OHLCV data
- POST /trader/v1/accounts/{accountId}/orders - Place orders
- GET /trader/v1/accounts/{accountId}/positions - Get account positions
- GET /trader/v1/accounts - Get all accounts

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-09
"""

import json
import time
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - MockServer - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Order statuses"""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELED = "CANCELED"


@dataclass
class MockPosition:
    """Mock position data"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    is_bot_managed: bool = False
    account_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "averagePrice": self.average_price,
            "currentPrice": self.current_price,
            "isBotManaged": self.is_bot_managed,
            "marketValue": self.quantity * self.current_price,
            "gainLoss": (self.current_price - self.average_price) * self.quantity
        }


@dataclass
class MockOrder:
    """Mock order data"""
    order_id: str
    account_id: str
    symbol: str
    quantity: int
    price: Optional[float]
    order_type: str
    status: str
    created_at: str
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "orderId": self.order_id,
            "accountId": self.account_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.price,
            "orderType": self.order_type,
            "status": self.status,
            "createdAt": self.created_at,
            "dryRun": self.dry_run
        }


class MockSchwabAPI:
    """Mock Schwab API backend"""

    def __init__(self):
        """Initialize mock API"""
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.positions: Dict[str, List[MockPosition]] = {}
        self.orders: Dict[str, List[MockOrder]] = {}
        self.order_counter = 0

        # Initialize test accounts
        self.accounts = {
            "XXXX1234": {
                "accountId": "XXXX1234",
                "displayName": "Paper Trading Account",
                "accountType": "CASH",
                "accountStatus": "OPEN",
                "buying_power": 30000.00,
                "cash_balance": 15000.00,
                "securities_value": 15000.00
            }
        }

        # Initialize positions and orders for each account
        for account_id in self.accounts:
            self.positions[account_id] = [
                MockPosition("SPY", 10, 450.00, 455.00, is_bot_managed=True, account_id=account_id),
                MockPosition("QQQ", 5, 380.00, 385.00, is_bot_managed=False, account_id=account_id),
            ]
            self.orders[account_id] = []

    def exchange_token(self, client_id: str, code: str, code_verifier: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        logger.info(f"Token exchange: client_id={client_id}, code={code}")

        # Generate mock tokens
        access_token = f"mock_access_{hashlib.md5(f'{client_id}{code}'.encode()).hexdigest()}"
        refresh_token = f"mock_refresh_{hashlib.md5(f'{client_id}{code_verifier}'.encode()).hexdigest()}"

        # Store tokens
        self.tokens[client_id] = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 1800,
            "expires_at": (datetime.now() + timedelta(minutes=30)).isoformat(),
            "scope": "api",
            "refresh_token_expires_in": 7776000  # 90 days
        }

        return self.tokens[client_id]

    def refresh_token(self, client_id: str, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token"""
        logger.info(f"Token refresh: client_id={client_id}")

        if client_id not in self.tokens:
            raise ValueError("Invalid client_id")

        # Generate new access token
        new_access_token = f"mock_access_{hashlib.md5(f'{client_id}{time.time()}'.encode()).hexdigest()}"

        # Update tokens
        self.tokens[client_id]["access_token"] = new_access_token
        self.tokens[client_id]["expires_at"] = (datetime.now() + timedelta(minutes=30)).isoformat()

        return self.tokens[client_id]

    def validate_token(self, token: str) -> bool:
        """Validate access token"""
        for client_data in self.tokens.values():
            if client_data.get("access_token") == token:
                return True
        return False

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get single stock quote"""
        logger.info(f"Get quote: symbol={symbol}")

        # Generate realistic mock data
        base_prices = {
            "SPY": 455.00,
            "QQQ": 385.00,
            "XLE": 72.50,
            "XLV": 142.00,
            "XLK": 198.50,
            "GOOGL": 145.30,
            "MSFT": 410.00,
            "AAPL": 230.00,
        }

        price = base_prices.get(symbol, 100.00)

        return {
            "symbol": symbol,
            "quote": {
                "bid": price - 0.02,
                "ask": price + 0.02,
                "bidSize": 1000,
                "askSize": 1500,
                "lastPrice": price,
                "lastSize": 100,
                "lastTrade": datetime.now().isoformat(),
                "bidTime": datetime.now().isoformat(),
                "askTime": datetime.now().isoformat(),
                "change": 1.50,
                "changePercent": 0.33,
                "openPrice": price - 2.00,
                "highPrice": price + 3.00,
                "lowPrice": price - 3.50,
                "closePrice": price,
                "volume": 50000000,
                "openInterest": 0,
                "totalVolume": 50000000,
                "volatility": 0.15
            }
        }

    def get_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get multiple stock quotes"""
        logger.info(f"Get quotes: symbols={symbols}")

        result = {}
        for symbol in symbols:
            result[symbol] = self.get_quote(symbol)
        return result

    def get_options_chain(self, symbol: str, contract_type: str = "ALL") -> Dict[str, Any]:
        """Get options chain"""
        logger.info(f"Get options chain: symbol={symbol}, contract_type={contract_type}")

        base_price = 100.00
        expiry_dates = [
            (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d"),
        ]

        expirations = {}
        for expiry in expiry_dates:
            strikes = {}
            for strike in [95.00, 100.00, 105.00, 110.00]:
                if contract_type in ["CALL", "ALL"]:
                    strikes[str(strike)] = {
                        "call": {
                            "symbol": f"{symbol}{expiry.replace('-', '')}C{int(strike)}",
                            "bid": max(0, base_price - strike) + 0.50,
                            "ask": max(0, base_price - strike) + 0.75,
                            "last": max(0, base_price - strike) + 0.60,
                            "volume": 500,
                            "openInterest": 1000,
                            "impliedVolatility": 0.25
                        }
                    }
                if contract_type in ["PUT", "ALL"]:
                    strikes[str(strike)]["put"] = {
                        "symbol": f"{symbol}{expiry.replace('-', '')}P{int(strike)}",
                        "bid": max(0, strike - base_price) + 0.50,
                        "ask": max(0, strike - base_price) + 0.75,
                        "last": max(0, strike - base_price) + 0.60,
                        "volume": 500,
                        "openInterest": 1200,
                        "impliedVolatility": 0.25
                    }
                expirations[expiry] = strikes

        return {
            "symbol": symbol,
            "status": "SUCCESS",
            "expirations": expirations
        }

    def get_price_history(self, symbol: str, period_type: str = "month",
                         period: int = 1) -> Dict[str, Any]:
        """Get historical OHLCV data"""
        logger.info(f"Get price history: symbol={symbol}, period={period} {period_type}")

        candles = []
        base_price = 100.00

        # Generate 30 candles
        for i in range(30):
            timestamp = int((datetime.now() - timedelta(days=30-i)).timestamp() * 1000)
            open_price = base_price + (i * 0.5)
            close_price = open_price + (i % 3 - 1) * 0.25

            candles.append({
                "open": open_price,
                "high": max(open_price, close_price) + 0.50,
                "low": min(open_price, close_price) - 0.25,
                "close": close_price,
                "volume": 1000000 + (i * 50000),
                "datetime": timestamp
            })

        return {
            "symbol": symbol,
            "candles": candles,
            "empty": False
        }

    def get_accounts(self) -> List[Dict[str, Any]]:
        """Get all accounts"""
        logger.info("Get accounts")
        return list(self.accounts.values())

    def get_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get single account"""
        logger.info(f"Get account: account_id={account_id}")
        return self.accounts.get(account_id)

    def get_positions(self, account_id: str) -> List[Dict[str, Any]]:
        """Get account positions"""
        logger.info(f"Get positions: account_id={account_id}")

        if account_id not in self.positions:
            return []

        return [pos.to_dict() for pos in self.positions[account_id]]

    def add_position(self, account_id: str, position: MockPosition):
        """Add a position to account"""
        if account_id not in self.positions:
            self.positions[account_id] = []
        self.positions[account_id].append(position)

    def remove_position(self, account_id: str, symbol: str):
        """Remove position from account"""
        if account_id in self.positions:
            self.positions[account_id] = [
                p for p in self.positions[account_id] if p.symbol != symbol
            ]

    def place_order(self, account_id: str, symbol: str, quantity: int,
                   order_type: str, price: Optional[float] = None,
                   dry_run: bool = False) -> Dict[str, Any]:
        """Place an order"""
        logger.info(f"Place order: account_id={account_id}, symbol={symbol}, qty={quantity}, type={order_type}, dry_run={dry_run}")

        self.order_counter += 1
        order_id = f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}{self.order_counter:04d}"

        order = MockOrder(
            order_id=order_id,
            account_id=account_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            order_type=order_type,
            status="ACCEPTED" if not dry_run else "PENDING",
            created_at=datetime.now().isoformat(),
            dry_run=dry_run
        )

        if account_id not in self.orders:
            self.orders[account_id] = []

        self.orders[account_id].append(order)

        return order.to_dict()

    def get_orders(self, account_id: str) -> List[Dict[str, Any]]:
        """Get account orders"""
        logger.info(f"Get orders: account_id={account_id}")

        if account_id not in self.orders:
            return []

        return [order.to_dict() for order in self.orders[account_id]]

    def cancel_order(self, account_id: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        logger.info(f"Cancel order: account_id={account_id}, order_id={order_id}")

        if account_id in self.orders:
            for order in self.orders[account_id]:
                if order.order_id == order_id:
                    order.status = "CANCELED"
                    return order.to_dict()

        return {"error": "Order not found"}


class MockSchwabRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock API"""

    # Class variable to store the mock API instance
    mock_api: Optional[MockSchwabAPI] = None

    def do_POST(self):
        """Handle POST requests"""
        path = self.path
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else ""

        logger.debug(f"POST {path}\n{body}")

        try:
            if path == "/v1/oauth/token":
                self._handle_oauth_token(body)
            elif "/accounts/" in path and "/orders" in path:
                self._handle_place_order(path, body)
            else:
                self._send_response(404, {"error": "Not found"})
        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}")
            self._send_response(500, {"error": str(e)})

    def do_GET(self):
        """Handle GET requests"""
        path = self.path

        # Parse query parameters
        parsed_url = urlparse(path)
        query_params = parse_qs(parsed_url.query)

        logger.debug(f"GET {path}")

        try:
            if "/marketdata/v1/quotes" in path:
                self._handle_get_quotes(parsed_url.path, query_params)
            elif "/marketdata/v1/chains" in path:
                self._handle_get_chains(query_params)
            elif "/marketdata/v1/pricehistory" in path:
                self._handle_get_pricehistory(query_params)
            elif "/accounts" in path and "/positions" in path:
                self._handle_get_positions(path)
            elif path == "/v1/accounts" or path.startswith("/trader/v1/accounts") and not "/positions" in path and not "/orders" in path:
                self._handle_get_accounts(path)
            else:
                self._send_response(404, {"error": "Not found"})
        except Exception as e:
            logger.error(f"Error handling GET {path}: {e}")
            self._send_response(500, {"error": str(e)})

    def _handle_oauth_token(self, body: str):
        """Handle OAuth token endpoint"""
        data = json.loads(body) if body else {}
        grant_type = data.get("grant_type")
        client_id = data.get("client_id", "test_client")

        if grant_type == "authorization_code":
            code = data.get("code", "test_code")
            code_verifier = data.get("code_verifier", "test_verifier")
            token_data = self.mock_api.exchange_token(client_id, code, code_verifier)
            self._send_response(200, token_data)
        elif grant_type == "refresh_token":
            refresh_token = data.get("refresh_token", "")
            token_data = self.mock_api.refresh_token(client_id, refresh_token)
            self._send_response(200, token_data)
        else:
            self._send_response(400, {"error": "invalid_request"})

    def _handle_get_quotes(self, path: str, query_params: Dict[str, List[str]]):
        """Handle GET quotes endpoint"""
        # Extract symbols from query params
        symbols_param = query_params.get("symbols", [None])[0]

        if not symbols_param:
            self._send_response(400, {"error": "Missing symbols parameter"})
            return

        symbols = symbols_param.split(",")

        if len(symbols) == 1:
            data = self.mock_api.get_quote(symbols[0])
        else:
            data = self.mock_api.get_quotes(symbols)

        self._send_response(200, data)

    def _handle_get_chains(self, query_params: Dict[str, List[str]]):
        """Handle GET options chains endpoint"""
        symbol = query_params.get("symbol", ["SPY"])[0]
        contract_type = query_params.get("contractType", ["ALL"])[0]

        data = self.mock_api.get_options_chain(symbol, contract_type)
        self._send_response(200, data)

    def _handle_get_pricehistory(self, query_params: Dict[str, List[str]]):
        """Handle GET price history endpoint"""
        symbol = query_params.get("symbol", ["SPY"])[0]
        period_type = query_params.get("periodType", ["month"])[0]
        period = int(query_params.get("period", ["1"])[0])

        data = self.mock_api.get_price_history(symbol, period_type, period)
        self._send_response(200, data)

    def _handle_get_accounts(self, path: str):
        """Handle GET accounts endpoint"""
        if "/accounts/" in path:
            # Get single account
            account_id = path.split("/")[-1]
            account = self.mock_api.get_account(account_id)
            if account:
                self._send_response(200, {"account": account})
            else:
                self._send_response(404, {"error": "Account not found"})
        else:
            # Get all accounts
            accounts = self.mock_api.get_accounts()
            self._send_response(200, {"accounts": accounts})

    def _handle_get_positions(self, path: str):
        """Handle GET positions endpoint"""
        # Extract account ID from path
        parts = path.split("/")
        account_id = None
        for i, part in enumerate(parts):
            if part == "accounts" and i + 1 < len(parts):
                account_id = parts[i + 1]
                break

        if not account_id:
            self._send_response(400, {"error": "Invalid path"})
            return

        positions = self.mock_api.get_positions(account_id)
        self._send_response(200, {"positions": positions})

    def _handle_place_order(self, path: str, body: str):
        """Handle POST order endpoint"""
        data = json.loads(body) if body else {}

        # Extract account ID from path
        parts = path.split("/")
        account_id = None
        for i, part in enumerate(parts):
            if part == "accounts" and i + 1 < len(parts):
                account_id = parts[i + 1]
                break

        if not account_id:
            self._send_response(400, {"error": "Invalid path"})
            return

        # Extract order details
        symbol = data.get("symbol", "SPY")
        quantity = data.get("quantity", 1)
        order_type = data.get("orderType", "MARKET")
        price = data.get("price")
        dry_run = data.get("dryRun", False)

        order = self.mock_api.place_order(account_id, symbol, quantity, order_type, price, dry_run)
        self._send_response(201, order)

    def _send_response(self, status_code: int, data: Any):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


class MockSchwabServer:
    """Mock Schwab API server manager"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        """Initialize mock server"""
        self.host = host
        self.port = port
        self.mock_api = MockSchwabAPI()
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start mock server"""
        MockSchwabRequestHandler.mock_api = self.mock_api

        self.server = HTTPServer((self.host, self.port), MockSchwabRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        logger.info(f"Mock Schwab API server started on {self.host}:{self.port}")
        time.sleep(0.5)  # Allow server to start

    def stop(self):
        """Stop mock server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()

        if self.thread:
            self.thread.join(timeout=5)

        logger.info("Mock Schwab API server stopped")

    def get_api(self) -> MockSchwabAPI:
        """Get mock API instance"""
        return self.mock_api

    def reset(self):
        """Reset mock API state"""
        self.mock_api = MockSchwabAPI()
        MockSchwabRequestHandler.mock_api = self.mock_api


if __name__ == "__main__":
    # Start mock server for testing
    server = MockSchwabServer()
    server.start()

    try:
        print(f"Mock Schwab API running on http://127.0.0.1:8765")
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()

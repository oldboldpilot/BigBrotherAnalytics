#!/usr/bin/env python3
"""
BigBrotherAnalytics - Test Token Sender

Sends test OAuth tokens to the TokenReceiver via Unix domain socket or TCP socket.
Useful for testing token refresh functionality without a real OAuth server.

Usage:
    # Send to Unix socket
    ./send_test_token.py "test_access_token_12345"

    # Send to TCP socket
    ./send_test_token.py --tcp "test_access_token_67890"

    # Continuous sending (every 30 seconds)
    ./send_test_token.py --continuous --interval 30

    # Send custom token from file
    ./send_test_token.py --file tokens.txt

Author: Olumuyiwa Oluwasanmi
Date: November 13, 2025
"""

import argparse
import socket
import time
import sys
from datetime import datetime


def send_token_unix(token: str, socket_path: str) -> bool:
    """Send token via Unix domain socket."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        sock.sendall(token.encode('utf-8'))
        sock.close()
        return True
    except FileNotFoundError:
        print(f"Error: Socket file not found: {socket_path}", file=sys.stderr)
        print("Is the TokenReceiver running?", file=sys.stderr)
        return False
    except ConnectionRefusedError:
        print(f"Error: Connection refused to {socket_path}", file=sys.stderr)
        print("Is the TokenReceiver listening?", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error sending token via Unix socket: {e}", file=sys.stderr)
        return False


def send_token_tcp(token: str, host: str, port: int) -> bool:
    """Send token via TCP socket."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.sendall(token.encode('utf-8'))
        sock.close()
        return True
    except ConnectionRefusedError:
        print(f"Error: Connection refused to {host}:{port}", file=sys.stderr)
        print("Is the TokenReceiver listening on TCP?", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error sending token via TCP: {e}", file=sys.stderr)
        return False


def generate_test_token() -> str:
    """Generate a test OAuth-like token."""
    timestamp = int(time.time())
    return f"test_access_token_{timestamp}"


def main():
    parser = argparse.ArgumentParser(
        description="Send test tokens to BigBrother TokenReceiver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "my_test_token"
  %(prog)s --tcp "my_test_token"
  %(prog)s --continuous --interval 30
  %(prog)s --file tokens.txt --tcp
        """
    )

    parser.add_argument(
        "token",
        nargs="?",
        help="Token string to send (auto-generated if not provided)"
    )

    parser.add_argument(
        "--tcp",
        action="store_true",
        help="Use TCP socket instead of Unix socket"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP host (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="TCP port (default: 9999)"
    )

    parser.add_argument(
        "--unix-socket",
        default="/tmp/bigbrother_token.sock",
        help="Unix socket path (default: /tmp/bigbrother_token.sock)"
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Send tokens continuously"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Interval between sends in seconds (default: 30)"
    )

    parser.add_argument(
        "--file",
        help="Read token from file (one token per line for continuous mode)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Get token(s) to send
    tokens = []
    if args.file:
        try:
            with open(args.file, 'r') as f:
                tokens = [line.strip() for line in f if line.strip()]
            if not tokens:
                print(f"Error: No tokens found in {args.file}", file=sys.stderr)
                return 1
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
    elif args.token:
        tokens = [args.token]
    else:
        # Auto-generate token
        tokens = [generate_test_token()]

    # Determine send function
    if args.tcp:
        send_func = lambda token: send_token_tcp(token, args.host, args.port)
        transport = f"TCP {args.host}:{args.port}"
    else:
        send_func = lambda token: send_token_unix(token, args.unix_socket)
        transport = f"Unix socket {args.unix_socket}"

    print(f"Token Sender - BigBrotherAnalytics")
    print(f"Transport: {transport}")
    print(f"Mode: {'Continuous' if args.continuous else 'Single'}")
    if args.continuous:
        print(f"Interval: {args.interval} seconds")
    print()

    # Send token(s)
    token_idx = 0
    success_count = 0
    failure_count = 0

    try:
        while True:
            # Get next token
            if args.file and args.continuous:
                token = tokens[token_idx % len(tokens)]
                token_idx += 1
            elif args.continuous:
                token = generate_test_token()
            else:
                token = tokens[0]

            # Display token info
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            token_preview = token[:50] + "..." if len(token) > 50 else token

            if args.verbose:
                print(f"[{timestamp}] Sending token: {token_preview}")
            else:
                print(f"[{timestamp}] Sending token ({len(token)} bytes)...", end=" ")

            # Send token
            if send_func(token):
                success_count += 1
                if args.verbose:
                    print(f"[{timestamp}] ✓ Token sent successfully")
                else:
                    print("✓")
            else:
                failure_count += 1
                if not args.verbose:
                    print("✗")

            # Break if single send mode
            if not args.continuous:
                break

            # Wait before next send
            if args.verbose:
                print(f"Waiting {args.interval} seconds...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    # Print statistics
    print()
    print("=" * 60)
    print("Statistics:")
    print(f"  Tokens sent successfully: {success_count}")
    print(f"  Tokens failed: {failure_count}")
    print(f"  Success rate: {100 * success_count / (success_count + failure_count):.1f}%" if (success_count + failure_count) > 0 else "N/A")
    print("=" * 60)

    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

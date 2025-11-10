"""
BigBrotherAnalytics: Alert Email Templates
HTML email templates for various alert types

Author: Olumuyiwa Oluwasanmi
Date: 2025-11-10
Phase 4, Week 3: Custom Alerts System
"""

from datetime import datetime
from typing import Dict, Any


def get_severity_color(severity: str) -> str:
    """Get color code for severity level."""
    colors = {
        'INFO': '#2196F3',      # Blue
        'WARNING': '#FF9800',   # Orange
        'ERROR': '#F44336',     # Red
        'CRITICAL': '#D32F2F'   # Dark Red
    }
    return colors.get(severity, '#757575')


def get_alert_icon(alert_type: str) -> str:
    """Get emoji icon for alert type."""
    icons = {
        'trading': '📈',
        'data': '📊',
        'system': '⚠️',
        'performance': '⚡'
    }
    return icons.get(alert_type, '🔔')


def render_base_template(title: str, severity: str, content: str) -> str:
    """
    Render base HTML email template.

    Args:
        title: Email title
        severity: Alert severity
        content: HTML content to insert

    Returns:
        Complete HTML email
    """
    color = get_severity_color(severity)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 600px;
            margin: 20px auto;
            background-color: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .header {{
            background-color: {color};
            color: #ffffff;
            padding: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }}
        .content {{
            padding: 30px;
        }}
        .alert-badge {{
            display: inline-block;
            background-color: {color};
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 15px;
        }}
        .alert-message {{
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 20px;
            color: #1a1a1a;
        }}
        .details {{
            background-color: #f8f9fa;
            border-left: 4px solid {color};
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .details-row {{
            display: flex;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .details-row:last-child {{
            border-bottom: none;
        }}
        .details-label {{
            font-weight: 600;
            width: 120px;
            color: #666;
        }}
        .details-value {{
            flex: 1;
            color: #1a1a1a;
        }}
        .context {{
            background-color: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
        }}
        .footer {{
            background-color: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 12px;
            border-top: 1px solid #e0e0e0;
        }}
        .button {{
            display: inline-block;
            background-color: {color};
            color: #ffffff;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 4px;
            font-weight: 600;
            margin: 20px 0;
        }}
        .timestamp {{
            color: #666;
            font-size: 14px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
        </div>
        <div class="content">
            <div class="alert-badge">{severity}</div>
            {content}
            <div class="timestamp">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}
            </div>
        </div>
        <div class="footer">
            <p>BigBrotherAnalytics Alert System</p>
            <p>AI-Powered Algorithmic Trading Platform</p>
        </div>
    </div>
</body>
</html>
"""
    return html


def render_trading_alert(alert_data: Dict[str, Any]) -> str:
    """Render trading alert email."""
    message = alert_data.get('message', '')
    subtype = alert_data.get('alert_subtype', '')
    context = alert_data.get('context', {})
    severity = alert_data.get('severity', 'INFO')

    icon = get_alert_icon('trading')

    content = f"""
        <div class="alert-message">{icon} {message}</div>
        <div class="details">
            <div class="details-row">
                <div class="details-label">Alert Type:</div>
                <div class="details-value">Trading - {subtype.replace('_', ' ').title()}</div>
            </div>
    """

    # Add context details
    if 'symbol' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Symbol:</div>
                <div class="details-value">{context['symbol']}</div>
            </div>
        """

    if 'pnl' in context:
        pnl = float(context['pnl'])
        pnl_color = 'green' if pnl >= 0 else 'red'
        content += f"""
            <div class="details-row">
                <div class="details-label">P&L:</div>
                <div class="details-value" style="color: {pnl_color}; font-weight: 600;">
                    ${pnl:,.2f}
                </div>
            </div>
        """

    if 'loss' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Loss Amount:</div>
                <div class="details-value" style="color: red; font-weight: 600;">
                    ${float(context['loss']):,.2f}
                </div>
            </div>
        """

    if 'size' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Position Size:</div>
                <div class="details-value">${float(context['size']):,.2f}</div>
            </div>
        """

    if 'action' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Action:</div>
                <div class="details-value" style="font-weight: 600;">{context['action']}</div>
            </div>
        """

    if 'reason' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Reason:</div>
                <div class="details-value">{context['reason']}</div>
            </div>
        """

    content += """
        </div>
    """

    # Add dashboard link
    content += """
        <a href="http://localhost:8501" class="button">View Dashboard</a>
    """

    title = f"Trading Alert - {severity}"
    return render_base_template(title, severity, content)


def render_data_alert(alert_data: Dict[str, Any]) -> str:
    """Render data alert email."""
    message = alert_data.get('message', '')
    subtype = alert_data.get('alert_subtype', '')
    context = alert_data.get('context', {})
    severity = alert_data.get('severity', 'INFO')

    icon = get_alert_icon('data')

    content = f"""
        <div class="alert-message">{icon} {message}</div>
        <div class="details">
            <div class="details-row">
                <div class="details-label">Alert Type:</div>
                <div class="details-value">Data - {subtype.replace('_', ' ').title()}</div>
            </div>
    """

    # Add context details
    if 'records' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Records:</div>
                <div class="details-value">{context['records']:,}</div>
            </div>
        """

    if 'count' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Count:</div>
                <div class="details-value">{context['count']:,}</div>
            </div>
        """

    if 'spike_percent' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Spike:</div>
                <div class="details-value" style="color: red; font-weight: 600;">
                    {float(context['spike_percent']):.1f}% above average
                </div>
            </div>
        """

    if 'days_old' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Data Age:</div>
                <div class="details-value" style="color: orange; font-weight: 600;">
                    {context['days_old']} days old
                </div>
            </div>
        """

    if 'error' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Error:</div>
                <div class="details-value" style="color: red;">{context['error']}</div>
            </div>
        """

    content += """
        </div>
    """

    title = f"Data Alert - {severity}"
    return render_base_template(title, severity, content)


def render_system_alert(alert_data: Dict[str, Any]) -> str:
    """Render system health alert email."""
    message = alert_data.get('message', '')
    subtype = alert_data.get('alert_subtype', '')
    context = alert_data.get('context', {})
    severity = alert_data.get('severity', 'INFO')

    icon = get_alert_icon('system')

    content = f"""
        <div class="alert-message">{icon} {message}</div>
        <div class="details">
            <div class="details-row">
                <div class="details-label">Alert Type:</div>
                <div class="details-value">System - {subtype.replace('_', ' ').title()}</div>
            </div>
    """

    if 'reason' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Reason:</div>
                <div class="details-value">{context['reason']}</div>
            </div>
        """

    if 'error' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Error:</div>
                <div class="details-value" style="font-family: monospace; color: red;">
                    {context['error']}
                </div>
            </div>
        """

    if 'count' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Error Count:</div>
                <div class="details-value" style="font-weight: 600;">{context['count']}</div>
            </div>
        """

    content += """
        </div>
    """

    # Add action buttons for critical alerts
    if severity == 'CRITICAL':
        content += """
            <div style="margin-top: 20px; padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;">
                <strong>⚠️ Immediate action required!</strong>
                <p style="margin: 10px 0 0 0;">This is a critical system alert. Please investigate immediately.</p>
            </div>
        """

    title = f"System Alert - {severity}"
    return render_base_template(title, severity, content)


def render_performance_alert(alert_data: Dict[str, Any]) -> str:
    """Render performance alert email."""
    message = alert_data.get('message', '')
    subtype = alert_data.get('alert_subtype', '')
    context = alert_data.get('context', {})
    severity = alert_data.get('severity', 'INFO')

    icon = get_alert_icon('performance')

    content = f"""
        <div class="alert-message">{icon} {message}</div>
        <div class="details">
            <div class="details-row">
                <div class="details-label">Alert Type:</div>
                <div class="details-value">Performance - {subtype.replace('_', ' ').title()}</div>
            </div>
    """

    # Add performance metrics
    if 'duration_ms' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Duration:</div>
                <div class="details-value" style="font-weight: 600;">
                    {float(context['duration_ms']):.2f} ms
                </div>
            </div>
        """

    if 'duration_us' in context:
        content += f"""
            <div class="details-row">
                <div class="details-label">Duration:</div>
                <div class="details-value" style="font-weight: 600;">
                    {float(context['duration_us']):.2f} μs
                </div>
            </div>
        """

    if 'usage_percent' in context:
        usage = float(context['usage_percent'])
        color = 'red' if usage > 90 else ('orange' if usage > 80 else 'green')
        content += f"""
            <div class="details-row">
                <div class="details-label">Usage:</div>
                <div class="details-value" style="color: {color}; font-weight: 600;">
                    {usage:.1f}%
                </div>
            </div>
        """

    if 'free_percent' in context:
        free = float(context['free_percent'])
        color = 'red' if free < 10 else ('orange' if free < 20 else 'green')
        content += f"""
            <div class="details-row">
                <div class="details-label">Free Space:</div>
                <div class="details-value" style="color: {color}; font-weight: 600;">
                    {free:.1f}%
                </div>
            </div>
        """

    if 'current_ms' in context and 'baseline_ms' in context:
        current = float(context['current_ms'])
        baseline = float(context['baseline_ms'])
        multiplier = current / baseline
        content += f"""
            <div class="details-row">
                <div class="details-label">Current:</div>
                <div class="details-value">{current:.2f} ms</div>
            </div>
            <div class="details-row">
                <div class="details-label">Baseline:</div>
                <div class="details-value">{baseline:.2f} ms</div>
            </div>
            <div class="details-row">
                <div class="details-label">Multiplier:</div>
                <div class="details-value" style="color: red; font-weight: 600;">
                    {multiplier:.1f}x slower
                </div>
            </div>
        """

    content += """
        </div>
    """

    title = f"Performance Alert - {severity}"
    return render_base_template(title, severity, content)


def render_alert_email(alert_data: Dict[str, Any]) -> str:
    """
    Render HTML email for any alert type.

    Args:
        alert_data: Alert data dictionary

    Returns:
        HTML email string
    """
    alert_type = alert_data.get('alert_type', 'unknown')

    if alert_type == 'trading':
        return render_trading_alert(alert_data)
    elif alert_type == 'data':
        return render_data_alert(alert_data)
    elif alert_type == 'system':
        return render_system_alert(alert_data)
    elif alert_type == 'performance':
        return render_performance_alert(alert_data)
    else:
        # Generic template
        message = alert_data.get('message', 'Alert')
        severity = alert_data.get('severity', 'INFO')
        content = f'<div class="alert-message">{message}</div>'
        return render_base_template(f"Alert - {severity}", severity, content)

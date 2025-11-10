#!/bin/bash
###############################################################################
# BigBrotherAnalytics: Cron Job Setup Script
#
# Sets up automated daily data updates and signal recalculation
#
# Schedule:
# - Daily data update: 6:00 AM ET (after market open)
# - Signal recalculation: 6:30 AM ET (after data update)
# - Daily summary notification: 7:00 AM ET
#
# Author: Olumuyiwa Oluwasanmi
# Date: 2025-11-10
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}BigBrotherAnalytics - Cron Job Setup${NC}"
echo -e "${BLUE}========================================================================${NC}\n"

# Check if running on WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}Warning: Detected WSL environment${NC}"
    echo -e "${YELLOW}Cron may not work as expected on WSL. Consider using Windows Task Scheduler.${NC}"
    echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Setup cancelled${NC}"
        exit 1
    fi
fi

# Paths
DAILY_UPDATE_SCRIPT="${SCRIPT_DIR}/daily_update.py"
RECALC_SIGNALS_SCRIPT="${SCRIPT_DIR}/recalculate_signals.py"
NOTIFY_SCRIPT="${SCRIPT_DIR}/notify.py"
PYTHON_BIN="$(which python3)"
CRON_FILE="${SCRIPT_DIR}/bigbrother_cron.txt"
LOG_DIR="${PROJECT_ROOT}/logs/automated_updates"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Verify scripts exist
echo -e "${BLUE}Verifying scripts...${NC}"
if [[ ! -f "$DAILY_UPDATE_SCRIPT" ]]; then
    echo -e "${RED}Error: daily_update.py not found at $DAILY_UPDATE_SCRIPT${NC}"
    exit 1
fi

if [[ ! -f "$RECALC_SIGNALS_SCRIPT" ]]; then
    echo -e "${RED}Error: recalculate_signals.py not found at $RECALC_SIGNALS_SCRIPT${NC}"
    exit 1
fi

if [[ ! -f "$NOTIFY_SCRIPT" ]]; then
    echo -e "${RED}Error: notify.py not found at $NOTIFY_SCRIPT${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All scripts found${NC}\n"

# Make scripts executable
echo -e "${BLUE}Making scripts executable...${NC}"
chmod +x "$DAILY_UPDATE_SCRIPT"
chmod +x "$RECALC_SIGNALS_SCRIPT"
chmod +x "$NOTIFY_SCRIPT"
echo -e "${GREEN}✓ Scripts are executable${NC}\n"

# Create cron job entries
echo -e "${BLUE}Creating cron job entries...${NC}"

cat > "$CRON_FILE" << EOF
# BigBrotherAnalytics Automated Updates
# Generated on $(date)
#
# Schedule (all times in Eastern Time):
# - 6:00 AM ET: Daily data update (BLS employment & jobless claims)
# - 6:30 AM ET: Signal recalculation
# - 7:00 AM ET: Daily summary notification
#
# Note: Cron times are in local system time. Adjust if not in ET.

# Environment variables (add your API keys here)
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# BLS_API_KEY=your_bls_api_key_here
# SMTP_USER=your_email@example.com
# SMTP_PASSWORD=your_email_password_here
# EMAIL_TO=recipient@example.com
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Daily data update at 6:00 AM ET
0 6 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${DAILY_UPDATE_SCRIPT} >> ${LOG_DIR}/cron_daily_update.log 2>&1

# Signal recalculation at 6:30 AM ET (after data update)
30 6 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${RECALC_SIGNALS_SCRIPT} >> ${LOG_DIR}/cron_signal_recalc.log 2>&1

# Daily summary notification at 7:00 AM ET
0 7 * * * cd ${PROJECT_ROOT} && ${PYTHON_BIN} ${NOTIFY_SCRIPT} --slack --type summary >> ${LOG_DIR}/cron_notify.log 2>&1

# Weekly cleanup of old logs (every Sunday at 2:00 AM)
0 2 * * 0 find ${LOG_DIR} -name "*.log" -mtime +30 -delete

EOF

echo -e "${GREEN}✓ Cron job file created: $CRON_FILE${NC}\n"

# Display the cron jobs
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}Cron Job Entries:${NC}"
echo -e "${BLUE}========================================================================${NC}"
cat "$CRON_FILE"
echo -e "${BLUE}========================================================================${NC}\n"

# Ask if user wants to install the cron jobs
echo -e "${YELLOW}Do you want to install these cron jobs now? (y/n)${NC}"
read -r install_response

if [[ "$install_response" =~ ^[Yy]$ ]]; then
    # Backup existing crontab
    echo -e "${BLUE}Backing up existing crontab...${NC}"
    crontab -l > "${SCRIPT_DIR}/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || true
    echo -e "${GREEN}✓ Backup created${NC}\n"

    # Install cron jobs (append to existing crontab)
    echo -e "${BLUE}Installing cron jobs...${NC}"
    (crontab -l 2>/dev/null; echo ""; cat "$CRON_FILE") | crontab -
    echo -e "${GREEN}✓ Cron jobs installed${NC}\n"

    # Verify installation
    echo -e "${BLUE}Current crontab (BigBrotherAnalytics entries):${NC}"
    crontab -l | grep -A 20 "BigBrotherAnalytics" || echo -e "${YELLOW}Warning: Could not verify installation${NC}"
    echo ""

    echo -e "${GREEN}========================================================================${NC}"
    echo -e "${GREEN}Cron jobs successfully installed!${NC}"
    echo -e "${GREEN}========================================================================${NC}\n"
else
    echo -e "${YELLOW}Cron jobs NOT installed.${NC}"
    echo -e "${YELLOW}To install manually, run: crontab $CRON_FILE${NC}\n"
fi

# Configuration instructions
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}IMPORTANT: Configuration Required${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo -e ""
echo -e "${YELLOW}1. BLS API Key (Optional but recommended):${NC}"
echo -e "   Export in your shell profile or add to cron:"
echo -e "   ${GREEN}export BLS_API_KEY='your_api_key_here'${NC}"
echo -e "   Get free API key at: https://data.bls.gov/registrationEngine/"
echo -e ""
echo -e "${YELLOW}2. Email Notifications (Optional):${NC}"
echo -e "   Set the following environment variables:"
echo -e "   ${GREEN}export SMTP_HOST='smtp.gmail.com'${NC}"
echo -e "   ${GREEN}export SMTP_PORT='587'${NC}"
echo -e "   ${GREEN}export SMTP_USER='your_email@gmail.com'${NC}"
echo -e "   ${GREEN}export SMTP_PASSWORD='your_app_password'${NC}"
echo -e "   ${GREEN}export EMAIL_FROM='your_email@gmail.com'${NC}"
echo -e "   ${GREEN}export EMAIL_TO='recipient@example.com'${NC}"
echo -e ""
echo -e "${YELLOW}3. Slack Notifications (Optional):${NC}"
echo -e "   Set the Slack webhook URL:"
echo -e "   ${GREEN}export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/YOUR/WEBHOOK/URL'${NC}"
echo -e "   Create webhook at: https://api.slack.com/messaging/webhooks"
echo -e ""
echo -e "${YELLOW}4. Edit Cron Jobs:${NC}"
echo -e "   To edit cron jobs: ${GREEN}crontab -e${NC}"
echo -e "   To remove all cron jobs: ${GREEN}crontab -r${NC}"
echo -e "   To view current cron jobs: ${GREEN}crontab -l${NC}"
echo -e ""
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}Testing${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo -e ""
echo -e "Test the scripts manually before relying on cron:"
echo -e ""
echo -e "${GREEN}1. Test daily update (test mode):${NC}"
echo -e "   cd ${PROJECT_ROOT}"
echo -e "   ${PYTHON_BIN} ${DAILY_UPDATE_SCRIPT} --test"
echo -e ""
echo -e "${GREEN}2. Test signal recalculation (test mode):${NC}"
echo -e "   ${PYTHON_BIN} ${RECALC_SIGNALS_SCRIPT} --test"
echo -e ""
echo -e "${GREEN}3. Test notifications (test mode):${NC}"
echo -e "   ${PYTHON_BIN} ${NOTIFY_SCRIPT} --slack --test --type summary"
echo -e ""
echo -e "${BLUE}========================================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}========================================================================${NC}\n"

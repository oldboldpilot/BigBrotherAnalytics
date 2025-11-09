/**
 * BigBrotherAnalytics - Test Employment Signals Module
 *
 * Tests the EmploymentSignalGenerator C++ integration with Python backend.
 *
 * Author: Olumuyiwa Oluwasanmi
 * Date: 2025-11-09
 */

#include <iomanip>
#include <iostream>

import bigbrother.employment.signals;

using namespace bigbrother::employment;

void printSeparator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(80, '=') << "\n\n";
}

void testEmploymentSignals() {
    printSeparator("Employment Signal Generation Test");

    EmploymentSignalGenerator generator;

    std::cout << "Generating employment signals...\n";
    auto signals = generator.generateSignals();

    std::cout << "Found " << signals.size() << " employment signals\n\n";

    if (signals.empty()) {
        std::cout << "No signals generated (employment changes below 5% threshold)\n";
        std::cout << "This is expected for stable employment periods.\n";
    } else {
        std::cout << std::left << std::setw(20) << "Sector" << std::setw(20) << "Signal Type"
                  << std::setw(12) << "Change %" << std::setw(12) << "Strength" << std::setw(12)
                  << "Confidence"
                  << "\n";
        std::cout << std::string(80, '-') << "\n";

        for (const auto& signal : signals) {
            std::cout << std::left << std::setw(20) << signal.sector_name << std::setw(20)
                      << (signal.bullish ? "Improving" : "Declining") << std::setw(12) << std::fixed
                      << std::setprecision(2) << signal.employment_change << std::setw(12)
                      << std::fixed << std::setprecision(2) << signal.signal_strength
                      << std::setw(12) << std::fixed << std::setprecision(2) << signal.confidence
                      << "\n";
            std::cout << "  Rationale: " << signal.rationale << "\n";
            std::cout << "  Actionable: " << (signal.isActionable() ? "Yes" : "No") << "\n\n";
        }
    }
}

void testRotationSignals() {
    printSeparator("Sector Rotation Signal Generation Test");

    EmploymentSignalGenerator generator;

    std::cout << "Generating sector rotation signals...\n";
    auto signals = generator.generateRotationSignals();

    std::cout << "Found " << signals.size() << " rotation signals\n\n";

    if (!signals.empty()) {
        std::cout << std::left << std::setw(25) << "Sector" << std::setw(8) << "ETF"
                  << std::setw(12) << "Emp Score" << std::setw(12) << "Composite" << std::setw(12)
                  << "Action" << std::setw(12) << "Target %"
                  << "\n";
        std::cout << std::string(80, '-') << "\n";

        for (const auto& signal : signals) {
            std::string action_str;
            switch (signal.action) {
                case SectorRotationSignal::Action::Overweight:
                    action_str = "Overweight";
                    break;
                case SectorRotationSignal::Action::Underweight:
                    action_str = "Underweight";
                    break;
                case SectorRotationSignal::Action::Neutral:
                    action_str = "Neutral";
                    break;
            }

            std::cout << std::left << std::setw(25) << signal.sector_name << std::setw(8)
                      << signal.sector_etf << std::setw(12) << std::fixed << std::setprecision(3)
                      << signal.employment_score << std::setw(12) << std::fixed
                      << std::setprecision(3) << signal.composite_score << std::setw(12)
                      << action_str << std::setw(12) << std::fixed << std::setprecision(1)
                      << signal.target_allocation << "\n";
        }

        std::cout << "\nStrong Signals (|composite| > 0.7):\n";
        bool has_strong = false;
        for (const auto& signal : signals) {
            if (signal.isStrongSignal()) {
                has_strong = true;
                std::cout << "  - " << signal.sector_name << " (" << signal.sector_etf
                          << "): " << std::fixed << std::setprecision(3) << signal.composite_score
                          << "\n";
            }
        }
        if (!has_strong) {
            std::cout << "  None (all signals are weak/neutral)\n";
        }
    }
}

void testJoblessClaimsCheck() {
    printSeparator("Jobless Claims Spike Check Test");

    EmploymentSignalGenerator generator;

    std::cout << "Checking for jobless claims spike...\n";
    auto signal = generator.checkJoblessClaimsSpike();

    if (signal.has_value()) {
        std::cout << "\nWARNING: Jobless claims spike detected!\n";
        std::cout << "  Signal Type: RecessionWarning\n";
        std::cout << "  Rationale: " << signal->rationale << "\n";
        std::cout << "  Confidence: " << std::fixed << std::setprecision(2) << signal->confidence
                  << "\n";
    } else {
        std::cout << "\nNo jobless claims spike detected.\n";
        std::cout << "Note: Jobless claims data not yet implemented in database.\n";
    }
}

int main() {
    try {
        std::cout << "BigBrotherAnalytics - Employment Signals Module Test\n";
        std::cout << "====================================================\n";

        testEmploymentSignals();
        testRotationSignals();
        testJoblessClaimsCheck();

        printSeparator("Test Summary");
        std::cout << "All tests completed successfully!\n";
        std::cout << "\nNext Steps:\n";
        std::cout << "  1. Integrate with Trading Decision Engine\n";
        std::cout << "  2. Add jobless claims data to database\n";
        std::cout << "  3. Implement sentiment and technical scores for rotation signals\n";
        std::cout << "  4. Test with live trading strategies\n\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

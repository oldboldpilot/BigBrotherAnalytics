/**
 * BigBrother Analytics - Utils Module
 *
 * C++23 module for utility functions and classes.
 * Using modules significantly speeds up compilation by:
 * - Eliminating redundant header parsing
 * - Providing better encapsulation
 * - Enabling better optimization opportunities
 *
 * Usage:
 *   import bigbrother.utils;
 *
 * Instead of:
 *   #include "utils/logger.hpp"
 *   #include "utils/config.hpp"
 *   // etc.
 */

export module bigbrother.utils;

// Export all public interfaces from utils
export import bigbrother.utils.logger;
export import bigbrother.utils.config;
export import bigbrother.utils.database;
export import bigbrother.utils.timer;
export import bigbrother.utils.types;
export import bigbrother.utils.math;

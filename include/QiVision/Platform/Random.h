#pragma once

/**
 * @file Random.h
 * @brief Random number generation utilities
 *
 * Provides thread-safe random number generation for:
 * - RANSAC sampling
 * - Monte Carlo methods
 * - Noise generation
 * - Random initialization
 */

#include <cstdint>
#include <cstddef>
#include <vector>
#include <random>
#include <algorithm>
#include <type_traits>

namespace Qi::Vision::Platform {

/**
 * @brief Thread-safe random number generator
 *
 * Uses MT19937-64 for high quality randomness.
 * Each thread has its own generator instance.
 */
class Random {
public:
    /**
     * @brief Get thread-local random instance
     * @return Reference to thread-local Random instance
     */
    static Random& Instance();

    /**
     * @brief Set global seed for reproducibility
     * @param seed Seed value
     *
     * This reseeds the current thread's generator.
     * For multi-threaded reproducibility, call from each thread.
     */
    void SetSeed(uint64_t seed);

    /**
     * @brief Get current seed (for debugging)
     */
    uint64_t GetSeed() const { return seed_; }

    // =========================================================================
    // Integer Generation
    // =========================================================================

    /**
     * @brief Generate random 32-bit unsigned integer
     */
    uint32_t Uint32();

    /**
     * @brief Generate random 64-bit unsigned integer
     */
    uint64_t Uint64();

    /**
     * @brief Generate random integer in range [min, max] (inclusive)
     */
    int32_t Int(int32_t min, int32_t max);

    /**
     * @brief Generate random integer in range [0, max) (exclusive max)
     */
    size_t Index(size_t max);

    // =========================================================================
    // Floating Point Generation
    // =========================================================================

    /**
     * @brief Generate random float in [0, 1)
     */
    float Float();

    /**
     * @brief Generate random double in [0, 1)
     */
    double Double();

    /**
     * @brief Generate random float in [min, max)
     */
    float Float(float min, float max);

    /**
     * @brief Generate random double in [min, max)
     */
    double Double(double min, double max);

    /**
     * @brief Generate random number from standard normal distribution N(0,1)
     */
    double Gaussian();

    /**
     * @brief Generate random number from normal distribution N(mean, stddev)
     */
    double Gaussian(double mean, double stddev);

    // =========================================================================
    // Boolean Generation
    // =========================================================================

    /**
     * @brief Generate random boolean with 50% probability
     */
    bool Bool();

    /**
     * @brief Generate random boolean with given probability of true
     * @param probabilityTrue Probability in [0, 1]
     */
    bool Bool(double probabilityTrue);

    // =========================================================================
    // Sampling Functions (for RANSAC)
    // =========================================================================

    /**
     * @brief Sample k unique indices from range [0, n)
     * @param n Total number of items
     * @param k Number of items to sample (must be <= n)
     * @return Vector of k unique indices
     *
     * Uses reservoir sampling for efficiency when k << n.
     */
    std::vector<size_t> SampleIndices(size_t n, size_t k);

    /**
     * @brief Sample k unique items from a vector
     * @param items Source items
     * @param k Number of items to sample
     * @return Vector of k sampled items (copies)
     */
    template<typename T>
    std::vector<T> Sample(const std::vector<T>& items, size_t k);

    /**
     * @brief Shuffle a vector in place
     * @param items Vector to shuffle
     */
    template<typename T>
    void Shuffle(std::vector<T>& items);

    /**
     * @brief Get underlying generator (for use with STL distributions)
     */
    std::mt19937_64& Generator() { return gen_; }

private:
    Random();
    Random(const Random&) = delete;
    Random& operator=(const Random&) = delete;

    std::mt19937_64 gen_;
    uint64_t seed_;

    // Cached distributions for common use cases
    std::uniform_real_distribution<double> unitDist_{0.0, 1.0};
    std::normal_distribution<double> normalDist_{0.0, 1.0};
};

// =========================================================================
// Template Implementations
// =========================================================================

template<typename T>
std::vector<T> Random::Sample(const std::vector<T>& items, size_t k) {
    if (k >= items.size()) {
        return items; // Return all items
    }

    auto indices = SampleIndices(items.size(), k);
    std::vector<T> result;
    result.reserve(k);
    for (size_t idx : indices) {
        result.push_back(items[idx]);
    }
    return result;
}

template<typename T>
void Random::Shuffle(std::vector<T>& items) {
    std::shuffle(items.begin(), items.end(), gen_);
}

// =========================================================================
// Convenience Free Functions
// =========================================================================

/**
 * @brief Generate random integer in [min, max]
 */
inline int32_t RandomInt(int32_t min, int32_t max) {
    return Random::Instance().Int(min, max);
}

/**
 * @brief Generate random double in [min, max)
 */
inline double RandomDouble(double min, double max) {
    return Random::Instance().Double(min, max);
}

/**
 * @brief Generate random double in [0, 1)
 */
inline double RandomDouble() {
    return Random::Instance().Double();
}

/**
 * @brief Generate Gaussian random number N(mean, stddev)
 */
inline double RandomGaussian(double mean = 0.0, double stddev = 1.0) {
    return Random::Instance().Gaussian(mean, stddev);
}

/**
 * @brief Sample k unique indices from [0, n)
 */
inline std::vector<size_t> RandomSample(size_t n, size_t k) {
    return Random::Instance().SampleIndices(n, k);
}

/**
 * @brief Set random seed for reproducibility
 */
inline void SetRandomSeed(uint64_t seed) {
    Random::Instance().SetSeed(seed);
}

} // namespace Qi::Vision::Platform

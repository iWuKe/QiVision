/**
 * @file test_random.cpp
 * @brief Unit tests for Platform/Random.h
 */

#include <QiVision/Platform/Random.h>
#include <gtest/gtest.h>

#include <set>
#include <cmath>
#include <thread>
#include <atomic>

using namespace Qi::Vision::Platform;

class RandomTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set seed for reproducible tests
        Random::Instance().SetSeed(12345);
    }
};

// ============================================================================
// Basic Integer Tests
// ============================================================================

TEST_F(RandomTest, Uint32GeneratesValues) {
    uint32_t v1 = Random::Instance().Uint32();
    uint32_t v2 = Random::Instance().Uint32();
    // Very unlikely to be the same with a good RNG
    // (but possible, so this test is probabilistic)
    bool different = false;
    for (int i = 0; i < 10; ++i) {
        if (Random::Instance().Uint32() != v1) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

TEST_F(RandomTest, Uint64GeneratesValues) {
    uint64_t v1 = Random::Instance().Uint64();
    uint64_t v2 = Random::Instance().Uint64();
    EXPECT_NE(v1, v2); // Extremely unlikely to be equal
}

TEST_F(RandomTest, IntInRange) {
    for (int i = 0; i < 1000; ++i) {
        int32_t v = Random::Instance().Int(10, 20);
        EXPECT_GE(v, 10);
        EXPECT_LE(v, 20);
    }
}

TEST_F(RandomTest, IntSwapsMinMax) {
    // Should handle min > max
    for (int i = 0; i < 100; ++i) {
        int32_t v = Random::Instance().Int(20, 10);
        EXPECT_GE(v, 10);
        EXPECT_LE(v, 20);
    }
}

TEST_F(RandomTest, IntSingleValue) {
    // When min == max, should always return that value
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(Random::Instance().Int(42, 42), 42);
    }
}

TEST_F(RandomTest, IndexInRange) {
    for (int i = 0; i < 1000; ++i) {
        size_t v = Random::Instance().Index(10);
        EXPECT_LT(v, 10u);
    }
}

TEST_F(RandomTest, IndexZeroMax) {
    // Edge case: max = 0
    EXPECT_EQ(Random::Instance().Index(0), 0u);
}

// ============================================================================
// Floating Point Tests
// ============================================================================

TEST_F(RandomTest, FloatInUnitRange) {
    for (int i = 0; i < 1000; ++i) {
        float v = Random::Instance().Float();
        EXPECT_GE(v, 0.0f);
        EXPECT_LT(v, 1.0f);
    }
}

TEST_F(RandomTest, DoubleInUnitRange) {
    for (int i = 0; i < 1000; ++i) {
        double v = Random::Instance().Double();
        EXPECT_GE(v, 0.0);
        EXPECT_LT(v, 1.0);
    }
}

TEST_F(RandomTest, FloatInCustomRange) {
    for (int i = 0; i < 1000; ++i) {
        float v = Random::Instance().Float(5.0f, 10.0f);
        EXPECT_GE(v, 5.0f);
        EXPECT_LT(v, 10.0f);
    }
}

TEST_F(RandomTest, DoubleInCustomRange) {
    for (int i = 0; i < 1000; ++i) {
        double v = Random::Instance().Double(-100.0, 100.0);
        EXPECT_GE(v, -100.0);
        EXPECT_LT(v, 100.0);
    }
}

TEST_F(RandomTest, FloatSwapsMinMax) {
    for (int i = 0; i < 100; ++i) {
        float v = Random::Instance().Float(10.0f, 5.0f);
        EXPECT_GE(v, 5.0f);
        EXPECT_LT(v, 10.0f);
    }
}

// ============================================================================
// Gaussian Distribution Tests
// ============================================================================

TEST_F(RandomTest, GaussianStandardDistribution) {
    // Generate many samples and check mean/stddev are approximately correct
    const int n = 10000;
    double sum = 0.0;
    double sumSq = 0.0;

    for (int i = 0; i < n; ++i) {
        double v = Random::Instance().Gaussian();
        sum += v;
        sumSq += v * v;
    }

    double mean = sum / n;
    double variance = (sumSq / n) - (mean * mean);
    double stddev = std::sqrt(variance);

    // With 10000 samples, mean should be close to 0 and stddev close to 1
    EXPECT_NEAR(mean, 0.0, 0.05);     // ±0.05 tolerance
    EXPECT_NEAR(stddev, 1.0, 0.05);   // ±0.05 tolerance
}

TEST_F(RandomTest, GaussianCustomDistribution) {
    const int n = 10000;
    const double targetMean = 50.0;
    const double targetStddev = 10.0;
    double sum = 0.0;
    double sumSq = 0.0;

    for (int i = 0; i < n; ++i) {
        double v = Random::Instance().Gaussian(targetMean, targetStddev);
        sum += v;
        sumSq += v * v;
    }

    double mean = sum / n;
    double variance = (sumSq / n) - (mean * mean);
    double stddev = std::sqrt(variance);

    EXPECT_NEAR(mean, targetMean, 0.5);       // ±0.5 tolerance
    EXPECT_NEAR(stddev, targetStddev, 0.5);   // ±0.5 tolerance
}

// ============================================================================
// Boolean Tests
// ============================================================================

TEST_F(RandomTest, BoolGeneratesBoth) {
    bool foundTrue = false;
    bool foundFalse = false;

    for (int i = 0; i < 100; ++i) {
        if (Random::Instance().Bool()) {
            foundTrue = true;
        } else {
            foundFalse = true;
        }
        if (foundTrue && foundFalse) break;
    }

    EXPECT_TRUE(foundTrue);
    EXPECT_TRUE(foundFalse);
}

TEST_F(RandomTest, BoolWithProbability) {
    const int n = 10000;
    int trueCount = 0;

    for (int i = 0; i < n; ++i) {
        if (Random::Instance().Bool(0.7)) {
            ++trueCount;
        }
    }

    double ratio = static_cast<double>(trueCount) / n;
    EXPECT_NEAR(ratio, 0.7, 0.03);  // ±3% tolerance
}

TEST_F(RandomTest, BoolEdgeProbabilities) {
    // 0 probability should always return false
    for (int i = 0; i < 100; ++i) {
        EXPECT_FALSE(Random::Instance().Bool(0.0));
    }

    // 1 probability should always return true
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(Random::Instance().Bool(1.0));
    }
}

// ============================================================================
// Sampling Tests (Critical for RANSAC)
// ============================================================================

TEST_F(RandomTest, SampleIndicesCorrectSize) {
    auto indices = Random::Instance().SampleIndices(100, 10);
    EXPECT_EQ(indices.size(), 10u);
}

TEST_F(RandomTest, SampleIndicesUnique) {
    auto indices = Random::Instance().SampleIndices(100, 20);
    std::set<size_t> uniqueSet(indices.begin(), indices.end());
    EXPECT_EQ(uniqueSet.size(), indices.size());
}

TEST_F(RandomTest, SampleIndicesInRange) {
    auto indices = Random::Instance().SampleIndices(50, 15);
    for (size_t idx : indices) {
        EXPECT_LT(idx, 50u);
    }
}

TEST_F(RandomTest, SampleIndicesKGreaterThanN) {
    // When k >= n, should return all indices
    auto indices = Random::Instance().SampleIndices(10, 15);
    EXPECT_EQ(indices.size(), 10u);

    std::set<size_t> uniqueSet(indices.begin(), indices.end());
    EXPECT_EQ(uniqueSet.size(), 10u);
}

TEST_F(RandomTest, SampleIndicesEqualsN) {
    auto indices = Random::Instance().SampleIndices(10, 10);
    EXPECT_EQ(indices.size(), 10u);

    std::set<size_t> uniqueSet(indices.begin(), indices.end());
    EXPECT_EQ(uniqueSet.size(), 10u);
}

TEST_F(RandomTest, SampleVectorCorrectSize) {
    std::vector<int> items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto sampled = Random::Instance().Sample(items, 5);
    EXPECT_EQ(sampled.size(), 5u);
}

TEST_F(RandomTest, SampleVectorUnique) {
    std::vector<int> items = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    auto sampled = Random::Instance().Sample(items, 5);

    std::set<int> sampledSet(sampled.begin(), sampled.end());
    EXPECT_EQ(sampledSet.size(), 5u);

    // All sampled items should be from original
    for (int v : sampled) {
        EXPECT_TRUE(std::find(items.begin(), items.end(), v) != items.end());
    }
}

// ============================================================================
// Shuffle Tests
// ============================================================================

TEST_F(RandomTest, ShufflePreservesElements) {
    std::vector<int> items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> original = items;

    Random::Instance().Shuffle(items);

    // Same size
    EXPECT_EQ(items.size(), original.size());

    // Same elements (just reordered)
    std::vector<int> sortedOriginal = original;
    std::vector<int> sortedShuffled = items;
    std::sort(sortedOriginal.begin(), sortedOriginal.end());
    std::sort(sortedShuffled.begin(), sortedShuffled.end());
    EXPECT_EQ(sortedOriginal, sortedShuffled);
}

TEST_F(RandomTest, ShuffleChangesOrder) {
    std::vector<int> items = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<int> original = items;

    // Shuffle multiple times to ensure at least one changes the order
    bool changed = false;
    for (int trial = 0; trial < 10; ++trial) {
        items = original;
        Random::Instance().Shuffle(items);
        if (items != original) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

// ============================================================================
// Seed Reproducibility Tests
// ============================================================================

TEST_F(RandomTest, SeedReproducibility) {
    Random::Instance().SetSeed(99999);

    std::vector<double> first;
    for (int i = 0; i < 100; ++i) {
        first.push_back(Random::Instance().Double());
    }

    // Reset with same seed
    Random::Instance().SetSeed(99999);

    std::vector<double> second;
    for (int i = 0; i < 100; ++i) {
        second.push_back(Random::Instance().Double());
    }

    EXPECT_EQ(first, second);
}

TEST_F(RandomTest, DifferentSeedsDifferentSequence) {
    Random::Instance().SetSeed(11111);
    std::vector<double> first;
    for (int i = 0; i < 100; ++i) {
        first.push_back(Random::Instance().Double());
    }

    Random::Instance().SetSeed(22222);
    std::vector<double> second;
    for (int i = 0; i < 100; ++i) {
        second.push_back(Random::Instance().Double());
    }

    EXPECT_NE(first, second);
}

// ============================================================================
// Free Function Tests
// ============================================================================

TEST_F(RandomTest, FreeFunctionRandomInt) {
    for (int i = 0; i < 100; ++i) {
        int32_t v = RandomInt(0, 100);
        EXPECT_GE(v, 0);
        EXPECT_LE(v, 100);
    }
}

TEST_F(RandomTest, FreeFunctionRandomDouble) {
    for (int i = 0; i < 100; ++i) {
        double v = RandomDouble(10.0, 20.0);
        EXPECT_GE(v, 10.0);
        EXPECT_LT(v, 20.0);
    }
}

TEST_F(RandomTest, FreeFunctionRandomGaussian) {
    // Just ensure it doesn't crash and returns reasonable values
    double v = RandomGaussian(0.0, 1.0);
    EXPECT_FALSE(std::isnan(v));
    EXPECT_FALSE(std::isinf(v));
}

TEST_F(RandomTest, FreeFunctionRandomSample) {
    auto indices = RandomSample(100, 10);
    EXPECT_EQ(indices.size(), 10u);
}

TEST_F(RandomTest, FreeFunctionSetRandomSeed) {
    SetRandomSeed(54321);
    double v1 = RandomDouble();

    SetRandomSeed(54321);
    double v2 = RandomDouble();

    EXPECT_EQ(v1, v2);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(RandomTest, ThreadLocalInstances) {
    // Each thread should have its own instance
    std::atomic<int> successCount{0};
    const int numThreads = 4;
    const int samplesPerThread = 1000;

    auto threadFunc = [&successCount, samplesPerThread]() {
        // Set a unique seed for this thread
        Random::Instance().SetSeed(
            std::hash<std::thread::id>{}(std::this_thread::get_id())
        );

        bool allInRange = true;
        for (int i = 0; i < samplesPerThread; ++i) {
            double v = Random::Instance().Double();
            if (v < 0.0 || v >= 1.0) {
                allInRange = false;
                break;
            }
        }

        if (allInRange) {
            ++successCount;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(threadFunc);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successCount.load(), numThreads);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(RandomTest, LargeRangeSampling) {
    // Sample from a large range
    auto indices = Random::Instance().SampleIndices(1000000, 100);
    EXPECT_EQ(indices.size(), 100u);

    std::set<size_t> uniqueSet(indices.begin(), indices.end());
    EXPECT_EQ(uniqueSet.size(), 100u);
}

TEST_F(RandomTest, SingleElementSample) {
    auto indices = Random::Instance().SampleIndices(100, 1);
    EXPECT_EQ(indices.size(), 1u);
    EXPECT_LT(indices[0], 100u);
}

TEST_F(RandomTest, EmptyVectorSample) {
    std::vector<int> empty;
    auto sampled = Random::Instance().Sample(empty, 5);
    EXPECT_TRUE(sampled.empty());
}

TEST_F(RandomTest, ShuffleEmptyVector) {
    std::vector<int> empty;
    Random::Instance().Shuffle(empty);
    EXPECT_TRUE(empty.empty());
}

TEST_F(RandomTest, ShuffleSingleElement) {
    std::vector<int> single = {42};
    Random::Instance().Shuffle(single);
    EXPECT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], 42);
}

#pragma once

// Internal-only caliper measurement helpers (not part of public SDK)

#include <QiVision/Core/QImage.h>
#include <QiVision/Measure/MeasureTypes.h>
#include <QiVision/Measure/MeasureHandle.h>

#include <vector>

namespace Qi::Vision::Measure {

std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureRectangle2& handle,
                                    double sigma,
                                    double threshold,
                                    const std::string& transition,
                                    const std::string& select);

std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureArc& handle,
                                    double sigma,
                                    double threshold,
                                    const std::string& transition,
                                    const std::string& select);

std::vector<EdgeResult> MeasurePos(const QImage& image,
                                    const MeasureConcentricCircles& handle,
                                    double sigma,
                                    double threshold,
                                    const std::string& transition,
                                    const std::string& select);

std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureRectangle2& handle,
                                      double sigma,
                                      double threshold,
                                      const std::string& transition,
                                      const std::string& select);

std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureArc& handle,
                                      double sigma,
                                      double threshold,
                                      const std::string& transition,
                                      const std::string& select);

std::vector<PairResult> MeasurePairs(const QImage& image,
                                      const MeasureConcentricCircles& handle,
                                      double sigma,
                                      double threshold,
                                      const std::string& transition,
                                      const std::string& select);

std::vector<EdgeResult> FuzzyMeasurePos(const QImage& image,
                                         const MeasureRectangle2& handle,
                                         double sigma,
                                         double threshold,
                                         const std::string& transition,
                                         const std::string& select,
                                         double fuzzyThresh = 0.5,
                                         MeasureStats* stats = nullptr);

std::vector<EdgeResult> FuzzyMeasurePos(const QImage& image,
                                         const MeasureArc& handle,
                                         double sigma,
                                         double threshold,
                                         const std::string& transition,
                                         const std::string& select,
                                         double fuzzyThresh = 0.5,
                                         MeasureStats* stats = nullptr);

std::vector<EdgeResult> FuzzyMeasurePos(const QImage& image,
                                         const MeasureConcentricCircles& handle,
                                         double sigma,
                                         double threshold,
                                         const std::string& transition,
                                         const std::string& select,
                                         double fuzzyThresh = 0.5,
                                         MeasureStats* stats = nullptr);

std::vector<PairResult> FuzzyMeasurePairs(const QImage& image,
                                           const MeasureRectangle2& handle,
                                           double sigma,
                                           double threshold,
                                           const std::string& transition,
                                           const std::string& select,
                                           double fuzzyThresh = 0.5,
                                           MeasureStats* stats = nullptr);

std::vector<PairResult> FuzzyMeasurePairs(const QImage& image,
                                           const MeasureArc& handle,
                                           double sigma,
                                           double threshold,
                                           const std::string& transition,
                                           const std::string& select,
                                           double fuzzyThresh = 0.5,
                                           MeasureStats* stats = nullptr);

std::vector<PairResult> FuzzyMeasurePairs(const QImage& image,
                                           const MeasureConcentricCircles& handle,
                                           double sigma,
                                           double threshold,
                                           const std::string& transition,
                                           const std::string& select,
                                           double fuzzyThresh = 0.5,
                                           MeasureStats* stats = nullptr);

std::vector<double> ExtractMeasureProfile(const QImage& image,
                                           const MeasureRectangle2& handle,
                                           const std::string& interp = "bilinear");

std::vector<double> ExtractMeasureProfile(const QImage& image,
                                           const MeasureArc& handle,
                                           const std::string& interp = "bilinear");

std::vector<double> ExtractMeasureProfile(const QImage& image,
                                           const MeasureConcentricCircles& handle,
                                           const std::string& interp = "bilinear");

Point2d ProfileToImage(const MeasureRectangle2& handle, double profilePos);
Point2d ProfileToImage(const MeasureArc& handle, double profilePos);
Point2d ProfileToImage(const MeasureConcentricCircles& handle, double profilePos);

int32_t GetNumSamples(const MeasureRectangle2& handle);
int32_t GetNumSamples(const MeasureArc& handle);
int32_t GetNumSamples(const MeasureConcentricCircles& handle);

std::vector<EdgeResult> SelectEdges(const std::vector<EdgeResult>& edges,
                                     EdgeSelectMode mode,
                                     int32_t maxCount = MAX_EDGES);

std::vector<PairResult> SelectPairs(const std::vector<PairResult>& pairs,
                                     PairSelectMode mode,
                                     int32_t maxCount = MAX_EDGES);

enum class EdgeSortBy { Position, Amplitude, Score };
void SortEdges(std::vector<EdgeResult>& edges, EdgeSortBy criterion, bool ascending = true);

enum class PairSortBy { Position, Width, Score, Symmetry };
void SortPairs(std::vector<PairResult>& pairs, PairSortBy criterion, bool ascending = true);

} // namespace Qi::Vision::Measure


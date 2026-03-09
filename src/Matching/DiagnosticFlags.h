#pragma once

namespace Qi::Vision::Internal {

struct ScaleDiagFlags {
    // v1 switches (post-processing)
    bool bypassS7Cluster = false;          // A: skip SpatialNMSCluster
    bool disableAngleScaleSuppress = false; // B: skip Phase3 angle/scale suppress
    bool mode1ForceNoScalePath = false;     // C: force no-scale subpath in LS mode1
    bool preNmsMinScoreGate = false;        // D: minScore filter before SpatialNMSCluster

    // v2 switches (upstream)
    bool disableCoarseAngleSuppress = false; // B1: skip angle suppress in CollectCandidatesNMS
    bool disableCoarseSpatialNMS = false;    // B2: skip 3x3 spatial NMS in CollectCandidatesNMS
    bool clampRefineScale = false;           // B4: clamp testS to [scaleMin,scaleMax] in RefineAtLevelScaled
};

extern ScaleDiagFlags g_scaleDiag;

} // namespace Qi::Vision::Internal

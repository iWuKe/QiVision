#pragma once

#include <QiVision/Core/Export.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t QV_Bool;
#define QV_TRUE 1
#define QV_FALSE 0

typedef enum QV_Status {
    QV_STATUS_OK = 0,
    QV_STATUS_INVALID_ARGUMENT = 1,
    QV_STATUS_OUT_OF_RANGE = 2,
    QV_STATUS_INSUFFICIENT_DATA = 3,
    QV_STATUS_CONVERGENCE_FAILED = 4,
    QV_STATUS_IO_ERROR = 5,
    QV_STATUS_UNSUPPORTED = 6,
    QV_STATUS_VERSION_MISMATCH = 7,
    QV_STATUS_OUT_OF_MEMORY = 8,
    QV_STATUS_INTERNAL_ERROR = 9,
    QV_STATUS_UNKNOWN_ERROR = 10
} QV_Status;

/* Opaque handles for future C API wrappers */
typedef struct QV_Handle_ QV_Handle;
typedef struct QV_Image_ QV_Image;
typedef struct QV_Region_ QV_Region;
typedef struct QV_Contour_ QV_Contour;
typedef struct QV_ContourArray_ QV_ContourArray;
typedef struct QV_Matrix_ QV_Matrix;
typedef struct QV_ShapeModel_ QV_ShapeModel;
typedef struct QV_NCCModel_ QV_NCCModel;
typedef struct QV_MetrologyModel_ QV_MetrologyModel;
typedef struct QV_CaliperArray_ QV_CaliperArray;

QIVISION_API const char* QIVISION_CALL QV_StatusToString(QV_Status status);

QIVISION_API QV_Status QIVISION_CALL QV_GetLastError(void);
QIVISION_API const char* QIVISION_CALL QV_GetLastErrorMessage(void);
QIVISION_API void QIVISION_CALL QV_ClearLastError(void);

QIVISION_API const char* QIVISION_CALL QV_GetVersionString(void);
QIVISION_API QV_Status QIVISION_CALL QV_GetVersionNumbers(int* major, int* minor, int* patch);

#ifdef __cplusplus
} // extern "C"
#endif

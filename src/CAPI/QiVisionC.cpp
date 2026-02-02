#include <QiVision/CAPI/QiVisionC.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/QiVision.h>

#include <new>
#include <string>

namespace {

thread_local QV_Status g_lastStatus = QV_STATUS_OK;
thread_local std::string g_lastMessage;

void SetLastError(QV_Status status, const char* message) {
    g_lastStatus = status;
    g_lastMessage = message ? message : "";
}

[[maybe_unused]] void SetLastError(QV_Status status, const std::string& message) {
    g_lastStatus = status;
    g_lastMessage = message;
}

void ClearLastErrorInternal() {
    g_lastStatus = QV_STATUS_OK;
    g_lastMessage.clear();
}

[[maybe_unused]] QV_Status StatusFromException(const std::exception& ex) {
    using namespace Qi::Vision;
    if (dynamic_cast<const InvalidArgumentException*>(&ex)) {
        return QV_STATUS_INVALID_ARGUMENT;
    }
    if (dynamic_cast<const OutOfRangeException*>(&ex)) {
        return QV_STATUS_OUT_OF_RANGE;
    }
    if (dynamic_cast<const InsufficientDataException*>(&ex)) {
        return QV_STATUS_INSUFFICIENT_DATA;
    }
    if (dynamic_cast<const ConvergenceException*>(&ex)) {
        return QV_STATUS_CONVERGENCE_FAILED;
    }
    if (dynamic_cast<const IOException*>(&ex)) {
        return QV_STATUS_IO_ERROR;
    }
    if (dynamic_cast<const UnsupportedException*>(&ex)) {
        return QV_STATUS_UNSUPPORTED;
    }
    if (dynamic_cast<const VersionMismatchException*>(&ex)) {
        return QV_STATUS_VERSION_MISMATCH;
    }
    return QV_STATUS_INTERNAL_ERROR;
}

} // namespace

extern "C" {

QIVISION_API const char* QIVISION_CALL QV_StatusToString(QV_Status status) {
    switch (status) {
        case QV_STATUS_OK: return "OK";
        case QV_STATUS_INVALID_ARGUMENT: return "Invalid argument";
        case QV_STATUS_OUT_OF_RANGE: return "Out of range";
        case QV_STATUS_INSUFFICIENT_DATA: return "Insufficient data";
        case QV_STATUS_CONVERGENCE_FAILED: return "Convergence failed";
        case QV_STATUS_IO_ERROR: return "I/O error";
        case QV_STATUS_UNSUPPORTED: return "Unsupported";
        case QV_STATUS_VERSION_MISMATCH: return "Version mismatch";
        case QV_STATUS_OUT_OF_MEMORY: return "Out of memory";
        case QV_STATUS_INTERNAL_ERROR: return "Internal error";
        case QV_STATUS_UNKNOWN_ERROR: return "Unknown error";
        default: return "Unknown error";
    }
}

QIVISION_API QV_Status QIVISION_CALL QV_GetLastError(void) {
    return g_lastStatus;
}

QIVISION_API const char* QIVISION_CALL QV_GetLastErrorMessage(void) {
    return g_lastMessage.c_str();
}

QIVISION_API void QIVISION_CALL QV_ClearLastError(void) {
    ClearLastErrorInternal();
}

QIVISION_API const char* QIVISION_CALL QV_GetVersionString(void) {
    return Qi::Vision::GetVersion();
}

QIVISION_API QV_Status QIVISION_CALL QV_GetVersionNumbers(int* major, int* minor, int* patch) {
    if (!major || !minor || !patch) {
        SetLastError(QV_STATUS_INVALID_ARGUMENT, "QV_GetVersionNumbers: null output pointer");
        return QV_STATUS_INVALID_ARGUMENT;
    }
    Qi::Vision::GetVersion(*major, *minor, *patch);
    ClearLastErrorInternal();
    return QV_STATUS_OK;
}

} // extern "C"

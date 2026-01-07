# 标定与坐标系转换设计规则

> 本文档定义 QiVision 中标定 (Calibration) 和坐标系转换 (Coordinate Transform) 的设计规范

---

## 1. 坐标系定义

### 1.1 坐标系类型

| 坐标系 | 英文名 | 单位 | 原点 | 用途 |
|--------|--------|------|------|------|
| 图像坐标系 | Image | 像素 | 左上角 | 图像处理、模板匹配 |
| 相机坐标系 | Camera | mm | 光心 | 3D 视觉、深度计算 |
| 世界坐标系 | World | mm | 用户定义 | 物理测量、定位 |
| 机器人基座坐标系 | RobotBase | mm | 机器人基座 | 手眼标定 |
| 机器人工具坐标系 | RobotTool | mm | TCP | 抓取定位 |
| 标定板坐标系 | CalibPlate | mm | 标定板原点 | 标定过程 |

### 1.2 坐标系约定

```
图像坐标系 (右手系):
    Origin ──────► Col (X)
      │
      │
      ▼
     Row (Y)

注意: Row = Y, Col = X
     Point2d(x, y) 对应像素 (col, row)

世界坐标系 (右手系):
    Z
    │
    │
    Origin ──────► X
   /
  /
 Y

相机坐标系 (右手系):
    Z (光轴方向, 指向场景)
    │
    │
    Origin ──────► X
   /
  /
 Y
```

### 1.3 QiVision 中的坐标表示

```cpp
// 2D 像素坐标（亚像素）
struct Point2d {
    double x;  // 对应 col
    double y;  // 对应 row
};

// 3D 空间坐标
struct Point3d {
    double x;
    double y;
    double z;
};

// 注意：函数参数顺序遵循 (row, col) 还是 (x, y) 必须在命名中明确
// 推荐：图像操作用 (row, col)，几何计算用 (x, y)
```

---

## 2. 位姿 (Pose) 表示

### 2.1 位姿定义

位姿 = 位置 (Translation) + 姿态 (Rotation)

表示一个坐标系相对于另一个坐标系的变换关系。

### 2.2 QPose 设计规则

```cpp
namespace Qi::Vision {

// 位姿表示：6 自由度
struct QPose {
    // 位置 (平移)
    double tx, ty, tz;  // mm
    
    // 姿态 (旋转) - 使用欧拉角 (Halcon 兼容)
    double rx, ry, rz;  // rad，旋转顺序 ZYX (Tait-Bryan)
    
    // 旋转顺序定义
    enum class RotationOrder {
        ZYX,    // Halcon 默认: Rz * Ry * Rx
        XYZ,
        ZXZ     // 欧拉角
    };
    
    RotationOrder order = RotationOrder::ZYX;
};

// 注意事项：
// 1. 欧拉角有万向锁问题，内部计算建议转为旋转矩阵或四元数
// 2. 对外接口保持欧拉角（与 Halcon 兼容）
// 3. rx, ry, rz 范围: [-π, π]

}
```

### 2.3 旋转表示转换

```cpp
namespace Qi::Vision::Internal {

// 支持的旋转表示
// 1. 欧拉角 (EulerAngles) - 对外接口
// 2. 旋转矩阵 (RotationMatrix) - 内部计算
// 3. 四元数 (Quaternion) - 插值、组合
// 4. 轴角 (AxisAngle) - 特定算法

// 必须实现的转换函数
Matrix3x3 EulerToRotationMatrix(double rx, double ry, double rz, RotationOrder order);
void RotationMatrixToEuler(const Matrix3x3& R, double& rx, double& ry, double& rz, RotationOrder order);
Quaternion RotationMatrixToQuaternion(const Matrix3x3& R);
Matrix3x3 QuaternionToRotationMatrix(const Quaternion& q);

}
```

---

## 3. 变换矩阵

### 3.1 2D 变换矩阵

| 变换类型 | 自由度 | 矩阵形式 | 用途 |
|----------|--------|----------|------|
| 刚体变换 | 3 | 2×3 | 模板匹配结果 |
| 相似变换 | 4 | 2×3 | 各向同性缩放匹配 |
| 仿射变换 | 6 | 2×3 | 各向异性缩放匹配 |
| 透视变换 | 8 | 3×3 | 平面标定、图像校正 |

```cpp
namespace Qi::Vision {

// 2D 齐次变换矩阵 (3x3)
class QHomMat2d {
public:
    // 创建
    static QHomMat2d Identity();
    static QHomMat2d FromTranslation(double tx, double ty);
    static QHomMat2d FromRotation(double angle, double cx = 0, double cy = 0);
    static QHomMat2d FromScale(double sx, double sy, double cx = 0, double cy = 0);
    static QHomMat2d FromAffine(double a11, double a12, double a21, double a22, double tx, double ty);
    
    // 组合: this * other (先 other 后 this)
    QHomMat2d Compose(const QHomMat2d& other) const;
    
    // 逆变换
    QHomMat2d Invert() const;
    
    // 点变换
    Point2d TransformPoint(const Point2d& p) const;
    std::vector<Point2d> TransformPoints(const std::vector<Point2d>& points) const;
    
    // 轮廓变换
    QContour TransformContour(const QContour& contour) const;
    
    // 区域变换 (需要重采样)
    QRegion TransformRegion(const QRegion& region) const;
    
    // 分解 (提取参数)
    void Decompose(double& tx, double& ty, double& angle, double& sx, double& sy) const;
    
    // 数据访问
    double At(int row, int col) const;
    const double* Data() const;  // row-major
    
private:
    double data_[9];  // 3x3, row-major
};

}
```

### 3.2 3D 变换矩阵

```cpp
namespace Qi::Vision {

// 3D 齐次变换矩阵 (4x4)
class QHomMat3d {
public:
    // 从位姿创建
    static QHomMat3d FromPose(const QPose& pose);
    
    // 转换为位姿
    QPose ToPose() const;
    
    // 组合
    QHomMat3d Compose(const QHomMat3d& other) const;
    
    // 逆变换
    QHomMat3d Invert() const;
    
    // 点变换
    Point3d TransformPoint(const Point3d& p) const;
    
    // 提取旋转部分
    Matrix3x3 GetRotation() const;
    
    // 提取平移部分
    Point3d GetTranslation() const;
    
private:
    double data_[16];  // 4x4, row-major
};

}
```

---

## 4. 相机模型

### 4.1 相机内参

```cpp
namespace Qi::Vision::Calib {

struct CameraIntrinsics {
    // 焦距 (像素单位)
    double fx, fy;
    
    // 主点
    double cx, cy;
    
    // 畸变系数 (Brown-Conrady 模型)
    // 径向畸变
    double k1, k2, k3;      // 一般 k3 可选
    // 切向畸变
    double p1, p2;
    
    // 图像尺寸 (标定时的尺寸)
    int width, height;
    
    // 像素尺寸 (可选, 用于计算真实焦距)
    double pixelSizeX = 0;  // mm/pixel
    double pixelSizeY = 0;
};

// 内参矩阵 K (3x3)
// | fx  0  cx |
// | 0  fy  cy |
// | 0   0   1 |

}
```

### 4.2 相机外参

```cpp
namespace Qi::Vision::Calib {

// 相机外参 = 相机相对于世界坐标系的位姿
// 或者说：世界坐标系到相机坐标系的变换
struct CameraExtrinsics {
    QPose pose;  // 世界 → 相机
    
    // 等价表示
    // R: 旋转矩阵 (3x3)
    // t: 平移向量 (3x1)
    // P_camera = R * P_world + t
};

}
```

### 4.3 相机模型类

```cpp
namespace Qi::Vision::Calib {

class CameraModel {
public:
    // 设置/获取参数
    void SetIntrinsics(const CameraIntrinsics& intrinsics);
    void SetExtrinsics(const CameraExtrinsics& extrinsics);
    
    // 投影: 3D → 2D
    Point2d Project(const Point3d& worldPoint) const;
    std::vector<Point2d> Project(const std::vector<Point3d>& worldPoints) const;
    
    // 反投影: 2D → 射线 (需要深度信息才能得到 3D 点)
    Ray3d Unproject(const Point2d& imagePoint) const;
    
    // 反投影到指定平面 (Z=0 平面或任意平面)
    Point3d UnprojectToPlane(const Point2d& imagePoint, const Plane3d& plane) const;
    Point3d UnprojectToZPlane(const Point2d& imagePoint, double z = 0) const;
    
    // 畸变校正
    Point2d UndistortPoint(const Point2d& distorted) const;
    Point2d DistortPoint(const Point2d& undistorted) const;
    QImage UndistortImage(const QImage& image) const;
    
    // 生成去畸变映射表 (用于高效批量处理)
    void CreateUndistortMaps(QImage& mapX, QImage& mapY) const;
    
private:
    CameraIntrinsics intrinsics_;
    CameraExtrinsics extrinsics_;
};

}
```

---

## 5. 坐标系转换 API

### 5.1 核心转换函数

```cpp
namespace Qi::Vision::Calib {

// ============================================================
// 2D 图像坐标 ↔ 世界坐标 (假设物体在 Z=0 平面)
// ============================================================

// 图像点 → 世界点 (Z=0 平面)
Point2d ImageToWorld(
    const Point2d& imagePoint,
    const CameraModel& camera
);

std::vector<Point2d> ImageToWorld(
    const std::vector<Point2d>& imagePoints,
    const CameraModel& camera
);

// 世界点 → 图像点
Point2d WorldToImage(
    const Point2d& worldPoint,  // Z 默认为 0
    const CameraModel& camera
);

// ============================================================
// 3D 坐标系转换
// ============================================================

// 世界坐标 → 相机坐标
Point3d WorldToCamera(const Point3d& worldPoint, const CameraModel& camera);

// 相机坐标 → 图像坐标
Point2d CameraToImage(const Point3d& cameraPoint, const CameraModel& camera);

// 图像坐标 → 相机坐标 (需要深度)
Point3d ImageToCamera(const Point2d& imagePoint, double depth, const CameraModel& camera);

// ============================================================
// 使用变换矩阵的通用转换
// ============================================================

// 2D 点变换
Point2d TransformPoint2d(const Point2d& point, const QHomMat2d& transform);

// 3D 点变换  
Point3d TransformPoint3d(const Point3d& point, const QHomMat3d& transform);

// 变换矩阵组合
// 结果表示: 先做 second，再做 first
QHomMat2d ComposeTransform(const QHomMat2d& first, const QHomMat2d& second);
QHomMat3d ComposeTransform(const QHomMat3d& first, const QHomMat3d& second);

}
```

### 5.2 模板匹配结果坐标转换

**这是最常见的应用场景**

```cpp
namespace Qi::Vision::Matching {

// 匹配结果已包含变换信息
struct MatchResult {
    double row, col;        // 匹配位置 (图像坐标)
    double angle;           // 旋转角度 (rad)
    double scaleRow, scaleCol;  // 缩放
    double score;           // 匹配得分
    QHomMat2d transform;    // 模板 → 图像 的变换矩阵
};

}

namespace Qi::Vision::Calib {

// ============================================================
// 匹配结果 → 世界坐标
// ============================================================

struct MatchWorldResult {
    Point2d position;       // 世界坐标位置 (mm)
    double angle;           // 世界坐标系下的角度 (rad)
    double scaleX, scaleY;  // 缩放 (如果标定了像素尺寸)
};

// 将匹配结果转换到世界坐标
MatchWorldResult MatchResultToWorld(
    const Matching::MatchResult& match,
    const CameraModel& camera
);

// ============================================================
// 模板点 → 世界坐标
// ============================================================

// 将模板上的参考点转换到世界坐标
// templatePoint: 模板坐标系下的点
// match: 匹配结果
// camera: 相机模型
Point2d TemplatePointToWorld(
    const Point2d& templatePoint,
    const Matching::MatchResult& match,
    const CameraModel& camera
);

// 批量转换
std::vector<Point2d> TemplatePointsToWorld(
    const std::vector<Point2d>& templatePoints,
    const Matching::MatchResult& match,
    const CameraModel& camera
);

}
```

### 5.3 使用示例

```cpp
// 示例: 模板匹配后获取世界坐标

// 1. 创建模板并匹配
ShapeModel model;
model.Create(templateImage, params);
auto matches = model.Find(searchImage);

// 2. 设置相机模型 (已标定)
CameraModel camera;
camera.SetIntrinsics(intrinsics);
camera.SetExtrinsics(extrinsics);

// 3. 获取匹配位置的世界坐标
if (!matches.empty()) {
    const auto& match = matches[0];
    
    // 方法 A: 直接转换匹配中心
    auto worldResult = MatchResultToWorld(match, camera);
    printf("World position: (%.3f, %.3f) mm, angle: %.2f deg\n",
           worldResult.position.x, worldResult.position.y,
           worldResult.angle * 180 / M_PI);
    
    // 方法 B: 转换模板上的特定点 (如抓取点)
    Point2d grabPointInTemplate(50.0, 30.0);  // 模板坐标系
    Point2d grabPointInWorld = TemplatePointToWorld(
        grabPointInTemplate, match, camera);
    
    // 方法 C: 手动分步转换
    // 3.1 模板点 → 图像点
    Point2d grabPointInImage = match.transform.TransformPoint(grabPointInTemplate);
    // 3.2 图像点 → 世界点
    Point2d grabPointInWorld2 = ImageToWorld(grabPointInImage, camera);
}
```

---

## 6. 手眼标定

### 6.1 手眼标定类型

```
Eye-in-Hand (眼在手上):
    相机安装在机器人末端
    Camera → Tool → Base
    求解: T_camera_to_tool

Eye-to-Hand (眼在手外):
    相机固定在外部
    Camera → Base (固定)
    Object → Tool → Base
    求解: T_camera_to_base
```

### 6.2 手眼标定 API

```cpp
namespace Qi::Vision::Calib {

enum class HandEyeType {
    EyeInHand,
    EyeToHand
};

struct HandEyeCalibrationData {
    // 每组数据包含:
    // 1. 机器人位姿 (工具相对于基座)
    std::vector<QPose> robotPoses;
    
    // 2. 标定板位姿 (标定板相对于相机)
    //    通过 CalibrateExtrinsics 获得
    std::vector<QPose> calibPlatePoses;
};

struct HandEyeResult {
    QPose cameraToTool;    // Eye-in-Hand: 相机→工具
    QPose cameraToBase;    // Eye-to-Hand: 相机→基座
    
    double reprojError;    // 重投影误差 (pixel)
    double poseError;      // 位姿误差 (mm)
};

// 手眼标定
HandEyeResult CalibrateHandEye(
    const HandEyeCalibrationData& data,
    HandEyeType type
);

}
```

### 6.3 手眼坐标转换

```cpp
namespace Qi::Vision::Calib {

// Eye-in-Hand: 图像点 → 机器人基座坐标
Point3d ImageToRobotBase_EyeInHand(
    const Point2d& imagePoint,
    double objectZ,                 // 物体 Z 坐标 (基座坐标系)
    const CameraModel& camera,
    const QPose& cameraToTool,      // 手眼标定结果
    const QPose& toolToBase         // 当前机器人位姿
);

// Eye-to-Hand: 图像点 → 机器人基座坐标
Point3d ImageToRobotBase_EyeToHand(
    const Point2d& imagePoint,
    double objectZ,                 // 物体 Z 坐标 (基座坐标系)
    const CameraModel& camera,
    const QPose& cameraToBase       // 手眼标定结果 (固定)
);

// 计算抓取位姿
QPose CalculateGraspPose(
    const Matching::MatchResult& match,
    const Point2d& graspPointInTemplate,  // 模板中的抓取点
    double approachHeight,                 // 接近高度
    const CameraModel& camera,
    const QPose& cameraToTool,
    const QPose& currentToolToBase
);

}
```

---

## 7. 多相机系统

### 7.1 相机间变换

```cpp
namespace Qi::Vision::Calib {

// 双目标定结果
struct StereoCalibrationResult {
    CameraIntrinsics leftCamera;
    CameraIntrinsics rightCamera;
    QPose rightToLeft;  // 右相机相对于左相机的位姿
    
    // 立体校正矩阵
    QHomMat3d R1, R2;   // 旋转校正
    Matrix3x4 P1, P2;   // 投影矩阵
};

// 相机间点转换
Point2d TransformBetweenCameras(
    const Point2d& pointInCam1,
    double depth,
    const CameraModel& cam1,
    const CameraModel& cam2,
    const QPose& cam2ToCam1
);

}
```

---

## 8. 内部实现规则

### 8.1 Internal 层依赖

| 模块 | 依赖的 Internal 模块 |
|------|---------------------|
| Calib | Matrix, Solver, Eigen, Fitting, Homography, SubPixel |

### 8.2 实现要点

1. **齐次坐标** - 所有变换使用齐次坐标，避免特殊处理
2. **数值稳定性** - 矩阵求逆使用 SVD，避免直接求逆
3. **奇异性检查** - 检测退化情况（如点共线）
4. **精度传播** - 标定误差影响后续转换精度

### 8.3 坐标系转换链验证

```cpp
// 每次坐标系转换都应该可逆验证
Point3d original(100, 200, 0);
Point2d imagePoint = WorldToImage(original, camera);
Point3d recovered = ImageToWorld3d(imagePoint, 0, camera);

// |original - recovered| < epsilon
assert(Distance(original, recovered) < 0.01);  // mm
```

---

## 9. 精度规格

### 9.1 标定精度

| 标定类型 | 条件 | 精度要求 |
|----------|------|----------|
| 相机内参 | 标准标定板 | 重投影误差 < 0.3 pixel |
| 相机外参 | - | 位置误差 < 0.5 mm |
| 手眼标定 | ≥15 组数据 | 位置误差 < 1.0 mm |

### 9.2 坐标转换精度

| 转换类型 | 条件 | 精度要求 |
|----------|------|----------|
| 图像→世界 (Z=0) | 相机高度 500mm | < 0.1 mm |
| 模板点→世界 | 含匹配误差 | < 0.2 mm |

---

## 10. Halcon 算子对应

| QiVision API | Halcon 算子 |
|--------------|-------------|
| `ImageToWorld` | `image_points_to_world_plane` |
| `WorldToImage` | `project_3d_point` |
| `QHomMat2d::TransformPoint` | `affine_trans_point_2d` |
| `QHomMat3d::FromPose` | `pose_to_hom_mat3d` |
| `QHomMat3d::Compose` | `hom_mat3d_compose` |
| `CalibrateHandEye` | `hand_eye_calibration` |
| `CameraModel::UndistortImage` | `change_radial_distortion_image` |

---

## 11. 变更日志

- 2024-12-24: 创建文档，定义坐标系规范和转换 API

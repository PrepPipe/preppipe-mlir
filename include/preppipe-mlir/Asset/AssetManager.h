// SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef PREPPIPE_ASSET_MANAGER_H
#define PREPPIPE_ASSET_MANAGER_H

#include "mlir/IR/MLIRContext.h"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace llvm {
class raw_ostream;
}

namespace preppipe {
namespace asset {

/// AssetManager 类负责管理资源素材，包括：
/// 1. 维护临时目录
/// 2. 管理 path 到实际存储路径的映射
/// 3. 处理资源的添加、获取和管理
class AssetManager {
public:
  /// 构造函数
  AssetManager();

  /// 析构函数
  ~AssetManager();

  /// 添加资源素材
  /// @param path 资源的逻辑路径（对应 AssetOp 的 path 属性）
  /// @param data 资源的二进制数据
  /// @param preferred_name 资源的首选名称
  /// @return 资源的实际存储路径
  std::string addAsset(const std::string &path, const std::vector<char> &data,
                       const std::string &preferred_name = "");

  /// 添加外部资源（已存在于文件系统中）
  /// @param path 资源的逻辑路径（对应 AssetOp 的 path 属性）
  /// @param external_path 资源的外部实际路径
  /// @return 资源的实际存储路径
  std::string addExternalAsset(const std::string &path,
                               const std::string &external_path);

  /// 获取资源的实际存储路径
  /// @param path 资源的逻辑路径
  /// @return 资源的实际存储路径，如果不存在则返回 std::nullopt
  std::optional<std::string> getAssetPath(const std::string &path) const;

  /// 获取临时目录路径
  /// @return 临时目录的绝对路径
  const std::string &getTempDir() const;

  /// 清理所有资源（可选）
  void cleanup();

  /// 打印资源映射信息
  /// @param os 输出流
  void print(llvm::raw_ostream &os) const;

private:
  /// 创建临时目录
  void createTempDir();

  /// 为资源生成唯一文件名
  /// @param preferred_name 首选名称
  /// @param data 资源数据（用于生成哈希）
  /// @return 唯一文件名
  std::string generateUniqueFilename(const std::string &preferred_name,
                                     const std::vector<char> &data);

  /// 临时目录路径
  std::string tempDir;

  /// 资源映射：逻辑路径 -> 实际存储路径
  std::unordered_map<std::string, std::string> assetMap;

  /// 外部资源映射：外部路径 -> 实际存储路径
  std::unordered_map<std::string, std::string> externalAssetMap;

  /// 文件名计数器，用于生成唯一文件名
  unsigned filenameCounter = 0;
};

} // namespace asset
} // namespace preppipe

#endif // PREPPIPE_ASSET_MANAGER_H
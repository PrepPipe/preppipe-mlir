// SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
// SPDX-License-Identifier: Apache-2.0

#include "preppipe-mlir/Asset/AssetManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

using namespace preppipe::asset;

namespace {

/// 生成随机字符串，用于临时目录名称
std::string generateRandomString(size_t length) {
  const std::string chars =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, chars.size() - 1);

  std::string result;
  result.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    result += chars[dis(gen)];
  }
  return result;
}

/// 计算数据的哈希值，用于生成唯一文件名
std::string calculateHash(const std::vector<char> &data) {
  llvm::ArrayRef<uint8_t> dataRef(
      reinterpret_cast<const uint8_t *>(data.data()), data.size());
  uint64_t hash = llvm::xxHash64(dataRef);

  std::stringstream ss;
  ss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return ss.str();
}

/// 确保目录存在，如果不存在则创建
bool ensureDirectoryExists(const std::string &dirPath) {
  std::error_code ec = llvm::sys::fs::create_directories(
      dirPath, true, llvm::sys::fs::owner_all | llvm::sys::fs::group_all);
  return !ec;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// AssetManager Implementation
//===----------------------------------------------------------------------===//

AssetManager::AssetManager() { createTempDir(); }

AssetManager::~AssetManager() { cleanup(); }

void AssetManager::createTempDir() {
  // 创建临时目录
  std::string basePath = "/tmp";
  std::string tempName = "preppipe_asset_" + generateRandomString(8);

  // 使用 SmallString 进行路径拼接
  llvm::SmallString<128> tempPath(basePath);
  llvm::sys::path::append(tempPath, tempName);
  tempDir = tempPath.str();

  if (!ensureDirectoryExists(tempDir)) {
    // 如果创建失败，使用当前目录作为后备
    tempDir = "./preppipe_asset_temp";
    ensureDirectoryExists(tempDir);
  }
}

std::string AssetManager::addAsset(const std::string &path,
                                   const std::vector<char> &data,
                                   const std::string &preferred_name) {
  // 检查是否已经存在该逻辑路径的资源
  if (auto it = assetMap.find(path); it != assetMap.end()) {
    return it->second;
  }

  // 生成唯一文件名
  std::string filename = generateUniqueFilename(preferred_name, data);

  // 构建实际存储路径
  llvm::SmallString<128> actualPath(tempDir);
  llvm::sys::path::append(actualPath, filename);

  // 写入文件
  std::string actualPathStr = actualPath.str().str();
  std::ofstream outFile(actualPathStr.c_str(), std::ios::binary);
  if (!outFile) {
    // 不使用异常，返回空字符串表示失败
    return "";
  }
  outFile.write(data.data(), data.size());
  outFile.close();

  // 添加到映射
  assetMap[path] = actualPathStr;
  return actualPathStr;
}

std::string AssetManager::addExternalAsset(const std::string &path,
                                           const std::string &external_path) {
  // 检查是否已经存在该逻辑路径的资源
  if (auto it = assetMap.find(path); it != assetMap.end()) {
    return it->second;
  }

  // 检查外部资源是否已经被添加过
  if (auto it = externalAssetMap.find(external_path);
      it != externalAssetMap.end()) {
    assetMap[path] = it->second;
    return it->second;
  }

  // 外部资源直接使用原路径
  assetMap[path] = external_path;
  externalAssetMap[external_path] = external_path;
  return external_path;
}

std::optional<std::string>
AssetManager::getAssetPath(const std::string &path) const {
  auto it = assetMap.find(path);
  if (it != assetMap.end()) {
    return it->second;
  }
  return std::nullopt;
}

const std::string &AssetManager::getTempDir() const { return tempDir; }

void AssetManager::cleanup() {
  // 清理临时目录
  if (!tempDir.empty()) {
    // 只清理我们创建的临时目录，不清理外部资源
    if (llvm::sys::fs::exists(tempDir)) {
      llvm::sys::fs::remove_directories(tempDir);
    }
    tempDir.clear();
  }

  // 清空映射
  assetMap.clear();
  externalAssetMap.clear();
  filenameCounter = 0;
}

void AssetManager::print(llvm::raw_ostream &os) const {
  os << "AssetManager Info:\n";
  os << "  Temp Dir: " << tempDir << "\n";
  os << "  Asset Map Size: " << assetMap.size() << "\n";
  os << "  External Asset Map Size: " << externalAssetMap.size() << "\n";

  if (!assetMap.empty()) {
    os << "  Asset Map:\n";
    for (const auto &[path, actualPath] : assetMap) {
      os << "    " << path << " -> " << actualPath << "\n";
    }
  }
}

std::string
AssetManager::generateUniqueFilename(const std::string &preferred_name,
                                     const std::vector<char> &data) {
  std::string filename;

  if (!preferred_name.empty()) {
    // 使用首选名称作为基础
    filename = preferred_name;
  } else {
    // 使用哈希作为文件名基础
    filename = calculateHash(data);
  }

  // 添加计数器确保唯一性
  std::string uniqueFilename = filename;
  while (true) {
    // 使用 SmallString 进行路径拼接
    llvm::SmallString<128> testPath(tempDir);
    llvm::sys::path::append(testPath, uniqueFilename);

    // 转换为 std::string 以便 fs::exists 检查
    std::string testPathStr = testPath.str().str();
    if (!llvm::sys::fs::exists(testPathStr)) {
      break;
    }

    // 如果文件已存在，添加计数器后缀
    filenameCounter++;
    uniqueFilename = filename + "_" + std::to_string(filenameCounter);
  }

  return uniqueFilename;
}
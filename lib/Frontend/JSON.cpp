// SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
// SPDX-License-Identifier: Apache-2.0

#include "preppipe-mlir/Frontend/JSON.h"
#include "preppipe-mlir/Dialect/InputModel/IR/InputModel.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::preppipe;
using namespace mlir::preppipe::frontend;
using namespace mlir::preppipe::InputModel;
using namespace llvm::json;

namespace {

/// Parse a JSON value representing a document element and convert it to MLIR operations.
/// \param builder The MLIR builder to use.
/// \param jsonElement The JSON element to parse.
/// \param block The block to add the operations to.
bool parseElement(OpBuilder &builder, const llvm::json::Value &jsonElement, Block &block) {
  // TODO: Implement parsing of JSON elements into InputModel operations
  return true;
}

/// Parse a JSON value representing a paragraph and convert it to MLIR operations.
/// \param builder The MLIR builder to use.
/// \param jsonParagraph The JSON paragraph to parse.
/// \param block The block to add the operations to.
bool parseParagraph(OpBuilder &builder, const llvm::json::Value &jsonParagraph, Block &block) {
  // TODO: Implement parsing of JSON paragraphs into InputModel operations
  return true;
}

/// Parse a JSON value representing a document file and convert it to an IMDocumentOp.
/// \param builder The MLIR builder to use.
/// \param jsonFile The JSON file to parse.
/// \param module The module to add the IMDocumentOp to.
bool parseFile(OpBuilder &builder, const llvm::json::Value &jsonFile, ModuleOp module) {
  // Extract file name and path from JSON
  std::string name, path;
  if (const llvm::json::Object *fileObj = jsonFile.getAsObject()) {
    if (const llvm::json::Value *nameValue = fileObj->get("name")) {
      if (auto strRef = nameValue->getAsString()) {
        name = strRef->str();
      }
    }
    if (const llvm::json::Value *pathValue = fileObj->get("path")) {
      if (auto strRef = pathValue->getAsString()) {
        path = strRef->str();
      }
    }
  }

  // Create a new IMDocumentOp
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());
  auto documentOp = builder.create<IMDocumentOp>(builder.getUnknownLoc());

  // Get the body region of the document
  auto &bodyRegion = documentOp.getBody();
  auto *bodyBlock = builder.createBlock(&bodyRegion);
  builder.setInsertionPointToEnd(bodyBlock);

  // Parse the document body
  if (const llvm::json::Object *fileObj = jsonFile.getAsObject()) {
    if (const llvm::json::Value *bodyValue = fileObj->get("body")) {
      if (const llvm::json::Array *bodyArray = bodyValue->getAsArray()) {
        for (const llvm::json::Value &paragraphValue : *bodyArray) {
          if (!parseParagraph(builder, paragraphValue, *bodyBlock)) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

} // anonymous namespace

bool frontend::readJSONFiles(MLIRContext &context, ModuleOp module, 
                             const std::vector<std::string> &filenames) {
  OpBuilder builder(&context);

  for (const std::string &filename : filenames) {
    // Read the JSON file
    auto bufferOrError = llvm::MemoryBuffer::getFile(filename);
    if (std::error_code ec = bufferOrError.getError()) {
      llvm::errs() << "Error opening file " << filename << ": " << ec.message() << "\n";
      return false;
    }
    const llvm::MemoryBuffer &buffer = **bufferOrError;

    // Parse the JSON file
    std::string jsonStr = buffer.getBuffer().str();
    llvm::Expected<llvm::json::Value> jsonValue = llvm::json::parse(jsonStr);
    if (!jsonValue) {
      llvm::errs() << "Error parsing JSON file " << filename << ": " 
                  << llvm::toString(jsonValue.takeError()) << "\n";
      return false;
    }

    // Parse the JSON structure
    if (const llvm::json::Object *rootObj = jsonValue->getAsObject()) {
      if (const llvm::json::Value *filesValue = rootObj->get("files")) {
        if (const llvm::json::Array *filesArray = filesValue->getAsArray()) {
          for (const llvm::json::Value &fileValue : *filesArray) {
            if (!parseFile(builder, fileValue, module)) {
              llvm::errs() << "Error parsing file content in " << filename << "\n";
              return false;
            }
          }
        }
      }
    } else {
      llvm::errs() << "Error: JSON file " << filename << " does not have a root object\n";
      return false;
    }
  }

  return true;
}

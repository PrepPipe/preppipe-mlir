// SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
// SPDX-License-Identifier: Apache-2.0

#include "preppipe-mlir/Frontend/JSON.h"
#include "preppipe-mlir/Dialect/InputModel/IR/InputModel.h"
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipe.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::preppipe;
using namespace mlir::preppipe::frontend;
using namespace mlir::preppipe::InputModel;
using namespace mlir::preppipe::PrepPipe;
using JSONValue = llvm::json::Value;

namespace {

/// LocationContext tracks file path and provides location creation utilities.
struct LocationContext {
  std::string filePath;

  LocationContext(const std::string &path) : filePath(path) {}

  /// Create a FileLineColLoc from line and column numbers.
  /// Line and column are 1-indexed for display.
  Location getLocation(MLIRContext &context, unsigned line, unsigned column) {
    return FileLineColLoc::get(&context, filePath, line, column);
  }

  /// Compute the length of a text element's content.
  /// Returns the UTF-8 string length of the content.
  unsigned computeTextLength(const JSONValue &textElement) {
    if (const llvm::json::Object *textObj = textElement.getAsObject()) {
      const JSONValue *contentValue = textObj->get("content");
      if (contentValue) {
        if (auto contentStr = contentValue->getAsString()) {
          return contentStr->str().length();
        }
      }
    }
    return 0;
  }
};


/// Parse a text element with optional formatting.
/// Returns the content attributes to be used in IMElementOp.
/// Returns the length of the text content via length parameter.
bool parseTextElement(OpBuilder &builder, Location loc,
                     const llvm::json::Object *textObj,
                     SmallVector<Attribute> &contentAttrs,
                     unsigned &length) {
  if (!textObj) return false;

  const llvm::json::Value *contentValue = textObj->get("content");
  if (!contentValue) return false;

  auto contentStr = contentValue->getAsString();
  if (!contentStr) return false;

  // Store the length of the content
  length = contentStr->str().length();

  // Check for formatting
  const llvm::json::Value *formatValue = textObj->get("format");
  bool hasFormatting = formatValue != nullptr;

  if (hasFormatting) {
    // Create a styled text fragment using TextAttr
    SmallVector<NamedAttribute> styleAttrs;

    if (const llvm::json::Object *formatObj = formatValue->getAsObject()) {
      if (formatObj->get("bold")) {
        styleAttrs.push_back(builder.getNamedAttr("bold", builder.getBoolAttr(true)));
      }
      if (formatObj->get("italic")) {
        styleAttrs.push_back(builder.getNamedAttr("italic", builder.getBoolAttr(true)));
      }
      if (const llvm::json::Value *colorValue = formatObj->get("color")) {
        if (auto colorStr = colorValue->getAsString()) {
          styleAttrs.push_back(builder.getNamedAttr("color",
            builder.getStringAttr(colorStr->str())));
        }
      }
      if (const llvm::json::Value *bgColorValue = formatObj->get("backgroundColor")) {
        if (auto bgColorStr = bgColorValue->getAsString()) {
          styleAttrs.push_back(builder.getNamedAttr("backgroundColor",
            builder.getStringAttr(bgColorStr->str())));
        }
      }
      // Note: Size attribute is not in the JSON format currently, but we support it
      if (const llvm::json::Value *sizeValue = formatObj->get("size")) {
        if (auto sizeInt = sizeValue->getAsInteger()) {
          styleAttrs.push_back(builder.getNamedAttr("size",
            builder.getI64IntegerAttr(*sizeInt)));
        }
      }
    }

    // Create TextAttr with style dictionary
    auto styleDict = builder.getDictionaryAttr(styleAttrs);
    auto stringAttr = builder.getStringAttr(contentStr->str());
    auto textAttr = PrepPipe::TextAttr::get(builder.getContext(), stringAttr, styleDict);
    contentAttrs.push_back(textAttr);
  } else {
    // Plain text - use TextAttr without style
    auto stringAttr = builder.getStringAttr(contentStr->str());
    auto textAttr = PrepPipe::TextAttr::get(stringAttr);
    contentAttrs.push_back(textAttr);
  }

  return true;
}

/// Parse an asset reference element.
/// Returns the length (always 1 for non-text elements) via length parameter.
bool parseAssetRefElement(OpBuilder &builder, Location loc,
                          const llvm::json::Object *assetObj,
                          SmallVector<Attribute> &contentAttrs,
                          unsigned &length) {
  if (!assetObj) return false;

  const llvm::json::Value *refValue = assetObj->get("ref");
  if (!refValue) return false;

  auto refStr = refValue->getAsString();
  if (!refStr) return false;

  // Non-text elements are counted as 1 character wide
  length = 1;

  // Create an asset reference as a TextAttr
  // TODO: Create proper asset reference attribute/type later
  auto stringAttr = builder.getStringAttr(refStr->str());
  auto textAttr = PrepPipe::TextAttr::get(stringAttr);
  contentAttrs.push_back(textAttr);

  return true;
}

/// Parse a JSON value representing a document element and convert it to MLIR operations.
/// \param builder The MLIR builder to use.
/// \param locCtx The location context for creating locations.
/// \param jsonElement The JSON element to parse.
/// \param block The block to add the operations to.
/// \param line The line number (1-indexed).
/// \param column The column number (1-indexed), updated to reflect element length.
/// \return true on success, false on failure.
/// The column parameter is updated to point after this element.
bool parseElement(OpBuilder &builder, LocationContext &locCtx,
                  const JSONValue &jsonElement, Block &block,
                  unsigned line, unsigned &column) {
  builder.setInsertionPointToEnd(&block);

  if (const llvm::json::Object *elementObj = jsonElement.getAsObject()) {
    const JSONValue *typeValue = elementObj->get("type");
    if (!typeValue) return false;

    auto typeStr = typeValue->getAsString();
    if (!typeStr) return false;

    if (*typeStr == "text") {
      // Create location for this element
      Location loc = locCtx.getLocation(*builder.getContext(), line, column);

      SmallVector<Attribute> contentAttrs;
      unsigned elementLength = 0;
      if (!parseTextElement(builder, loc, elementObj, contentAttrs, elementLength)) {
        return false;
      }
      if (!contentAttrs.empty()) {
        auto contentArray = builder.getArrayAttr(contentAttrs);
        builder.create<IMElementOp>(loc, contentArray);
      }
      // Update column: element starts at current column, ends at column + length
      column += elementLength;
      return true;
    } else if (*typeStr == "assetref") {
      // Create location for this element
      Location loc = locCtx.getLocation(*builder.getContext(), line, column);

      SmallVector<Attribute> contentAttrs;
      unsigned elementLength = 0;
      if (!parseAssetRefElement(builder, loc, elementObj, contentAttrs, elementLength)) {
        return false;
      }
      if (!contentAttrs.empty()) {
        auto contentArray = builder.getArrayAttr(contentAttrs);
        builder.create<IMElementOp>(loc, contentArray);
      }
      // Update column: non-text elements are 1 character wide
      column += elementLength;
      return true;
    }
  }

  return false;
}

/// Parse a JSON value representing a paragraph and convert it to MLIR operations.
/// \param builder The MLIR builder to use.
/// \param locCtx The location context for creating locations.
/// \param jsonParagraph The JSON paragraph to parse.
/// \param block The block to add the operations to.
/// \param line The line number (1-indexed), updated if list is encountered.
/// \param column The column number (1-indexed), updated to reflect element lengths.
/// \return true on success, false on failure.
/// If a list is encountered, the line parameter is updated to reflect the final line after all list items.
bool parseParagraph(OpBuilder &builder, LocationContext &locCtx,
                   const JSONValue &jsonParagraph, Block &block,
                   unsigned &line, unsigned &column) {
  builder.setInsertionPointToEnd(&block);

  if (const llvm::json::Array *paragraphArray = jsonParagraph.getAsArray()) {
    // Check if this is a special block (centered, codeblock) or list
    if (paragraphArray->size() == 1) {
      const JSONValue &firstElement = (*paragraphArray)[0];
      if (const llvm::json::Object *elementObj = firstElement.getAsObject()) {
        const JSONValue *typeValue = elementObj->get("type");
        if (typeValue) {
          if (auto typeStr = typeValue->getAsString()) {
            if (*typeStr == "centered") {
              // Parse centered special block
              Location loc = locCtx.getLocation(*builder.getContext(), line, column);
              const JSONValue *contentValue = elementObj->get("content");
              if (contentValue) {
                if (auto contentStr = contentValue->getAsString()) {
                  SmallVector<Attribute> contentAttrs;
                  auto stringAttr = builder.getStringAttr(contentStr->str());
                  auto textAttr = PrepPipe::TextAttr::get(stringAttr);
                  contentAttrs.push_back(textAttr);
                  auto contentArray = builder.getArrayAttr(contentAttrs);
                  builder.create<IMSpecialBlockOp>(loc,
                    builder.getStringAttr("centered"), contentArray);
                  // Special blocks are treated as 1 character wide
                  column += 1;
                  return true;
                }
              }
            } else if (*typeStr == "codeblock") {
              // Parse codeblock special block
              Location loc = locCtx.getLocation(*builder.getContext(), line, column);
              const JSONValue *contentValue = elementObj->get("content");
              if (contentValue) {
                if (auto contentStr = contentValue->getAsString()) {
                  SmallVector<Attribute> contentAttrs;
                  auto stringAttr = builder.getStringAttr(contentStr->str());
                  auto textAttr = PrepPipe::TextAttr::get(stringAttr);
                  contentAttrs.push_back(textAttr);
                  auto contentArray = builder.getArrayAttr(contentAttrs);
                  builder.create<IMSpecialBlockOp>(loc,
                    builder.getStringAttr("bg_highlight"), contentArray);
                  // Special blocks are treated as 1 character wide
                  column += 1;
                  return true;
                }
              }
            } else if (*typeStr == "list") {
              // Parse list
              // List operation uses the current line number, starts at column 1
              Location listLoc = locCtx.getLocation(*builder.getContext(), line, 1);
              const JSONValue *itemsValue = elementObj->get("items");
              if (itemsValue) {
                if (const llvm::json::Array *itemsArray = itemsValue->getAsArray()) {
                  // Determine if numbered (check first item for numbering pattern)
                  // For now, assume bulleted lists (is_numbered = false)
                  bool isNumbered = false;
                  auto listOp = builder.create<IMListOp>(listLoc, builder.getBoolAttr(isNumbered));

                  // Get the body region of the list (single block)
                  auto &listBody = listOp.getBody();
                  auto *listBodyBlock = builder.createBlock(&listBody);
                  builder.setInsertionPointToEnd(listBodyBlock);

                  // Parse each list item
                  // First list item shares the same line as the list operation
                  // Subsequent items increment the line number
                  unsigned currentLine = line;
                  for (size_t i = 0; i < itemsArray->size(); ++i) {
                    const JSONValue &itemValue = (*itemsArray)[i];
                    if (const llvm::json::Array *itemArray = itemValue.getAsArray()) {
                      // First item uses same line as list, subsequent items increment
                      if (i > 0) {
                        currentLine++;
                      }

                      // Create IMListItemOp inside the list's body block
                      Location itemLoc = locCtx.getLocation(*builder.getContext(), currentLine, 1);
                      auto listItemOp = builder.create<IMListItemOp>(itemLoc);

                      // Get the body region of the list item (single block)
                      auto &listItemBody = listItemOp.getBody();
                      auto *listItemBodyBlock = builder.createBlock(&listItemBody);
                      builder.setInsertionPointToEnd(listItemBodyBlock);

                      // Parse each paragraph in the item (all in the same block)
                      // Each paragraph in a list item starts at column 1
                      // If a paragraph contains a nested list, the line number will be updated
                      unsigned itemColumn = 1;
                      unsigned itemLine = currentLine; // Track line for this item
                      for (const JSONValue &paragraphValue : *itemArray) {
                        if (const llvm::json::Array *paragraphArray = paragraphValue.getAsArray()) {
                          // Parse the paragraph elements into the list item's body block
                          // Note: paragraphs inside list items don't increment line numbers by themselves
                          // But if a paragraph contains a nested list, the line number will be updated
                          if (!parseParagraph(builder, locCtx, paragraphValue, *listItemBodyBlock, itemLine, itemColumn)) {
                            return false;
                          }
                          // Update currentLine if the paragraph contained a nested list
                          if (itemLine > currentLine) {
                            currentLine = itemLine;
                          }
                        }
                      }

                      // Restore insertion point to the list's body block for next item
                      builder.setInsertionPointToEnd(listBodyBlock);
                    }
                  }

                  // Update the line parameter to reflect the final line after all list items
                  line = currentLine;

                  // Restore insertion point to the original block
                  builder.setInsertionPointToEnd(&block);
                  // List is treated as 1 character wide for column calculation
                  column += 1;
                  return true;
                }
              }
            }
          }
        }
      }
    }

    // Regular paragraph: parse each element
    // Elements are processed left-to-right, column accumulates
    for (const JSONValue &elementValue : *paragraphArray) {
      if (!parseElement(builder, locCtx, elementValue, block, line, column)) {
        // Skip elements we can't parse for now
        continue;
      }
    }
    return true;
  }

  // Empty paragraph
  return true;
}

/// Parse a JSON value representing a document file and convert it to an IMDocumentOp.
/// \param builder The MLIR builder to use.
/// \param jsonFile The JSON file to parse.
/// \param module The module to add the IMDocumentOp to.
bool parseFile(OpBuilder &builder, const JSONValue &jsonFile, ModuleOp module) {
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

  // Create LocationContext with the file path
  // Use path if available, otherwise use name, otherwise use a default
  std::string filePath = path.empty() ? (name.empty() ? "<unknown>" : name) : path;
  LocationContext locCtx(filePath);

  // Create a new IMDocumentOp
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());
  // Document operation uses unknown location (it's a container)
  auto documentOp = builder.create<IMDocumentOp>(builder.getUnknownLoc());

  // Get the body region of the document
  auto &bodyRegion = documentOp.getBody();
  auto *bodyBlock = builder.createBlock(&bodyRegion);
  builder.setInsertionPointToEnd(bodyBlock);

  // Parse the document body
  // Line numbers correspond to indices in the body array (0-indexed internally, 1-indexed for display)
  if (const llvm::json::Object *fileObj = jsonFile.getAsObject()) {
    if (const JSONValue *bodyValue = fileObj->get("body")) {
      if (const llvm::json::Array *bodyArray = bodyValue->getAsArray()) {
        // Track the current line number, starting from 1
        // Each paragraph in the body array corresponds to a line number
        // If a paragraph contains a list, the list items consume additional lines
        unsigned currentLine = 1;
        for (size_t i = 0; i < bodyArray->size(); ++i) {
          const JSONValue &paragraphValue = (*bodyArray)[i];
          // Set the line number for this paragraph
          // For the first paragraph, it's line 1
          // For subsequent paragraphs, currentLine has been updated by the previous paragraph
          // (either incremented if it was a regular paragraph, or set to the final line
          // after a list if it contained a list)
          unsigned paragraphLine = currentLine;
          // Each paragraph starts at column 1
          unsigned column = 1;
          if (!parseParagraph(builder, locCtx, paragraphValue, *bodyBlock, paragraphLine, column)) {
            return false;
          }
          // After parsing, paragraphLine will be updated if the paragraph contained a list
          // Update currentLine to point to the line after this paragraph
          // If it was a regular paragraph, paragraphLine is still the same, so we increment
          // If it contained a list, paragraphLine was updated to the final line of the list
          currentLine = paragraphLine + 1;
        }
      }
    }
  }

  return true;
}

} // anonymous namespace

/// Parse embedded assets from JSON and create asset declarations.
/// \param builder The MLIR builder to use.
/// \param embeddedArray The JSON array of embedded assets.
/// \param module The module to add asset declarations to.
bool parseEmbeddedAssets(OpBuilder &builder, const llvm::json::Array *embeddedArray, ModuleOp module) {
  builder.setInsertionPointToEnd(&module.getBodyRegion().front());

  for (const JSONValue &assetValue : *embeddedArray) {
    if (const llvm::json::Object *assetObj = assetValue.getAsObject()) {
      // Extract asset information
      std::string srcref, extracted;
      if (const JSONValue *srcrefValue = assetObj->get("srcref")) {
        if (auto strRef = srcrefValue->getAsString()) {
          srcref = strRef->str();
        }
      }
      if (const JSONValue *extractedValue = assetObj->get("extracted")) {
        if (auto strRef = extractedValue->getAsString()) {
          extracted = strRef->str();
        }
      }

      // Extract size and bbox if available
      SmallVector<NamedAttribute> imageAttrs;
      if (const JSONValue *sizeValue = assetObj->get("size")) {
        if (const llvm::json::Array *sizeArray = sizeValue->getAsArray()) {
          if (sizeArray->size() >= 2) {
            int64_t width = 0, height = 0;
            if (auto widthVal = (*sizeArray)[0].getAsInteger()) {
              width = *widthVal;
            }
            if (auto heightVal = (*sizeArray)[1].getAsInteger()) {
              height = *heightVal;
            }
            imageAttrs.push_back(builder.getNamedAttr("width", builder.getI64IntegerAttr(width)));
            imageAttrs.push_back(builder.getNamedAttr("height", builder.getI64IntegerAttr(height)));
          }
        }
      }
      if (const JSONValue *bboxValue = assetObj->get("bbox")) {
        if (const llvm::json::Array *bboxArray = bboxValue->getAsArray()) {
          if (bboxArray->size() >= 4) {
            int64_t x = 0, y = 0, w = 0, h = 0;
            if (auto xVal = (*bboxArray)[0].getAsInteger()) x = *xVal;
            if (auto yVal = (*bboxArray)[1].getAsInteger()) y = *yVal;
            if (auto wVal = (*bboxArray)[2].getAsInteger()) w = *wVal;
            if (auto hVal = (*bboxArray)[3].getAsInteger()) h = *hVal;

            // Create bounding box attribute using PrepPipe dialect
            // TODO: Use PrepPipe::BoundingBoxAttr once properly integrated
            imageAttrs.push_back(builder.getNamedAttr("bbox_x", builder.getI64IntegerAttr(x)));
            imageAttrs.push_back(builder.getNamedAttr("bbox_y", builder.getI64IntegerAttr(y)));
            imageAttrs.push_back(builder.getNamedAttr("bbox_width", builder.getI64IntegerAttr(w)));
            imageAttrs.push_back(builder.getNamedAttr("bbox_height", builder.getI64IntegerAttr(h)));
          }
        }
      }

      // Create asset declaration
      // Use the extracted filename as the symbol name (without extension)
      std::string symName = extracted;
      size_t dotPos = symName.find_last_of('.');
      if (dotPos != std::string::npos) {
        symName = symName.substr(0, dotPos);
      }

      // TODO: Create proper ImageAssetOp once the operation is fully defined
      // For now, we'll skip creating the asset declaration as it requires
      // the PrepPipe asset operations to be fully implemented
      // builder.create<PrepPipe::ImageAssetOp>(loc, symName, srcref, imageAttrsDict);
      (void)symName; // Suppress unused variable warning
      (void)srcref;  // Suppress unused variable warning
    }
  }

  return true;
}

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
      // Parse embedded assets first
      if (const JSONValue *embeddedValue = rootObj->get("embedded")) {
        if (const llvm::json::Array *embeddedArray = embeddedValue->getAsArray()) {
          parseEmbeddedAssets(builder, embeddedArray, module);
        }
      }

      // Parse files
      if (const JSONValue *filesValue = rootObj->get("files")) {
        if (const llvm::json::Array *filesArray = filesValue->getAsArray()) {
          for (const JSONValue &fileValue : *filesArray) {
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

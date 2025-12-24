#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipe.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::preppipe;
using namespace mlir::preppipe::PrepPipe;

#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeBase.cpp.inc"

#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TextAttr custom printer and parser
//===----------------------------------------------------------------------===//

/// Print a UTF-8 string with minimal escaping (only escape quotes and backslashes)
static void printUTF8String(raw_ostream &os, StringRef str) {
  os << '"';
  for (unsigned char c : str) {
    if (c == '"' || c == '\\') {
      os << '\\' << c;
    } else if (c >= 0x20 && c < 0x7F) {
      // Printable ASCII
      os << c;
    } else {
      // UTF-8 bytes (including CJK characters) - print as-is
      os << c;
    }
  }
  os << '"';
}

Attribute TextAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // Parse the string content
  std::string contentStr;
  if (parser.parseString(&contentStr))
    return {};

  StringAttr content = StringAttr::get(parser.getContext(), contentStr);

  // Parse optional style dictionary
  DictionaryAttr style;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(style))
      return {};
  }

  if (parser.parseGreater())
    return {};

  return TextAttr::get(parser.getContext(), content, style);
}

void TextAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printUTF8String(printer.getStream(), getContent().getValue());
  if (getStyle()) {
    printer << ", ";
    printer.printAttributeWithoutType(getStyle());
  }
  printer << ">";
}

void PrepPipeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeOps.cpp.inc"

      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.cpp.inc"

      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeAttrs.cpp.inc"
      >();
}

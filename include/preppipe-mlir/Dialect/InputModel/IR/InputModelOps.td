// SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef PREPPIPE_INPUT_MODEL_OPS
#define PREPPIPE_INPUT_MODEL_OPS

include "preppipe-mlir/Dialect/InputModel/IR/InputModelBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for all InputModel operations
class InputModel_Op<string mnemonic, list<Trait> traits = []> :
  Op<InputModel_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===
// IMElementOp
//===----------------------------------------------------------------------===

def IMElementOp : InputModel_Op<"element", []> {
  let summary = "Represents content elements (text or images)";
  let description = [{
    InputModel element operation that combines content values with position information.
    For text content, if part of the text has special formatting (like bold or background color),
    the content may include multiple values.
    For non-text content, each IMElementOp will only contain a single value.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===
// IMErrorElementOp
//===----------------------------------------------------------------------===

def IMErrorElementOp : InputModel_Op<"error_element", []> {
  let summary = "Represents errors during content reading";
  let description = [{
    Represents errors that occurred during reading. The content operand contains the
    best representation of the content, and the error_code and error_msg attributes
    describe the error.
  }];

  let arguments = (ins
    StrAttr:$error_code,
    OptionalAttr<StrAttr>:$error_msg
  );
  let results = (outs);

  let assemblyFormat = "`errors` $error_code ($error_msg^)? attr-dict";
}

//===----------------------------------------------------------------------===
// IMSpecialBlockOp
//===----------------------------------------------------------------------===

def IMSpecialBlockOp : InputModel_Op<"special_block", []> {
  let summary = "Represents special blocks of content";
  let description = [{
    Used to embed unformatted content within text, or to describe other special
    formatted text, typically used for embedding backend instructions.
  }];

  let arguments = (ins
    StrAttr:$reason
  );
  let results = (outs);

  let assemblyFormat = "`reason` $reason attr-dict";
}

//===----------------------------------------------------------------------===
// IMFrameOp
//===----------------------------------------------------------------------===

def IMFrameOp : InputModel_Op<"frame", [SingleBlock]> {
  let summary = "Represents a top-level structure (document or text box)";
  let description = [{
    Represents the top-level structure of a document or text box.
    The body region contains the content blocks.
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$body attr-dict";
}

//===----------------------------------------------------------------------===
// IMListOp
//===----------------------------------------------------------------------===

def IMListOp : InputModel_Op<"list", []> {
  let summary = "Represents a list (bullet point or numbered)";
  let description = [{
    Represents a list with bullet points or numbers. Each list item is a separate region.
  }];

  let arguments = (ins
    BoolAttr:$is_numbered
  );
  let results = (outs);

  let regions = (region AnyRegion:$items);

  let assemblyFormat = "$items attr-dict";
}

//===----------------------------------------------------------------------===
// IMTableOp
//===----------------------------------------------------------------------===

def IMTableOp : InputModel_Op<"table", []> {
  let summary = "Represents a table";
  let description = [{
    Represents a table with rows and columns. Merged cells are split into sub-cells
    with identical content.
  }];

  let arguments = (ins
    I32Attr:$rowcount,
    I32Attr:$columncount
  );
  let results = (outs);

  let assemblyFormat = "$rowcount `x` $columncount attr-dict";
}

//===----------------------------------------------------------------------===
// IMDocumentOp
//===----------------------------------------------------------------------===

def IMDocumentOp : InputModel_Op<"document", [SingleBlock]> {
  let summary = "Represents a complete document";
  let description = [{
    Represents a complete document with a body region containing the content blocks.
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = "$body attr-dict";
}

#endif // PREPPIPE_INPUT_MODEL_OPS

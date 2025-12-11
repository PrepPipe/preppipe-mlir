// SPDX-FileCopyrightText: 2025 PrepPipe's Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef PREPPIPE_INPUT_MODEL_BASE
#define PREPPIPE_INPUT_MODEL_BASE

include "mlir/IR/DialectBase.td"

def InputModel_Dialect : Dialect {
  let name = "inputmodel";
  let cppNamespace = "::mlir::preppipe::InputModel";
  let summary = "Dialect for representing input document models";
  let description = [{
    The InputModel dialect provides operations for representing structured input
    documents, including text, images, and other content elements.
  }];

}

#endif // PREPPIPE_INPUT_MODEL_BASE

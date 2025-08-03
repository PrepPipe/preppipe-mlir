// RUN: preppipe-mlir-opt %s | FileCheck %s
module {
  // CHECK: preppipe.metadata
  preppipe.metadata <"comment"> "Yes"
  // CHECK: preppipe.metadata
  preppipe.metadata <"err"> "No"
}

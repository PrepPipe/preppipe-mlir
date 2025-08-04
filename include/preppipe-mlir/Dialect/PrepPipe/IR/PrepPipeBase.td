#ifndef PREPPIPE_PREPPIPE_BASE_TD
#define PREPPIPE_PREPPIPE_BASE_TD

include "mlir/IR/OpBase.td"

def PrepPipe_Dialect : Dialect {
  let name = "preppipe";
  let cppNamespace = "::mlir::preppipe::PrepPipe";
  let summary = "Top-level dialect for PrepPipe's visual novel lowering pipeline";
  let description = [{
    Top-level dialect for PrepPipe Compiler's visual novel lowering pipeline.

    This dialect contains operations, attributes, etc. that are used throughout
    the PrepPipe lowering pipeline. They are generally not converted during the
    lowering process. Operations, etc. that are specific to one particular
    abstraction level are defined in the dialects for those levels.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class PrepPipe_Op<string mnemonic, list<Trait> traits = []>
    : Op<PrepPipe_Dialect, mnemonic, traits> {
}

class PrepPipe_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<PrepPipe_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}



#endif // PREPPIPE_PREPPIPE_BASE_TD

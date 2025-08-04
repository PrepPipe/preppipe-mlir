#ifndef PREPPIPE_PREPPIPE_TYPES
#define PREPPIPE_PREPPIPE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeBase.td"

class PrepPipe_Type<string name, string typeMnemonic,
                 string baseCppClass = "::mlir::Type">
    : TypeDef<PrepPipe_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}

class PrepPipe_TypeWithContainedType<string name, string typeMnemonic> : PrepPipe_Type<name, typeMnemonic> {
  let parameters = (ins "::mlir::Type":$containedType);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$containedType), [{
      return Base::get(containedType.getContext(), containedType);
    }]>
  ];
}

def PPVoidType : PrepPipe_Type<"Void", "void"> {
  let summary = "Void type for PrepPipe";
}

// Specific asset types
def ImageType : PrepPipe_Type<"Image", "image"> {
  let summary = "General image type in PrepPipe";
  let description = [{
    This type represents an image in the PrepPipe pipeline.
    The image can be an expression (e.g., fill) or a reference to an existing image asset.
  }];
}

def AudioType : PrepPipe_Type<"Audio", "audio"> {
  let summary = "General audio type in PrepPipe";
  let description = [{
    This type represents an audio asset in the PrepPipe pipeline.
    The audio can be an expression or a reference to an existing audio asset.
  }];
}

#endif // PREPPIPE_PREPPIPE_TYPES

#ifndef PREPPIPE_PREPPIPE_TYPES
#define PREPPIPE_PREPPIPE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeDialect.td"

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

def PPVoidType : PrepPipe_Type<"Void", "pp.void"> {
  let summary = "Void type for PrepPipe";
  let description = [{
    This type represents a void type in the PrepPipe dialect.
    该类型表示 PrepPipe 方言中的空类型。
  }];
}

#endif // PREPPIPE_PREPPIPE_TYPES

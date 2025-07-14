#ifndef PREPPIPE_VNMODEL_TYPES
#define PREPPIPE_VNMODEL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "preppipe-mlir/Dialect/VNModel/IR/VNModelDialect.td"

class VNModel_Type<string name, string typeMnemonic,
                 string baseCppClass = "::mlir::Type">
    : TypeDef<VNModel_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}

class VNModel_TypeWithContainedType<string name, string typeMnemonic> : VNModel_Type<name, typeMnemonic> {
  let parameters = (ins "::mlir::Type":$containedType);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$containedType), [{
      return Base::get(containedType.getContext(), containedType);
    }]>
  ];
}

#endif // PREPPIPE_VNMODEL_TYPES

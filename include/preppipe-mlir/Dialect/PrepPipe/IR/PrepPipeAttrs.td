#ifndef PREPPIPE_PREPPIPE_ATTRS_TD
#define PREPPIPE_PREPPIPE_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeBase.td"

def BoundingBoxAttr : PrepPipe_Attr<"BoundingBox", "bbox"> {
  let parameters = (ins
    "int64_t":$x,
    "int64_t":$y,
    "int64_t":$width,
    "int64_t":$height
  );

  let assemblyFormat = "`<` $x `,` $y `,` $width `,` $height `>`";
}
// TODO attributes (bbox, device type, etc.)

#endif // PREPPIPE_PREPPIPE_ATTRS_TD

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

// TextAttr - contains a string and optional text attributes
// The style dictionary contains text styling information:
// - "bold": bool attribute
// - "italic": bool attribute
// - "size": int64_t attribute (size change, 0 is no change)
// - "color": string attribute (foreground color, e.g., "#ff0000")
// - "backgroundColor": string attribute (background/highlight color)
def TextAttr : PrepPipe_Attr<"Text", "text"> {
  let summary = "Text attribute with optional styling";
  let description = [{
    Text attribute that contains a UTF-8 string and optional text styling attributes.
    The style dictionary contains text styling information with the following keys:
    - "bold": bool value
    - "italic": bool value
    - "size": int64_t value (size change, 0 is no change)
    - "color": string value (foreground color code, e.g., "#ff0000")
    - "backgroundColor": string value (background/highlight color code)
  }];

  let parameters = (ins
    "::mlir::StringAttr":$content,
    OptionalParameter<"::mlir::DictionaryAttr">:$style
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$content,
                                        CArg<"::mlir::DictionaryAttr", "{}">:$style), [{
      return $_get(content.getContext(), content, style);
    }]>
  ];
}

#endif // PREPPIPE_PREPPIPE_ATTRS_TD
